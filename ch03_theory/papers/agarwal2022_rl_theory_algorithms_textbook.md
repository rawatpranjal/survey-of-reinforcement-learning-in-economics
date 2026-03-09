## Reinforcement Learning: Theory and Algorithms

Alekh Agarwal Nan Jiang Sham M. Kakade Wen Sun January 31, 2022

## WORKING DRAFT:

Please email bookrltheory@gmail.com with any typos or errors you find. We appreciate it!

## Contents

| 1 Fundamentals              | 1 Fundamentals                                           | 1 Fundamentals                                                        | 1 Fundamentals                                                        |   3 |
|-----------------------------|----------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|-----|
| 1 Markov Decision Processes | 1 Markov Decision Processes                              | 1 Markov Decision Processes                                           | 1 Markov Decision Processes                                           |   5 |
| 1.1                         |                                                          | Discounted (Infinite-Horizon) Markov Decision Processes . . .         | Discounted (Infinite-Horizon) Markov Decision Processes . . .         |   5 |
|                             |                                                          | 1.1.1                                                                 | The objective, policies, and values . . . . . . . . . . .             |   5 |
|                             |                                                          | 1.1.2                                                                 | Bellman Consistency Equations for Stationary Policies                 |   7 |
|                             |                                                          | 1.1.3                                                                 | Bellman Optimality Equations . . . . . . . . . . . . .                |   8 |
| 1.2                         | Finite-Horizon Markov Decision Processes . . . . . . . . | Finite-Horizon Markov Decision Processes . . . . . . . .              | . .                                                                   |  11 |
| 1.3                         | Computational Complexity . . . . . . . . .               | Computational Complexity . . . . . . . . .                            | . . . . . . . . . .                                                   |  12 |
|                             |                                                          | 1.3.1                                                                 | Value Iteration . . . . . . . . . . . . . . . . . . . . .             |  12 |
|                             |                                                          | 1.3.2                                                                 | Policy Iteration . . . . . . . . . . . . . . . . . . . . .            |  14 |
|                             |                                                          | 1.3.3                                                                 | Value Iteration for Finite Horizon MDPs . . . . . . . .               |  16 |
|                             |                                                          | 1.3.4                                                                 | The Linear Programming Approach . . . . . . . . . .                   |  16 |
|                             | 1.4                                                      | Sample Complexity and Sampling Models .                               | . . . . . . . . . .                                                   |  18 |
|                             | 1.5                                                      | Bonus: Advantages and The Performance Difference Lemma . . . .        | Bonus: Advantages and The Performance Difference Lemma . . . .        |  18 |
|                             | 1.6                                                      | Bibliographic Remarks and Further Reading .                           | . . . . . . . . .                                                     |  20 |
| 2                           | Sample Complexity with a Generative Model                | Sample Complexity with a Generative Model                             | Sample Complexity with a Generative Model                             |  21 |
|                             | 2.1                                                      | Warmup: a naive model-based approach                                  | . . . . . . . . . . . .                                               |  21 |
|                             | 2.2                                                      | Sublinear Sample Complexity                                           | . . . . . . . . . . . . . . . . .                                     |  23 |
|                             | 2.3                                                      | Minmax Optimal Sample Complexity (and the Model Based Approach) . . . | Minmax Optimal Sample Complexity (and the Model Based Approach) . . . |  24 |
|                             |                                                          | 2.3.1                                                                 | The Discounted Case . . . . . . . . . . . . . . . . . .               |  24 |
|                             |                                                          | 2.3.2                                                                 | Finite Horizon Setting . . . . . . . . . . . . . . . . .              |  25 |
| 2.4                         | Analysis . . . . . . . . . . . . . . . . . . . . . .     | Analysis . . . . . . . . . . . . . . . . . . . . . .                  | . . . . . . .                                                         |  26 |
|                             | 2.4.1 Variance Lemmas .                                  | 2.4.1 Variance Lemmas .                                               | . . . . . . . . . . . . . . . . . . .                                 |  26 |

|                                        | 2.4.2 Completing the proof                                    | . . . . . . . . . . . . . . . . . . . . . .                                 | 28   |
|----------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------|------|
| 2.5                                    | Scalings and Effective Horizon Dependencies . . . . . . . . . | . . . .                                                                     | 29   |
| 2.6                                    | Bibliographic Remarks and Further Readings . . . .            | . . . . . . . . .                                                           | 29   |
| 3 Linear Bellman Completeness          | 3 Linear Bellman Completeness                                 | 3 Linear Bellman Completeness                                               | 31   |
| 3.1                                    | The Linear Bellman Completeness Condition . .                 | . . . . . . . . . . .                                                       | 31   |
| 3.2                                    | The LSVI Algorithm . . .                                      | . . . . . . . . . . . . . . . . . . . . . . .                               | 32   |
| 3.3                                    | LSVI with D-Optimal Design .                                  | . . . . . . . . . . . . . . . . . . . . .                                   | 32   |
|                                        | 3.3.1                                                         | D-Optimal Design . . . . . . . . . . . . . . . . . . . . . . .              | 32   |
|                                        | 3.3.2                                                         | Performance Guarantees . . . . . . . . . . . . . . . . . . . .              | 33   |
|                                        | 3.3.3                                                         | Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . .            | 34   |
| 3.4                                    | How Strong is Bellman Completion as a Modeling? . . . . .     | . . . . .                                                                   | 35   |
| 3.5                                    | Offline Reinforcement Learning                                | . . . . . . . . . . . . . . . . . . . .                                     | 36   |
|                                        | 3.5.1                                                         | Offline Learning . . . . . . . . . . . . . . . . . . . . . . . .            | 36   |
|                                        | 3.5.2                                                         | Offline Policy Evaluation . . . . . . . . . . . . . . . . . . . .           | 36   |
| 3.6                                    | Bibliographic Remarks and Further Readings .                  | . . . . . . . . . . . .                                                     | 37   |
| 4 Fitted Dynamic Programming Methods   | 4 Fitted Dynamic Programming Methods                          | 4 Fitted Dynamic Programming Methods                                        | 39   |
| 4.1                                    | Fitted Q -Iteration (FQI) and Offline RL                      | . . . . . . . . . . . . . . . .                                             | 39   |
|                                        |                                                               | The FQI Algorithm . . . . . . . . . . . . . . . . . . . . . . .             | 40   |
|                                        | 4.1.1                                                         |                                                                             |      |
|                                        | 4.1.2                                                         | Performance Guarantees of FQI . . . . . . . . . . . . . . . .               | 40   |
| 4.2                                    | Fitted Policy-Iteration (FPI)                                 | . . . . . . . . . . . . . . . . . . . . . .                                 | 42   |
| 4.3                                    | Failure Cases Without Assumption . .                          | 4.1 . . . . . . . . . . . . . . . . .                                       | 43   |
| 4.4                                    | FQI for Policy Evaluation .                                   | . . . . . . . . . . . . . . . . . . . . .                                   | 43   |
| 4.5                                    | Bibliographic Remarks and Further Readings .                  | . . . . . . . . . . . .                                                     | 43   |
| 5 Statistical Limits of Generalization | 5 Statistical Limits of Generalization                        | 5 Statistical Limits of Generalization                                      | 45   |
| 5.1                                    | Agnostic Learning                                             | . . . . . . . . . . . . . . . . . . . . . . . . . . .                       | 46   |
|                                        | 5.1.1                                                         | Review: Binary Classification . . . . . . . . . . . . . . . . .             | 46   |
|                                        | 5.1.2                                                         | Importance Sampling and a Reduction to Supervised Learning                  | 47   |
| 5.2                                    | Linear                                                        | Realizability . . . . . . . . . . . . . . . . . . . . . . . . . . .         | 49   |
|                                        | 5.2.1                                                         | Offline Policy Evaluation with Linearly Realizable Values . .               | 49   |
|                                        | 5.2.2                                                         | Linearly Realizable Q glyph[star] . . . . . . . . . . . . . . . . . . . . . | 53   |

|    |                                       | 5.2.3 Linearly glyph[star] . . . . .              | Realizable π . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58   |
|----|---------------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------|
|    | 5.3                                   | Discussion: Studying Generalization in RL         | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59                    |
|    | 5.4                                   | Bibliographic Remarks and Further Readings        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59                      |
| 2  | Strategic Exploration                 | Strategic Exploration                             | 61                                                                                  |
| 6  | Multi-Armed &Linear Bandits           | Multi-Armed &Linear Bandits                       | 63                                                                                  |
|    | 6.1                                   | The K -Armed Bandit Problem . . . . .             | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63                |
|    |                                       | 6.1.1 The Upper Confidence Bound (UCB) Algorithm  | . . . . . . . . . . . . . . . . . . . . . . . . 63                                  |
|    | 6.2                                   | Linear Bandits: Handling Large Action Spaces      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65                        |
|    |                                       | 6.2.1 The LinUCB algorithm . . . . .              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66                |
|    |                                       | 6.2.2 Upper and Lower Bounds . . .                | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67                |
|    | 6.3                                   | LinUCB Analysis . . . . . . . . . . . .           | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68                |
|    |                                       | 6.3.1 Regret Analysis . . . . . . . . .           | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69                |
|    |                                       | 6.3.2 Confidence Analysis . . . . . .             | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71                |
|    | 6.4                                   | Bibliographic Remarks and Further Readings .      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71                        |
| 7  | Strategic Exploration in Tabular MDPs | Strategic Exploration in Tabular MDPs             | 73                                                                                  |
|    | 7.1                                   | On The Need for Strategic Exploration .           | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73                |
|    | 7.2                                   | The UCB-VI algorithm . . . . . . . . .            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74                |
|    | 7.3                                   | Analysis . . . . . . . . . . . . . . . . .        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75                |
|    |                                       | 7.3.1 Proof of Lemma 7.2 . . . . . .              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 78                |
|    | 7.4                                   | An Improved Regret Bound . . . . . . .            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80                |
|    | 7.5                                   | Phased Q -learning . . . . . . . . . . .          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 83                |
|    | 7.6                                   | Bibliographic Remarks and Further Readings        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84                      |
| 8  | Linearly Parameterized MDPs           | Linearly Parameterized MDPs                       | 85                                                                                  |
|    | 8.1                                   | Setting . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85                |
|    |                                       | 8.1.1 Low-Rank MDPs and Linear MDPs               | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85                    |
|    | 8.2                                   | Planning in Linear MDPs . . . . . . .             | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86                |
|    | 8.3                                   | Learning Transition using Ridge Linear Regression | . . . . . . . . . . . . . . . . . . . . . . . . . . . 87                            |
|    | 8.5                                   | Algorithm . . . . . . . . . . . . . . . .         | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92                |

| 8.6                                                 | Analysis of UCBVI for Linear MDPs                            | . . . . . . . . . . . .                                        | 93   |
|-----------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------|------|
|                                                     | 8.6.1                                                        | Proving Optimism . . . . . . . . . . . . . . . . . .           | 93   |
|                                                     | 8.6.2                                                        | Regret Decomposition . . . . . . . . . . . . . . . .           | 94   |
|                                                     | 8.6.3                                                        | Concluding the Final Regret Bound . . . . . . . . .            | 95   |
| 8.7                                                 | Bibliographic Remarks and Further Readings .                 | . . . . . . .                                                  | 96   |
| 9 Generalization with                               | 9 Generalization with                                        | 9 Generalization with                                          | 97   |
|                                                     | Bounded Bilinear Rank                                        | Bounded Bilinear Rank                                          |      |
| 9.1                                                 |                                                              | Hypothesis Classes . . . . . . . . . . . . . . . . . . . . . . | 97   |
| 9.2                                                 |                                                              | The Bellman Rank . . . . . . . . . . . . . . . . . . . . . .   | 98   |
| 9.3                                                 |                                                              | Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . | 99   |
|                                                     | 9.3.1                                                        | Examples that have small Q -Bellman rank . . . . .             | 99   |
|                                                     | 9.3.2                                                        | Examples that have small V -Bellman rank . . . . .             | 102  |
| 9.4                                                 | Bilinear Classes . .                                         | . . . . . . . . . . . . . . . . . . . . . .                    | 103  |
|                                                     | 9.4.1                                                        | Examples . . . . . . . . . . . . . . . . . . . . . . .         | 103  |
| 9.5                                                 | PAC-RL with Bounded Bilinear Rank . . . . . . . . . .        | . .                                                            | 104  |
|                                                     | 9.5.1                                                        | Algorithm . . . . . . . . . . . . . . . . . . . . . . .        | 105  |
|                                                     | 9.5.2                                                        | Sample Complexity . . . . . . . . . . . . . . . . . .          | 105  |
|                                                     | 9.5.3                                                        | Analysis . . . . . . . . . . . . . . . . . . . . . . .         | 108  |
| 9.6                                                 | The Eluder Dimension                                         | . . . . . . . . . . . . . . . . . . . .                        | 111  |
| 9.7                                                 | Bibliographic Remarks and Further Readings                   | . . . . . . . .                                                | 111  |
| 10 Deterministic MDPs with Linearly Parameterized Q | 10 Deterministic MDPs with Linearly Parameterized Q          | glyph[star]                                                    | 113  |
| 3                                                   | Policy Optimization                                          | Policy Optimization                                            | 115  |
| 11                                                  | Policy Gradient Methods and Non-Convex Optimization          | Policy Gradient Methods and Non-Convex Optimization            | 117  |
|                                                     | 11.1                                                         | Policy Gradient Expressions and the Likelihood Ratio Method    | 118  |
|                                                     | 11.2                                                         | (Non-convex) Optimization . . . . . . . . . . . . . . . . . .  | 119  |
|                                                     |                                                              | 11.2.1 Gradient ascent and convergence to stationary points    | 120  |
|                                                     | 11.2.2 Monte Carlo estimation and stochastic gradient ascent | 11.2.2 Monte Carlo estimation and stochastic gradient ascent   | 120  |
| 11.3 Bibliographic 12 Optimality                    | 11.3 Bibliographic 12 Optimality                             | 11.3 Bibliographic 12 Optimality                               | 123  |

| 12.1      | Vanishing Gradients and                                             | Saddle Points . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123   |
|-----------|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| 12.2      | Policy Gradient Ascent . . . . . . . . . . . .                      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 124                         |
| 12.3      | Log Barrier Regularization . . . . . . . . . .                      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 126                         |
| 12.4      | The Natural Policy Gradient . . . . . . . . .                       | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128                         |
| 12.5      | Bibliographic Remarks and Further Readings                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 132                         |
| 13        | Function Approximation and the NPG                                  | 133                                                                                     |
| 13.1      | Compatible function approximation and the NPG                       | . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133                             |
| 13.2      | Examples: NPG and Q -NPG . . . . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 135                         |
|           | 13.2.1 Log-linear Policy Classes and Soft Policy Iteration          | . . . . . . . . . . . . . . . . . . . . . . . 135                                       |
|           | 13.2.2 Neural Policy Classes . . . . . . . .                        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 136                         |
| 13.3      | The NPG 'Regret Lemma' . . . . . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 136                         |
| 13.4      | Q -NPG: Performance Bounds for Log-Linear Policies                  | . . . . . . . . . . . . . . . . . . . . . . . . . 138                                   |
|           | 13.4.1 Analysis . . . . . . . . . . . . . . .                       | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 140                         |
| 13.5      | Q -NPG Sample Complexity . . . . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 142                         |
| 13.6      | Bibliographic Remarks and Further Readings . .                      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . 142                             |
| 14 CPI,   | TRPO, and More                                                      | 143                                                                                     |
| 14.1      | Conservative Policy Iteration . . . . . . . . .                     | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143                         |
|           | 14.1.1 The CPI Algorithm . . . . . . . . . .                        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144                         |
| 14.2      | Trust Region Methods and Covariant Policy Search                    | . . . . . . . . . . . . . . . . . . . . . . . . . . . 148                               |
|           | 14.2.1 Proximal Policy Optimization . . . .                         | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 150                         |
| 14.3      | Bibliographic Remarks and Further Readings                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 151                         |
| 4         | Further Topics                                                      | 153                                                                                     |
| Imitation | 15.1 Setting . . . . . . . . . . . . . . . . . . . . .              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                             |
| 15.2      |                                                                     | 155                                                                                     |
|           | Offline IL: Behavior Cloning . . . . . . . . .                      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155                         |
| 15.3      | The Hybrid Setting: Statistical Benefit and Algorithm               | . . . . . . . . . . . . . . . . . . . . . . . . . . 156                                 |
|           | 15.3.1 Extension to Agnostic Setting . . . .                        | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 158                         |
| 15.4      | Maximum Entropy Inverse Reinforcement                               | Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . 159                      |
|           | 15.4.1 MaxEnt IRL: Formulation and The Principle of Maximum Entropy | . . . . . . . . . . . . . . 160                                                         |

|                 | 15.4.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . .                                                             | 160     |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------|---------|
| 15.5            | 15.4.3 Maximum Entropy RL: Implementing the Planning Oracle in Interactive Imitation Learning: . . .                             | 161     |
|                 | AggreVaTe and Its Statistical Benefit over Offline IL Setting                                                                    | 162     |
| 15.6            | Bibliographic Remarks and Further Readings . . . . . . . . . . .                                                                 | 165     |
| 16 Linear       | Quadratic Regulators                                                                                                             | 167     |
| 16.1            | The LQR Model . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                              | 167     |
| 16.2            | Value Iteration &The Algebraic Riccati Equations . . . . . . . . 16.2.1 Planning and Finite Horizon LQRs . . . . . . . . . . . . | 168 168 |
|                 | 16.2.2 Planning and Infinite Horizon LQRs . . . . . . . . . . . .                                                                | 170     |
| 16.3            | Convex Programs to find P and K glyph[star] . . . . . . . . . . . . . . . .                                                      | 170     |
|                 | 16.3.1 The Primal for Infinite Horizon LQR . . . . . . . . . . .                                                                 | 171     |
|                 | 16.3.2 The Dual . . . . . . . . . . . . . . . . . . . . . . . . . .                                                              | 171     |
| 16.4            | Policy Iteration, Gauss Newton, and NPG . . . . . . . . . . . . .                                                                | 172     |
|                 | 16.4.1 Gradient Expressions . . . . . . . . . . . . . . . . . . . .                                                              | 172     |
|                 | 16.4.2 Convergence Rates . . . . . . . . . . . . . . . . . . . . .                                                               | 173     |
|                 | 16.4.3 Gauss-Newton Analysis . . . . . . . . . . . . . . . . . .                                                                 | 174     |
| 16.5            | System Level Synthesis for Linear Dynamical Systems . . . . . .                                                                  | 176     |
| 16.6            | Bibliographic Remarks and Further Readings . . . . . . . . . . .                                                                 | 179     |
| 17              | Partially Observable Markov Decision Processes                                                                                   | 181     |
| Bibliography    | Bibliography                                                                                                                     | 183     |
| A Concentration | A Concentration                                                                                                                  | 193     |

## Notation

The reader might find it helpful to refer back to this notation section.

- We slightly abuse notation and let [ K ] denote the set { 0 , 1 , 2 , . . . K -1 } for an integer K .
- We let ∆( X ) denote the set of probability distribution over the set X .
- For a vector v , we let ( v ) 2 , √ v , and | v | be the component-wise square, square root, and absolute value operations.
- Inequalities between vectors are elementwise, e.g. for vectors v, v ′ , we say v ≤ v ′ , if the inequality holds elementwise.
- For a vector v , we refer to the j -th component of this vector by either v ( j ) or [ v ] j
- Denote the variance of any real valued f under a distribution D as:

<!-- formula-not-decoded -->

- We overload notation where, for a distribution µ over S , we write:

<!-- formula-not-decoded -->

- It is helpful to overload notation and let P also refer to a matrix of size ( S · A ) ×S where the entry P ( s,a ) ,s ′ is equal to P ( s ′ | s, a ) . We also will define P π to be the transition matrix on state-action pairs induced by a deterministic policy π . In particular, P π ( s,a ) , ( s ′ ,a ′ ) = P ( s ′ | s, a ) if a ′ = π ( s ′ ) and P π ( s,a ) , ( s ′ ,a ′ ) = 0 if a ′ = π ( s ′ ) . With this notation,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- For a vector Q ∈ R |S×A| , denote the greedy policy and value as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- For a vector Q ∈ R |S×A| , the Bellman optimality operator T : R |S×A| → R |S×A| is defined as:

<!-- formula-not-decoded -->

- For a vector x and a matrix M , unless explicitly stated otherwise, we let ‖ x ‖ and ‖ M ‖ denote the Euclidean and spectral norms, respectively. We use the notation ‖ x ‖ M = √ x glyph[latticetop] Mx , where x and M are assumed to be of appropriate size.

glyph[negationslash]

## Part 1 Fundamentals

## Chapter 1

## Markov Decision Processes

## 1.1 Discounted (Infinite-Horizon) Markov Decision Processes

In reinforcement learning, the interactions between the agent and the environment are often described by an infinitehorizon, discounted Markov Decision Process (MDP) M = ( S , A , P, r, γ, µ ) , specified by:

- A state space S , which may be finite or infinite. For mathematical convenience, we will assume that S is finite or countably infinite.
- An action space A , which also may be discrete or infinite. For mathematical convenience, we will assume that A is finite.
- A transition function P : S × A → ∆( S ) , where ∆( S ) is the space of probability distributions over S (i.e., the probability simplex). P ( s ′ | s, a ) is the probability of transitioning into state s ′ upon taking action a in state s . We use P s,a to denote the vector P ( · ∣ ∣ s, a ) .
- A reward function r : S × A → [0 , 1] . r ( s, a ) is the immediate reward associated with taking action a in state s . More generally, the r ( s, a ) could be a random variable (where the distribution depends on s, a ). While we largely focus on the case where r ( s, a ) is deterministic, the extension to methods with stochastic rewards are often straightforward.
- A discount factor γ ∈ [0 , 1) , which defines a horizon for the problem.
- An initial state distribution µ ∈ ∆( S ) , which specifies how the initial state s 0 is generated.

In many cases, we will assume that the initial state is fixed at s 0 , i.e. µ is a distribution supported only on s 0 .

## 1.1.1 The objective, policies, and values

Policies. In a given MDP M = ( S , A , P, r, γ, µ ) , the agent interacts with the environment according to the following protocol: the agent starts at some state s 0 ∼ µ ; at each time step t = 0 , 1 , 2 , . . . , the agent takes an action a t ∈ A , obtains the immediate reward r t = r ( s t , a t ) , and observes the next state s t +1 sampled according to s t +1 ∼ P ( ·| s t , a t ) . The interaction record at time t ,

<!-- formula-not-decoded -->

is called a trajectory , which includes the observed state at time t .

In the most general setting, a policy specifies a decision-making strategy in which the agent chooses actions adaptively based on the history of observations; precisely, a policy is a (possibly randomized) mapping from a trajectory to an action, i.e. π : H → ∆( A ) where H is the set of all possible trajectories (of all lengths) and ∆( A ) is the space of probability distributions over A . A stationary policy π : S → ∆( A ) specifies a decision-making strategy in which the agent chooses actions based only on the current state, i.e. a t ∼ π ( ·| s t ) . A deterministic, stationary policy is of the form π : S → A .

Values. We now define values for (general) policies. For a fixed policy and a starting state s 0 = s , we define the value function V π M : S → R as the discounted sum of future rewards

<!-- formula-not-decoded -->

where expectation is with respect to the randomness of the trajectory, that is, the randomness in state transitions and the stochasticity of π . Here, since r ( s, a ) is bounded between 0 and 1 , we have 0 ≤ V π M ( s ) ≤ 1 / (1 -γ ) .

Similarly, the action-value (or Q-value) function Q π M : S × A → R is defined as

<!-- formula-not-decoded -->

and Q π M ( s, a ) is also bounded by 1 / (1 -γ ) .

Goal. Given a state s , the goal of the agent is to find a policy π that maximizes the value, i.e. the optimization problem the agent seeks to solve is:

<!-- formula-not-decoded -->

where the max is over all (possibly non-stationary and randomized) policies. As we shall see, there exists a deterministic and stationary policy which is simultaneously optimal for all starting states s .

We drop the dependence on M and write V π when it is clear from context.

Example 1.1 (Navigation) . Navigation is perhaps the simplest to see example of RL. The state of the agent is their current location. The four actions might be moving 1 step along each of east, west, north or south. The transitions in the simplest setting are deterministic. Taking the north action moves the agent one step north of their location, assuming that the size of a step is standardized. The agent might have a goal state g they are trying to reach, and the reward is 0 until the agent reaches the goal, and 1 upon reaching the goal state. Since the discount factor γ &lt; 1 , there is incentive to reach the goal state earlier in the trajectory. As a result, the optimal behavior in this setting corresponds to finding the shortest path from the initial to the goal state, and the value function of a state, given a policy is γ d , where d is the number of steps required by the policy to reach the goal state.

Example 1.2 (Conversational agent) . This is another fairly natural RL problem. The state of an agent can be the current transcript of the conversation so far, along with any additional information about the world, such as the context for the conversation, characteristics of the other agents or humans in the conversation etc. Actions depend on the domain. In the most basic form, we can think of it as the next statement to make in the conversation. Sometimes, conversational agents are designed for task completion, such as travel assistant or tech support or a virtual office receptionist. In these cases, there might be a predefined set of slots which the agent needs to fill before they can find a good solution. For instance, in the travel agent case, these might correspond to the dates, source, destination and mode of travel. The actions might correspond to natural language queries to fill these slots.

In task completion settings, reward is naturally defined as a binary outcome on whether the task was completed or not, such as whether the travel was successfully booked or not. Depending on the domain, we could further refine it based on the quality or the price of the travel package found. In more generic conversational settings, the ultimate reward is whether the conversation was satisfactory to the other agents or humans, or not.

Example 1.3 (Strategic games) . This is a popular category of RL applications, where RL has been successful in achieving human level performance in Backgammon, Go, Chess, and various forms of Poker. The usual setting consists of the state being the current game board, actions being the potential next moves and reward being the eventual win/loss outcome or a more detailed score when it is defined in the game. Technically, these are multi-agent RL settings, and, yet, the algorithms used are often non-multi-agent RL algorithms.

## 1.1.2 Bellman Consistency Equations for Stationary Policies

Stationary policies satisfy the following consistency conditions:

Lemma 1.4. Suppose that π is a stationary policy. Then V π and Q π satisfy the following Bellman consistency equations : for all s ∈ S , a ∈ A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We leave the proof as an exercise to the reader.

It is helpful to view V π as vector of length |S| and Q π and r as vectors of length |S| · |A| . We overload notation and let P also refer to a matrix of size ( |S| · |A| ) ×|S| where the entry P ( s,a ) ,s ′ is equal to P ( s ′ | s, a ) .

We also will define P π to be the transition matrix on state-action pairs induced by a stationary policy π , specifically:

<!-- formula-not-decoded -->

In particular, for deterministic policies we have:

<!-- formula-not-decoded -->

With this notation, it is straightforward to verify:

<!-- formula-not-decoded -->

Corollary 1.5. Suppose that π is a stationary policy. We have that:

<!-- formula-not-decoded -->

where I is the identity matrix.

Proof: To see that the I -γP π is invertible, observe that for any non-zero vector x ∈ R |S||A| ,

<!-- formula-not-decoded -->

which implies I -γP π is full rank.

The following is also a helpful lemma:

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

Lemma 1.6. We have that:

<!-- formula-not-decoded -->

so we can view the ( s, a ) -th row of this matrix as an induced distribution over states and actions when following π after starting with s 0 = s and a 0 = a .

We leave the proof as an exercise to the reader.

## 1.1.3 Bellman Optimality Equations

A remarkable and convenient property of MDPs is that there exists a stationary and deterministic policy that simultaneously maximizes V π ( s ) for all s ∈ S . This is formalized in the following theorem:

Theorem 1.7. Let Π be the set of all non-stationary and randomized policies. Define:

<!-- formula-not-decoded -->

which is finite since V π ( s ) and Q π ( s, a ) are bounded between 0 and 1 / (1 -γ ) .

There exists a stationary and deterministic policy π such that for all s ∈ S and a ∈ A ,

<!-- formula-not-decoded -->

We refer to such a π as an optimal policy.

Proof: For any π ∈ Π and for any time t , π specifies a distribution over actions conditioned on the history of observations; here, we write π ( A t = a | S 0 = s 0 , A 0 = a 0 , R 0 = r 0 , . . . S t -1 = s t -1 , A t -1 = a t -1 , R t -1 = r t -1 , S t = s t ) as the probability that π selects action a at time t given an observed history s 0 , a 0 , r 0 , . . . s t -1 , a t -1 , r t -1 , s t . For the purposes of this proof, it is helpful to formally let S t , A t and R t denote random variables, which will distinguish them from outcomes, which is denoted by lower case variables. First, let us show that conditioned on ( S 0 , A 0 , R 0 , S 1 ) = ( s, a, r, s ′ ) , the maximum future discounted value, from time 1 onwards, is not a function of s, a, r . More precisely, we seek to show that:

<!-- formula-not-decoded -->

For any policy π , define an 'offset' policy π ( s,a,r ) , which is the policy that chooses actions on a trajectory τ according to the same distribution that π chooses actions on the trajectory ( s, a, r, τ ) . Precisely, for all t , define

<!-- formula-not-decoded -->

By the Markov property, we have that:

<!-- formula-not-decoded -->

where the first equality follows from a change of variables on the time index, along with the definition of the policy π ( s,a,r ) . Also, we have that, for all ( s, a, r ) , that the set { π ( s,a,r ) | π ∈ Π } is equal to Π itself, by the definition of Π and π ( s,a,r ) . This implies:

<!-- formula-not-decoded -->

thus proving Equation 0.3.

We now show the deterministic and stationary policy

<!-- formula-not-decoded -->

is optimal, i.e. that V ˜ π ( s ) = V glyph[star] ( s ) . For this, we have that:

<!-- formula-not-decoded -->

where step ( a ) uses the law of iterated expectations; step ( b ) uses Equation 0.3; and step ( c ) follows from the definition of ˜ π . Applying the same argument recursively leads to:

<!-- formula-not-decoded -->

Since V ˜ π ( s ) sup π ∈ Π V π ( s ) = V glyph[star] ( s ) for all s , we have that V ˜ π = V glyph[star] , which completes the proof of the first claim.

≤

For the same policy π , an analogous argument can be used prove the second claim.

˜

This shows that we may restrict ourselves to using stationary and deterministic policies without any loss in performance. The following theorem, also due to [Bellman, 1956], gives a precise characterization of the optimal value function.

Theorem 1.8 (Bellman optimality equations) . We say that a vector Q ∈ R |S||A| satisfies the Bellman optimality equations if:

<!-- formula-not-decoded -->

For any Q ∈ R |S||A| , we have that Q = Q glyph[star] if and only if Q satisfies the Bellman optimality equations. Furthermore, the deterministic policy defined by π ( s ) ∈ argmax a ∈A Q glyph[star] ( s, a ) is an optimal policy (where ties are broken in some arbitrary manner).

Before we prove this claim, we will provide a few definitions. Let π Q denote the greedy policy with respect to a vector Q ∈ R |S||A| , i.e

<!-- formula-not-decoded -->

where ties are broken in some arbitrary manner. With this notation, by the above theorem, the optimal policy π glyph[star] is given by:

<!-- formula-not-decoded -->

Let us also use the following notation to turn a vector Q ∈ R |S||A| into a vector of length |S| .

<!-- formula-not-decoded -->

The Bellman optimality operator T M : R |S||A| → R |S||A| is defined as:

<!-- formula-not-decoded -->

This allows us to rewrite the Bellman optimality equation in the concise form:

<!-- formula-not-decoded -->

and, so, the previous theorem states that Q = Q glyph[star] if and only if Q is a fixed point of the operator T .

Proof: Let us begin by showing that:

<!-- formula-not-decoded -->

Let π glyph[star] be an optimal stationary and deterministic policy, which exists by Theorem 1.7. Consider the policy which takes action a and then follows π glyph[star] . Due to that V glyph[star] ( s ) is the maximum value over all non-stationary policies (as shown in Theorem 1.7), which shows that V glyph[star] ( s ) ≥ max a Q glyph[star] ( s, a ) since a is arbitrary in the above. Also, by Lemma 1.4 and Theorem 1.7,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves our claim since we have upper and lower bounded V glyph[star] ( s ) by max a Q glyph[star] ( s, a ) .

We first show sufficiency, i.e. that Q glyph[star] (the state-action value of an optimal policy) satisfies Q glyph[star] = T Q glyph[star] . Now for all actions a ∈ A , we have:

<!-- formula-not-decoded -->

Here, the second equality follows from Theorem 1.7 and the final equality follows from Equation 0.5. This proves sufficiency.

For the converse, suppose Q = T Q for some Q . We now show that Q = Q glyph[star] . Let π = π Q . That Q = T Q implies that Q = r + γP π Q , and so:

<!-- formula-not-decoded -->

using Corollary 1.5 in the last step. In other words, Q is the action value of the policy π Q . Also, let us show that ( P π -P π ′ ) Q π ≥ 0 . To see this, observe that:

<!-- formula-not-decoded -->

where the last step uses that π = π Q . Now observe for any other deterministic and stationary policy π ′ :

<!-- formula-not-decoded -->

where the last step follows since we have shown ( P π -P π ′ ) Q π ≥ 0 and since (1 -γ )( I -γP π ′ ) -1 is a matrix with all positive entries (see Lemma 1.6). Thus, Q π = Q ≥ Q π ′ for all deterministic and stationary π ′ , which shows that π is an optimal policy. Thus, Q = Q π = Q glyph[star] , using Theorem 1.7. This completes the proof.

## 1.2 Finite-Horizon Markov Decision Processes

In some cases, it is natural to work with finite-horizon (and time-dependent) Markov Decision Processes (see the discussion below). Here, a finite horizon, time-dependent Markov Decision Process (MDP) M = ( S , A , { P } h , { r } h , H, µ ) is specified as follows:

- A state space S , which may be finite or infinite.
- An action space A , which also may be discrete or infinite.
- Atime-dependent transition function P h : S ×A → ∆( S ) , where ∆( S ) is the space of probability distributions over S (i.e., the probability simplex). P h ( s ′ | s, a ) is the probability of transitioning into state s ′ upon taking action a in state s at time step h . Note that the time-dependent setting generalizes the stationary setting where all steps share the same transition.
- Atime-dependent reward function r h : S ×A → [0 , 1] . r h ( s, a ) is the immediate reward associated with taking action a in state s at time step h .
- A integer H which defines the horizon of the problem.
- An initial state distribution µ ∈ ∆( S ) , which species how the initial state s 0 is generated.

Here, for a policy π , a state s , and h ∈ { 0 , . . . H -1 } , we define the value function V π h : S → R as

<!-- formula-not-decoded -->

where again the expectation is with respect to the randomness of the trajectory, that is, the randomness in state transitions and the stochasticity of π . Similarly, the state-action value (or Q-value) function Q π h : S × A → R is defined as

<!-- formula-not-decoded -->

We also use the notation V π ( s ) = V π 0 ( s ) .

Again, given a state s , the goal of the agent is to find a policy π that maximizes the value, i.e. the optimization problem the agent seeks to solve is:

<!-- formula-not-decoded -->

where recall that V π ( s ) = V π 0 ( s ) .

Theorem 1.9. (Bellman optimality equations) Define

<!-- formula-not-decoded -->

where the sup is over all non-stationary and randomized policies. Suppose that Q H = 0 . We have that Q h = Q glyph[star] h for all h ∈ [ H ] if and only if for all h ∈ [ H ] ,

<!-- formula-not-decoded -->

Furthermore, π ( s, h ) = argmax a ∈A Q glyph[star] h ( s, a ) is an optimal policy.

We leave the proof as an exercise to the reader.

Discussion: Stationary MDPs vs Time-Dependent MDPs For the purposes of this book, it is natural for us to study both of these models, where we typically assume stationary dynamics in the infinite horizon setting and timedependent dynamics in the finite-horizon setting. From a theoretical perspective, the finite horizon, time-dependent setting is often more amenable to analysis, where optimal statistical rates often require simpler arguments. However, we should note that from a practical perspective, time-dependent MDPs are rarely utilized because they lead to policies and value functions that are O ( H ) larger (to store in memory) than those in the stationary setting. In practice, we often incorporate temporal information directly into the definition of the state, which leads to more compact value functions and policies (when coupled with function approximation methods, which attempt to represent both the values and policies in a more compact form).

## 1.3 Computational Complexity

This section will be concerned with computing an optimal policy, when the MDP M = ( S , A , P, r, γ ) is known; this can be thought of as the solving the planning problem. While much of this book is concerned with statistical limits, understanding the computational limits can be informative. We will consider algorithms which give both exact and approximately optimal policies. In particular, we will be interested in polynomial time (and strongly polynomial time) algorithms.

Suppose that ( P, r, γ ) in our MDP M is specified with rational entries. Let L ( P, r, γ ) denote the total bit-size required to specify M , and assume that basic arithmetic operations + , -, × , ÷ take unit time. Here, we may hope for an algorithm which (exactly) returns an optimal policy whose runtime is polynomial in L ( P, r, γ ) and the number of states and actions.

More generally, it may also be helpful to understand which algorithms are strongly polynomial. Here, we do not want to explicitly restrict ( P, r, γ ) to be specified by rationals. An algorithm is said to be strongly polynomial if it returns an optimal policy with runtime that is polynomial in only the number of states and actions (with no dependence on L ( P, r, γ ) ).

The first two subsections will cover classical iterative algorithms that compute Q glyph[star] , and then we cover the linear programming approach.

## 1.3.1 Value Iteration

Perhaps the simplest algorithm for discounted MDPs is to iteratively apply the fixed point mapping: starting at some Q , we iteratively apply T :

<!-- formula-not-decoded -->

Table 0.1: Computational complexities of various approaches (we drop universal constants). Polynomial time algorithms depend on the bit complexity, L ( P, r, γ ) , while strongly polynomial algorithms do not. Note that only for a fixed value of γ are value and policy iteration polynomial time algorithms; otherwise, they are not polynomial time algorithms. Similarly, only for a fixed value of γ is policy iteration a strongly polynomial time algorithm. In contrast, the LP-approach leads to both polynomial time and strongly polynomial time algorithms; for the latter, the approach is an interior point algorithm. See text for further discussion, and Section 1.6 for references. Here, |S| 2 |A| is the assumed runtime per iteration of value iteration, and |S| 3 + |S| 2 |A| is the assumed runtime per iteration of policy iteration (note that for this complexity we would directly update the values V rather than Q values, as described in the text); these runtimes are consistent with assuming cubic complexity for linear system solving.

|                | Value Iteration                         | Policy Iteration                                                | LP-Algorithms             |
|----------------|-----------------------------------------|-----------------------------------------------------------------|---------------------------|
| Poly?          | |S| 2 |A| L ( P,r,γ ) log 1 1 - γ 1 - γ | ( |S| 3 + |S| 2 |A| ) L ( P,r,γ ) log 1 1 - γ 1 - γ             | |S| 3 |A| L ( P, r,γ )    |
| Strongly Poly? | 7                                       | ( |S| 3 + |S| 2 |A| ) · min { |A| |S| |S| , |S| 2 |A| log 1 - γ | |S| 4 |A| 4 log |S| 1 - γ |

This is algorithm is referred to as Q -value iteration .

Lemma 1.10. (contraction) For any two vectors Q,Q ′ ∈ R |S||A| ,

<!-- formula-not-decoded -->

Proof: First, let us show that for all s ∈ S , | V Q ( s ) -V Q ′ ( s ) | ≤ max a ∈A | Q ( s, a ) -Q ′ ( s, a ) | . Assume V Q ( s ) &gt; V Q ′ ( s ) (the other direction is symmetric), and let a be the greedy action for Q at s . Then

<!-- formula-not-decoded -->

Using this,

<!-- formula-not-decoded -->

where the first inequality uses that each element of P ( V Q -V Q ′ ) is a convex average of V Q -V Q ′ and the second inequality uses our claim above.

The following result bounds the sub-optimality of the greedy policy itself, based on the error in Q -value function.

Lemma 1.11. ( Q -Error Amplification) For any vector Q ∈ R |S||A| ,

<!-- formula-not-decoded -->

where 1 denotes the vector of all ones.

Proof: Fix state s and let a = π Q ( s ) . We have:

<!-- formula-not-decoded -->

where the first inequality uses Q ( s, π ( s )) Q ( s, π Q ( s )) = Q ( s, a ) due to the definition of π Q

glyph[star] ≤ .

Theorem 1.12. ( Q -value iteration convergence). Set Q (0) = 0 . For k = 0 , 1 , . . . , suppose:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Since ‖ Q glyph[star] ‖ ∞ ≤ 1 / (1 -γ ) , Q ( k ) = T k Q (0) and Q glyph[star] = T Q glyph[star] , Lemma 1.10 gives

<!-- formula-not-decoded -->

The proof is completed with our choice of k and using Lemma 1.11.

Iteration complexity for an exact solution. With regards to computing an exact optimal policy, when the gap between the current objective value and the optimal objective value is smaller than 2 -L ( P,r,γ ) , then the greedy policy will be optimal. This leads to claimed complexity in Table 0.1. Value iteration is not strongly polynomial algorithm due to that, in finite time, it may never return the optimal policy.

## 1.3.2 Policy Iteration

The policy iteration algorithm, for discounted MDPs, starts from an arbitrary policy π 0 , and repeats the following iterative procedure: for k = 0 , 1 , 2 , . . .

1. Policy evaluation. Compute Q π k
2. Policy improvement. Update the policy:

<!-- formula-not-decoded -->

In each iteration, we compute the Q-value function of π k , using the analytical form given in Equation 0.2, and update the policy to be greedy with respect to this new Q -value. The first step is often called policy evaluation , and the second step is often called policy improvement .

Lemma 1.13. We have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: First let us show that T Q π k ≥ Q π k . Note that the policies produced in policy iteration are always deterministic, so V π k ( s ) = Q π k ( s, π k ( s )) for all iterations k and states s . Hence,

<!-- formula-not-decoded -->

Now let us prove that Q π k +1 ≥ T Q π k . First, let us see that Q π k +1 ≥ Q π k :

<!-- formula-not-decoded -->

where we have used that π k +1 is the greedy policy in the first inequality and recursion in the second inequality. Using this,

<!-- formula-not-decoded -->

which completes the proof of the first claim.

For the second claim,

<!-- formula-not-decoded -->

where we have used that Q glyph[star] ≥ Q π k +1 ≥ T Q π k in second step and the contraction property of T (see Lemma 1.10) in the last step.

With this lemma, a convergence rate for the policy iteration algorithm immediately follows.

Theorem 1.14. (Policy iteration convergence). Let π 0 be any initial policy. For k ≥ log 1 (1 -γ ) glyph[epsilon1] 1 -γ , the k -th policy in policy iteration has the following performance bound:

<!-- formula-not-decoded -->

Iteration complexity for an exact solution. With regards to computing an exact optimal policy, it is clear from the previous results that policy iteration is no worse than value iteration. However, with regards to obtaining an exact solution MDP that is independent of the bit complexity, L ( P, r, γ ) , improvements are possible (and where we assume basic arithmetic operations on real numbers are order one cost). Naively, the number of iterations of policy iterations is bounded by the number of policies, namely |A| |S| ; here, a small improvement is possible, where the number of iterations of policy iteration can be bounded by |A| |S| |S| . Remarkably, for a fixed value of γ , policy iteration can be show to be a strongly polynomial time algorithm, where policy iteration finds an exact policy in at most |S| 2 |A| log |S| 2 1 -γ 1 -γ iterations. See Table 0.1 for a summary, and Section 1.6 for references.

## 1.3.3 Value Iteration for Finite Horizon MDPs

Let us now specify the value iteration algorithm for finite-horizon MDPs. For the finize-horizon setting, it turns out that the analogues of value iteration and policy iteration lead to identical algorithms. The value iteration algorithm is specified as follows:

1. Set Q H -1 ( s, a ) = r H -1 ( s, a ) .
2. For h = H -2 , . . . 0 , set:

<!-- formula-not-decoded -->

By Theorem 1.9, it follows that Q h ( s, a ) = Q glyph[star] h ( s, a ) and that π ( s, h ) = argmax a ∈A Q glyph[star] h ( s, a ) is an optimal policy.

## 1.3.4 The Linear Programming Approach

It is helpful to understand an alternative approach to finding an optimal policy for a known MDP. With regards to computation, consider the setting where our MDP M = ( S , A , P, r, γ, µ ) is known and P , r , and γ are all specified by rational numbers. Here, from a computational perspective, the previous iterative algorithms are, strictly speaking, not polynomial time algorithms, due to that they depend polynomially on 1 / (1 -γ ) , which is not polynomial in the description length of the MDP . In particular, note that any rational value of 1 -γ may be specified with only O (log 1 1 -γ ) bits of precision. In this context, we may hope for a fully polynomial time algorithm, when given knowledge of the MDP, which would have a computation time which would depend polynomially on the description length of the MDP M , when the parameters are specified as rational numbers. We now see that the LP approach provides a polynomial time algorithm.

## The Primal LP and A Polynomial Time Algorithm

Consider the following optimization problem with variables V ∈ R |S| :

<!-- formula-not-decoded -->

Provided that µ has full support, then the optimal value function V glyph[star] ( s ) is the unique solution to this linear program. With regards to computation time, linear programming approaches only depend on the description length of the coefficients in the program, due to that this determines the computational complexity of basic additions and multiplications. Thus, this approach will only depend on the bit length description of the MDP, when the MDP is specified by rational numbers.

Computational complexity for an exact solution. Table 0.1 shows the runtime complexity for the LP approach, where we assume a standard runtime for solving a linear program. The strongly polynomial algorithm is an interior point algorithm. See Section 1.6 for references.

Policy iteration and the simplex algorithm. It turns out that the policy iteration algorithm is actually the simplex method with block pivot. While the simplex method, in general, is not a strongly polynomial time algorithm, the policy iteration algorithm is a strongly polynomial time algorithm, provided we keep the discount factor fixed. See [Ye, 2011].

## The Dual LP and the State-Action Polytope

For a fixed (possibly stochastic) policy π , let us define a visitation measure over states and actions induced by following π after starting at s 0 . Precisely, define this distribution, d π s 0 , as follows:

<!-- formula-not-decoded -->

where Pr π ( s t = s, a t = a | s 0 ) is the probability that s t = s and a t = a , after starting at state s 0 and following π thereafter. It is straightforward to verify that d π s 0 is a distribution over S × A . We also overload notation and write:

<!-- formula-not-decoded -->

for a distribution µ over S . Recall Lemma 1.6 provides a way to easily compute d π µ ( s, a ) through an appropriate vector-matrix multiplication.

It is straightforward to verify that d π µ satisfies, for all states s ∈ S :

<!-- formula-not-decoded -->

Let us define the state-action polytope as follows:

<!-- formula-not-decoded -->

We now see that this set precisely characterizes all state-action visitation distributions.

Proposition 1.15. We have that K µ is equal to the set of all feasible state-action distributions, i.e. d ∈ K µ if and only if there exists a stationary (and possibly randomized) policy π such that d π µ = d .

With respect the variables d ∈ R |S|·|A| , the dual LP formulation is as follows:

<!-- formula-not-decoded -->

Note that K µ is itself a polytope, and one can verify that this is indeed the dual of the aforementioned LP. This approach provides an alternative approach to finding an optimal solution.

If d glyph[star] is the solution to this LP, and provided that µ has full support, then we have that:

<!-- formula-not-decoded -->

is an optimal policy. An alternative optimal policy is argmax a d glyph[star] ( s, a ) (and these policies are identical if the optimal policy is unique).

## 1.4 Sample Complexity and Sampling Models

Much of reinforcement learning is concerned with finding a near optimal policy (or obtaining near optimal reward) in settings where the MDPs is not known to the learner. We will study these questions in a few different models of how the agent obtains information about the unknown underlying MDP. In each of these settings, we are interested understanding the number of samples required to find a near optimal policy, i.e. the sample complexity . Ultimately, we interested in obtaining results which are applicable to cases where number of states and actions is large (or, possibly, countably or uncountably infinite). This is many ways analogous to the supervised learning question of generalization, though, as we shall see, this question is fundamentally more challenging in the reinforcement learning setting.

The Episodic Setting. In the episodic setting, in every episode, the learner acts for some finite number of steps, starting from a fixed starting state s 0 ∼ µ , the learner observes the trajectory, and the state resets to s 0 ∼ µ . This episodic model of feedback is applicable to both the finte-horizon and infinite horizon settings.

- (Finite Horizon MDPs) Here, each episode lasts for H -steps, and then the state is reset to s 0 ∼ µ .
- (Infinite Horizon MDPs) Even for infinite horizon MDPs it is natural to work in an episodic model for learning, where each episode terminates after a finite number of steps. Here, it is often natural to assume either the agent can terminate the episode at will or that the episode will terminate at each step with probability 1 -γ . After termination, we again assume that the state is reset to s 0 ∼ µ . Note that, if each step in an episode is terminated with probability 1 -γ , then the observed cumulative reward in an episode of a policy provides an unbiased estimate of the infinite-horizon, discounted value of that policy.

In this setting, we are often interested in either the number of episodes it takes to find a near optimal policy, which is a PAC (probably, approximately correct) guarantee, or we are interested in a regret guarantee (which we will study in Chapter 7). Both of these questions are with regards to statistical complexity (i.e. the sample complexity) of learning.

The episodic setting is challenging in that the agent has to engage in some exploration in order to gain information at the relevant state. As we shall see in Chapter 7, this exploration must be strategic, in the sense that simply behaving randomly will not lead to information being gathered quickly enough. It is often helpful to study the statistical complexity of learning in a more abstract sampling model, a generative model, which allows to avoid having to directly address this exploration issue. Furthermore, this sampling model is natural in its own right.

The generative model setting. A generative model takes as input a state action pair ( s, a ) and returns a sample s ′ ∼ P ( ·| s, a ) and the reward r ( s, a ) (or a sample of the reward if the rewards are stochastic).

The offline RL setting. The offline RL setting is where the agent has access to an offline dataset, say generated under some policy (or a collection of policies). In the simplest of these settings, we may assume our dataset is of the form { ( s, a, s ′ , r ) } where r is the reward (corresponding to r ( s, a ) if the reward is deterministic) and s ′ ∼ P ( ·| s, a ) . Furthermore, for simplicity, it can be helpful to assume that the s, a pairs in this dataset were sampled i.i.d. from some fixed distribution ν over S × A .

## 1.5 Bonus: Advantages and The Performance Difference Lemma

Throughout, we will overload notation where, for a distribution µ over S , we write:

<!-- formula-not-decoded -->

The advantage A π ( s, a ) of a policy π is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that:

for all state-action pairs.

Analogous to the state-action visitation distribution (see Equation 0.8), we can define a visitation measure over just the states. When clear from context, we will overload notation and also denote this distribution by d π s 0 , where:

<!-- formula-not-decoded -->

Here, Pr π ( s t = s | s 0 ) is the state visitation probability, under π starting at state s 0 . Again, we write:

for a distribution µ over S .

The following lemma is helpful in the analysis of RL algorithms.

Lemma 1.16. (The performance difference lemma) For all policies π, π ′ and distributions µ over S ,

<!-- formula-not-decoded -->

Proof: Let Pr π ( τ | s 0 = s ) denote the probability of observing a trajectory τ when starting in state s and following the policy π . By definition of d π θ s 0 , observe that for any function f : S × A → R ,

<!-- formula-not-decoded -->

Using a telescoping argument, we have:

<!-- formula-not-decoded -->

where step ( a ) rearranges terms in the summation via telescoping; step ( b ) uses the law of iterated expectations; step ( c ) follows by definition; and the final equality follows from Equation 0.10.

<!-- formula-not-decoded -->

## 1.6 Bibliographic Remarks and Further Reading

Werefer the reader to [Puterman, 1994] for a more detailed treatment of dynamic programming and MDPs. [Puterman, 1994] also contains a thorough treatment of the dual LP, along with a proof of Lemma 1.15

With regards to the computational complexity of policy iteration, [Ye, 2011] showed that policy iteration is a strongly polynomial time algorithm for a fixed discount rate 1 . Also, see [Ye, 2011] for a good summary of the computational complexities of various approaches. [Mansour and Singh, 1999] showed that the number of iterations of policy iteration can be bounded by |A| |S| |S| .

With regards to a strongly polynomial algorithm, the CIPA algorithm [Ye, 2005] is an interior point algorithm with the claimed runtime in Table 0.1.

Lemma 1.11 is due to Singh and Yee [1994].

The performance difference lemma is due to [Kakade and Langford, 2002, Kakade, 2003], though the lemma was implicit in the analysis of a number of prior works.

1 The stated strongly polynomial runtime in Table 0.1 for policy iteration differs from that in [Ye, 2011] due to we assume that the runtime per iteration of policy iteration is |S| 3 + |S| 2 |A| .

## Chapter 2

## Sample Complexity with a Generative Model

This chapter begins our study of the sample complexity, where we focus on the (minmax) number of transitions we need to observe in order to accurately estimate Q glyph[star] or in order to find a near optimal policy. We assume that we have access to a generative model (as defined in Section 1.4) and that the reward function is deterministic (the latter is often a mild assumption, due to that much of the difficulty in RL is due to the uncertainty in the transition model P ).

This chapter follows the results due to [Azar et al., 2013], along with some improved rates due to [Agarwal et al., 2020c]. One of the key observations in this chapter is that we can find a near optimal policy using a number of observed transitions that is sublinear in the model size, i.e. use a number of samples that is smaller than O ( |S| 2 |A| ) . In other words, we do not need to learn an accurate model of the world in order to learn to act near optimally.

Notation. We define ̂ M to be the empirical MDP that is identical to the original M , except that it uses ̂ P instead of P for the transition model. When clear from context, we drop the subscript on M on the values, action values (and one-step variances and variances which we define later). We let ̂ V π , ̂ Q π , ̂ Q glyph[star] , and ̂ π glyph[star] denote the value function, state-action value function, optimal state-action value, and optimal policy in ̂ M , respectively.

## 2.1 Warmup: a naive model-based approach

A central question in this chapter is: Do we require an accurate model of the world in order to find a near optimal policy? Recall that a generative model takes as input a state action pair ( s, a ) and returns a sample s ′ ∼ P ( ·| s, a ) and the reward r ( s, a ) (or a sample of the reward if the rewards are stochastic). Let us consider the most naive approach to learning (when we have access to a generative model): suppose we call our simulator N times at each state action pair. Let ̂ P be our empirical model, defined as follows:

<!-- formula-not-decoded -->

where count ( s ′ , s, a ) is the number of times the state-action pair ( s, a ) transitions to state s ′ . As the N is the number of calls for each state action pair, the total number of calls to our generative model is |S||A| N . As before, we can view ̂ P as a matrix of size |S||A| × |S| .

Note that since P has a |S| 2 |A| parameters, we would expect that observing O ( |S| 2 |A| ) transitions is sufficient to provide us with an accurate model. The following proposition shows that this is the case.

Proposition 2.1. There exists an absolute constant c such that the following holds. Suppose glyph[epsilon1] ∈ ( 0 , 1 1 -γ ) and that we obtain where we uniformly sample every state action pair. Then, with probability greater than 1 -δ , we have:

<!-- formula-not-decoded -->

- (Model accuracy) The transition model has error bounded as:

<!-- formula-not-decoded -->

- (Uniform value accuracy) For all policies π ,

which proves the claim.

Lemma 2.3. For any policy π , MDP M and vector v ∈ R |S|×|A| , we have ∥ ∥ ( I -γP π ) -1 v ∥ ∥ ∞ ≤ ‖ v ‖ ∞ / (1 -γ ) .

Proof: Note that v = ( I -γP π )( I -γP π ) -1 v = ( I -γP π ) w , where w = ( I -γP π ) -1 v . By triangle inequality, we have

<!-- formula-not-decoded -->

where the final inequality follows since P π w is an average of the elements of w by the definition of P π so that ‖ P π w ‖ ∞ ≤ ‖ w ‖ ∞ . Rearranging terms completes the proof.

Now we are ready to complete the proof of our proposition.

Proof: Using the concentration of a distribution in the glyph[lscript] 1 norm (Lemma A.8), we have that for a fixed s, a that, with probability greater than 1 -δ , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (Near optimal planning) Suppose that ̂ π is the optimal policy in ̂ M . We have that:

<!-- formula-not-decoded -->

Before we provide the proof, the following lemmas will be helpful throughout:

Lemma 2.2. (Simulation Lemma) For all π we have that:

<!-- formula-not-decoded -->

Proof: Using our matrix equality for Q π (see Equation 0.2), we have:

<!-- formula-not-decoded -->

where m is the number of samples used to estimate ̂ P ( ·| s, a ) . The first claim now follows by the union bound (and redefining δ and c appropriately).

For the second claim, we have that:

<!-- formula-not-decoded -->

where the penultimate step uses Holder's inequality. The second claim now follows.

For the final claim, first observe that | sup x f ( x ) -sup x g ( x ) | ≤ sup x | f ( x ) -g ( x ) | , where f and g are real valued functions. This implies:

<!-- formula-not-decoded -->

which proves the first inequality. The second inequality is left as an exercise to the reader.

## 2.2 Sublinear Sample Complexity

In the previous approach, we are able to accurately estimate the value of every policy in the unknown MDP M . However, with regards to planning, we only need an accurate estimate ̂ Q glyph[star] of Q glyph[star] , which we may hope would require less samples. Let us now see that the model based approach can be refined to obtain minmax optimal sample complexity, which we will see is sublinear in the model size.

We will state our results in terms of N , and recall that N is the # of calls to the generative models per state-action pair, so that:

# samples from generative model = |S||A| N.

Let us start with a crude bound on the optimal action-values, which provides a sublinear rate. In the next section, we will improve upon this to obtain the minmax optimal rate.

Proposition 2.4. (Crude Value Bounds) Let δ ≥ 0 . With probability greater than 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the first inequality above shows a sublinear rate on estimating the value function. Ultimately, we are interested in the value V ̂ π glyph[star] when we execute ̂ π glyph[star] , not just an estimate ̂ Q glyph[star] of Q glyph[star] . Here, by Lemma 1.11, we lose an additional horizon factor and have:

<!-- formula-not-decoded -->

As we see in Theorem 2.6, this is improvable.

Before we provide the proof, the following lemma will be helpful throughout.

where:

Lemma 2.5. (Component-wise Bounds) We have that:

<!-- formula-not-decoded -->

Proof: For the first claim, the optimality of π glyph[star] in M implies:

<!-- formula-not-decoded -->

where we have used Lemma 2.2 in the final step. This proves the first claim.

For the second claim,

<!-- formula-not-decoded -->

glyph[star] glyph[star]

where the inequality follows from ̂ P ̂ π Q glyph[star] ≤ ̂ P π Q glyph[star] , due to the optimality of π glyph[star] . This proves the second claim. Proof: Following from the simulation lemma (Lemma 2.2) and Lemma 2.3, we have:

<!-- formula-not-decoded -->

Also, the previous lemma, implies that:

<!-- formula-not-decoded -->

By applying Hoeffding's inequality and the union bound,

<!-- formula-not-decoded -->

which holds with probability greater than 1 -δ . This completes the proof.

## 2.3 Minmax Optimal Sample Complexity (and the Model Based Approach)

We now see that the model based approach is minmax optimal, for both the discounted case and the finite horizon setting.

## 2.3.1 The Discounted Case

Upper bounds. The following theorem refines our crude bound on ̂ Q glyph[star] .

Theorem 2.6. For δ ≥ 0 and for an appropriately chosen absolute constant c , we have that:

- (Value estimation) With probability greater than 1 -δ ,

<!-- formula-not-decoded -->

- (Sub-optimality) If N ≥ 1 (1 -γ ) 2 , then with probability greater than 1 -δ ,

<!-- formula-not-decoded -->

This immediately provides the following corollary.

Corollary 2.7. Provided that glyph[epsilon1] ≤ 1 and that

<!-- formula-not-decoded -->

then with probability greater than 1 -δ ,

Furthermore, provided that glyph[epsilon1] ≤ √ 1 1 -γ and that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then with probability greater than 1 -δ ,

<!-- formula-not-decoded -->

We only prove the first claim in Theorem 2.6 on the estimation accuracy. With regards to the sub-optimality, note that Theorem 1.11 already implies a sub-optimality gap, though with an amplification of the estimation error by 2 / (1 -γ ) . The argument for the improvement provided in the second claim is more involved (See Section 2.6 for further discussion).

Lower Bounds. Let us say that an estimation algorithm A , which is a map from samples to an estimate ̂ Q glyph[star] , is ( glyph[epsilon1], δ ) -good on MDP M if ‖ Q glyph[star] -̂ Q glyph[star] ‖ ∞ ≤ glyph[epsilon1] holds with probability greater than 1 -δ .

Theorem 2.8. There exists glyph[epsilon1] 0 , δ 0 , c and a set of MDPs M such that for glyph[epsilon1] ∈ (0 , glyph[epsilon1] 0 ) and δ ∈ (0 , δ 0 ) if algorithm A is ( glyph[epsilon1], δ ) -good on all M ∈ M , then A must use a number of samples that is lower bounded as follows

<!-- formula-not-decoded -->

In other words, this theorem shows that the model based approach minmax optimal.

## 2.3.2 Finite Horizon Setting

Recall the setting of finite horizon MDPs defined in Section 1.2. Again, we can consider the most naive approach to learning (when we have access to a generative model): suppose we call our simulator N times for every ( s, a, h ) ∈ S × A × [ H ] , i.e. we obtain N i.i.d. samples where s ′ ∼ P h ( ·| s, a ) , for every ( s, a, h ) ∈ S × A × [ H ] . Note that the total number of observed transitions is H |S||A| N .

Upper bounds. The following theorem provides an upper bound on the model based approach.

Theorem 2.9. For δ ≥ 0 and with probability greater than 1 -δ , we have that:

- (Value estimation)
- (Sub-optimality)

Equivalently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c is an absolute constant.

Note that the above bound requires N to be O ( H 2 ) in order to achieve an glyph[epsilon1] -optimal policy, while in the discounted case, we require N to be O (1 / (1 -γ ) 3 ) for the same guarantee. While this may seem like an improvement by a horizon factor, recall that for the finite horizon case, N corresponds to observing O ( H ) more transitions than in the discounted case.

Lower Bounds. In the minmax sense of Theorem 2.8, the previous upper bound provided by the model based approach for the finite horizon setting achieves the minmax optimal sample complexity.

## 2.4 Analysis

We now prove (the first claim in) Theorem 2.6.

## 2.4.1 Variance Lemmas

The key to the sharper analysis is to more sharply characterize the variance in our estimates.

Denote the variance of any real valued f under a distribution D as:

<!-- formula-not-decoded -->

Slightly abusing the notation, for V ∈ R |S| , we define the vector Var P ( V ) ∈ R |S||A| as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we characterize a relevant deviation in terms of the its variance.

Lemma 2.10. Let δ &gt; 0 . With probability greater than 1 -δ ,

<!-- formula-not-decoded -->

Proof: The claims follows from Bernstein's inequality along with a union bound over all state-action pairs.

The key ideas in the proof are in how we bound ‖ ( I -γ ̂ P π glyph[star] ) -1 √ Var P ( V glyph[star] ) ‖ ∞ and ‖ ( I -γ ̂ P ̂ π glyph[star] ) -1 √ Var P ( V glyph[star] ) ‖ ∞ . It is helpful to define Σ π M as the variance of the discounted reward, i.e.

<!-- formula-not-decoded -->

where the expectation is induced under the trajectories induced by π in M . It is straightforward to verify that ‖ Σ π M ‖ ∞ ≤ γ 2 / (1 -γ ) 2 .

The following lemma shows that Σ π M satisfies a Bellman consistency condition.

Lemma 2.11. (Bellman consistency of Σ ) For any MDP M ,

<!-- formula-not-decoded -->

where P is the transition model in MDP M .

The proof is left as an exercise to the reader.

Lemma 2.12. (Weighted Sum of Deviations) For any policy π and MDP M ,

<!-- formula-not-decoded -->

where P is the transition model of M .

Proof: Note that (1 -γ )( I -γP π ) -1 is matrix whose rows are a probability distribution. For a positive vector v and a distribution ν (where ν is vector of the same dimension of v ), Jensen's inequality implies that ν · √ v ≤ √ ν · v . This implies:

<!-- formula-not-decoded -->

where we have used that ‖ ( I -γP π ) -1 v ‖ ∞ ≤ 2 ‖ ( I -γ 2 P π ) -1 v ‖ ∞ (which we will prove shortly). The proof is completed as follows: by Equation 0.1, Σ π M = γ 2 ( I -γ 2 P π ) -1 Var P ( V π M ) , so taking v = Var P ( V π M ) and using that ‖ Σ π M ‖ ∞ ≤ γ 2 / (1 -γ ) 2 completes the proof.

Finally, to see that

<!-- formula-not-decoded -->

which proves the claim.

## 2.4.2 Completing the proof

Lemma 2.13. Let δ ≥ 0 . With probability greater than 1 -δ , we have:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: By definition,

<!-- formula-not-decoded -->

Nowweboundeach of these terms with Hoeffding's inequality and the union bound. For the first term, with probability greater than 1 -δ ,

For the second term, again with probability greater than 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used that ( · ) 2 is a component-wise operation in the second step. For the last term:

<!-- formula-not-decoded -->

where ∆ δ,N is defined in Proposition 2.4. To obtain a cumulative probability of error less than δ , we replace δ in the above claims with δ/ 3 . Combining these bounds completes the proof of the first claim. The argument in the above display also implies that Var ̂ P ( V glyph[star] ) ≤ 2∆ 2 δ,N +2Var ̂ P ( ̂ V glyph[star] ) which proves the second claim.

Using Lemma 2.10 and 2.13, we have the following corollary.

Corollary 2.14. Let δ ≥ 0 . With probability greater than 1 -δ , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and where c is an absolute constant.

Proof: (of Theorem 2.6) The proof consists of bounding the terms in Lemma 2.5. We have:

<!-- formula-not-decoded -->

where the first step uses Corollary 2.14; the second uses Lemma 2.12; and the last step uses that 2 ab ≤ a 2 + b 2 (and choosing a, b appropriately). The proof of the lower bound is analogous. Taking a different absolute constant completes the proof.

## 2.5 Scalings and Effective Horizon Dependencies

It will be helpful to more intuitively understand why 1 / (1 -γ ) 3 is the effective horizon dependency one might hope to expect, from a dimensional analysis viewpoint. Due to that Q glyph[star] is a quantity that is as large as 1 / (1 -γ ) , to account for this scaling, it is natural to look at obtaining relative accuracy.

In particular, if then with probability greater than 1 -δ , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(provided that glyph[epsilon1] ≤ √ 1 -γ using Theorem 2.6). In other words, if we had normalized the value functions 1 , then for additive accuracy (on our normalized value functions) our sample size would scale linearly with the effective horizon.

## 2.6 Bibliographic Remarks and Further Readings

The notion of a generative model was first introduced in [Kearns and Singh, 1999], which made the argument that, up to horizon factors and logarithmic factors, both model based methods and model free methods are comparable. [Kakade, 2003] gave an improved version of this rate (analogous to the crude bounds seen here).

The first claim in Theorem 2.6 is due to [Azar et al., 2013], and the proof in this section largely follows this work. Improvements are possible with regards to bounding the quality of ̂ π glyph[star] ; here, Theorem 2.6 shows that the model based approach is near optimal even for policy itself; showing that the quality of ̂ π glyph[star] does suffer any amplification factor of 1 / (1 -γ ) . [Sidford et al., 2018] provides the first proof of this improvement using a variance reduction algorithm with value iteration. The second claim in Theorem 2.6 is due to [Agarwal et al., 2020c], which shows that the naive model based approach is sufficient. The lower bound in Theorem 2.8 is due to [Azar et al., 2013].

1 Rescaling the value functions by multiplying by (1 -γ ) , i.e. Q π ← (1 -γ ) Q π , would keep the values bounded between 0 and 1 . Throughout,

We also remark that we may hope for the sub-optimality bounds (on the value of the argmax policy) to hold up to for 'large' glyph[epsilon1] , i.e. up to glyph[epsilon1] ≤ 1 / (1 -γ ) (see the second claim in Theorem 2.6). Here, the work in [Li et al., 2020] shows this limit is achievable, albeit with a slightly different algorithm where they introduce perturbations. It is currently an open question if the naive model based approach also achieves the non-asymptotic statistical limit.

This chapter also provided results, without proof, on the optimal sample complexity in the finite horizon setting (see Section 2.3.2). The proof of this claim would also follow from the line of reasoning in [Azar et al., 2013], with the added simplification that the sub-optimality analysis is simpler in the finite-horizon setting with time-dependent transition matrices (e.g. see [Yin et al., 2021]).

this book it is helpful to understand sample size with regards to normalized quantities.

## Chapter 3

## Linear Bellman Completeness

Up to now we have focussed on 'tabular' MDPs, where the sample complexity (and the computational complexity) scaled polynomially in the size of the state and action spaces. Ultimately, we seek to obtain methods which are applicable to cases where number of states and actions is large (or, possibly, countably or uncountably infinite).

In this chapter, we will consider one of the simplest conditions, where such a result is possible. In particular, we will consider a setting where we have a set of features which satisfy the linear Bellman completion condition, and we show that, with access to a generative model, there is a simple algorithm that learns a near optimal policy with polynomial sample complexity in the dimension of these features.

The linear Bellman completeness concept and first analysis was due to [Munos, 2005]. This chapter studies this concept in the setting of finite horizon MDPs, where we have access to a generative model.

## 3.1 The Linear Bellman Completeness Condition

We will work with given a feature mapping φ : S × A ↦→ R d , where we assume the following property is true: given any linear function f ( s, a ) := θ glyph[latticetop] φ ( s, a ) , we have that the Bellman operator applied to f ( s, a ) also returns a linear function with respect to φ . Precisely, we have:

Definition 3.1 (Linear Bellman Completeness) . Wesay the features φ satisfy the linear Bellman completeness property if for all θ ∈ R d and ( s, a, h ) ∈ S × A × [ H ] , there exists w ∈ R d such that:

<!-- formula-not-decoded -->

As w depends on θ , we use the notation T h : R d ↦→ R d to represent such a w , i.e., w := T h ( θ ) in the above equation.

Note that the above implies that r ( s, a ) is in the span of φ (to see this, take θ = 0 ). Furthermore, it also implies that Q glyph[star] h ( s, a ) is linear in φ , i.e., there exists θ glyph[star] h such that Q glyph[star] h ( s, a ) = ( θ glyph[star] h ) glyph[latticetop] φ ( s, a ) . We can easily see that tabular MDP is captured by the linear Bellman completeness with φ ( s, a ) ∈ R |S||A| being a one-hot encoding vector with zeros everywhere except one at the entry corresponding to state-action ( s, a ) . In Chapters 8 and 9, we will provide precise examples of models which satisfy the linear Bellman completeness condition.

We focus on the generative model setting, i.e., where we can input any ( s, a ) pair to obtain a next state sample s ′ ∼ P ( s, a ) and reward r ( s, a ) . Our goal here is to design an algorithm that finds a policy ̂ π such that V ̂ π ≥ V glyph[star] -glyph[epsilon1] , with number of samples scaling polynomially with respect to all relevant parameters d, H, 1 /glyph[epsilon1] .

## Algorithm 1 Least Squares Value Iteration

- 1: Input : D 0 , . . . , D H -1
- 3: for h = H -1 → 0 do
- 2: Set V H ( s ) = 0 for all s ∈ S
- 4: Solve least squares
- 6: end for

<!-- formula-not-decoded -->

- 5: Set V h ( s ) = max a ∈A θ glyph[latticetop] h φ ( s, a ) , ∀ s
- 7: Return: { ̂ π h ( s ) } H -1 h =0 where ̂ π h ( s ) := arg max a θ glyph[latticetop] h φ ( s, a ) , ∀ s, h .

## 3.2 The LSVI Algorithm

We first present the least square value iteration (LSVI) algorithm (Alg. 1) here. The algorithm takes H many datasets D 0 , . . . , D H -1 as inputs, where each D h = { s, a, r, s ′ } .

For each time step, LSVI estimate Q glyph[star] h ( s, a ) via θ glyph[latticetop] h φ ( s, a ) with θ h computed from a least square problem.

To make sure the algorithm succeeds in terms of finding a near optimal policy, we need to design the datasets D 0 , . . . , D H -1 . Intuitively, we may want to make sure that in each D h , we have ∑ s,a ∈D h φ ( s, a ) φ ( s, a ) glyph[latticetop] being full rank, so that least square has closed-form solution, and θ h has a good generalization bound. In the next section, we will leverage the generative model setting and the D-optimal design to construct such datasets.

## 3.3 LSVI with D-Optimal Design

In this section, we start by introducing the D-optimal design and its properties. We then use the D-optimal design to construct a dataset and give a generalization bound for the solution to the least squares problem on the constructed dataset. Finally, we present the sample complexity bound for LSVI and its analysis.

We will use the notation ‖ x ‖ 2 M = x glyph[latticetop] Mx for a matrix M and vector x of appropriate dimensions.

## 3.3.1 D-Optimal Design

We now specify a sampling distribution which, roughly speaking, ensures good coverage (in a spectral sense) over our feature set.

Theorem 3.2. Suppose X ⊂ R d is a compact set (and full dimensional). There exists a distribution ρ on X such that:

- ρ is supported on at most d ( d +1) / 2 points (all of which are in X ).
- Define

<!-- formula-not-decoded -->

We have that for all x ∈ X ,

<!-- formula-not-decoded -->

The distribution ρ is referred to as the D-optimal design.

With these two properties, one can show that for any x ∈ X , it can be written as a linear combination of the points on the support of ρ , i.e., x = ∑ x i ∈ support ( ρ ) α i ρ ( x i ) x i , where ‖ α ‖ 2 2 ≤ d .

The following specifies an explicit construction: ρ is the following maximizer:

<!-- formula-not-decoded -->

(and there exists an optimizer which is supported on at most d ( d + 1) / 2 points). The design ρ has a geometric interpretation: the centered ellipsoid E = { v : ‖ v ‖ 2 Σ -1 ≤ d } is the unique minimum volume centered ellipsoid containing X We do not provide the proof here (See Section 3.6).

In our setting, we will utilize the D-optimal design on the set Φ := { φ ( s, a ) : s, a ∈ S × A} .

Change of basis interpretation with D-optimal design. Consider the coordinate transformation:

<!-- formula-not-decoded -->

˜

<!-- formula-not-decoded -->

Furthermore, we still have that ‖ ˜ x ‖ 2 ˜ Σ -1 ≤ d . In this sense, we can interpret the D-optimal design as providing a well-conditioned sampling distribution for the set X .

## 3.3.2 Performance Guarantees

Now we explain how we can construct a dataset D h for h = 0 , . . . , H -1 using the D-optimal design ρ and the generative model:

- At h , for each s, a ∈ support ( ρ ) , we sample n s,a := glyph[ceilingleft] ρ ( s, a ) N glyph[ceilingright] many next states independently from P h ( ·| s, a ) ; combine all ( s, a, r ( s, a ) , s ′ ) and denote the whole dataset as D h
- Repeat for all h ∈ [0 , . . . , H -1] to construct D 0 , . . . , D H -1 .

The key property of the dataset D h is that the empirical covariance matrix ∑ s,a ∈D h φ ( s, a ) φ ( s, a ) glyph[latticetop] is full rank. Denote Λ h = ∑ s,a ∈D h φ ( s, a ) φ ( s, a ) glyph[latticetop] , we can verify that

<!-- formula-not-decoded -->

where the inequality comes from the fact that in D h , for each s, a ∈ support ( ρ ) , we have glyph[ceilingleft] ρ ( s, a ) N glyph[ceilingright] many copies of it. The D-optimal design ensures that Λ covers all φ ( s, a ) in the sense that max s,a φ ( s, a ) glyph[latticetop] Λ -1 h φ ( s, a ) glyph[latticetop] ≤ d due to the property of the D-optimal design. This is the key property that ensures the solution of the least squares on D h is accurate globally.

Now we state the main sample complexity bound theorem for LSVI.

Theorem 3.3 (Sample Complexity of LSVI) . Suppose our features satisfy the linear Bellman completion property. Fix δ ∈ (0 , 1) and glyph[epsilon1] ∈ (0 , 1) . Set parameter N := ⌈ 64 H 6 d 2 ln(1 /δ ) glyph[epsilon1] 2 ⌉ . With probability at least 1 -δ , Algorithm 1 outputs ̂ π such that:

<!-- formula-not-decoded -->

with total number of samples H ( d 2 + 64 H 6 d 2 ln(1 /δ ) glyph[epsilon1] 2 ) .

Here we have that:

## 3.3.3 Analysis

To prove the theorem, we first show that if we had { ̂ Q h } h such that the Bellman residual ‖ ̂ Q h -T h ̂ Q h +1 ‖ ∞ is small, then the performance of the greedy policies with respect to ̂ Q h is already near optimal. Note that this result is general and has nothing to do with the LSVI algorithm.

Lemma 3.4. Assume that for all h , we have ∥ ∥ ∥ ̂ Q h -T h +1 ̂ Q h +1 ∥ ∥ ∥ ∞ ≤ glyph[epsilon1] . Then we have:

<!-- formula-not-decoded -->

2. Policy performance: for ̂ π h ( s ) := argmax a ̂ Q h ( s, a ) , we have ∣ ∣ V ̂ π -V glyph[star] ∣ ∣ ≤ 2 H 2 glyph[epsilon1] .

Proof: The proof of this lemma is elementary, which is analog to what we have proved in Value Iteration in the discounted setting. For completeness, we provide a proof here.

Starting from Q H ( s, a ) = 0 , ∀ s, a , we have T H -1 Q H = r by definition. The condition implies that ‖ ̂ Q H -1 -r ‖ ∞ ≤ glyph[epsilon1] , which means that ‖ ̂ Q H -1 -Q glyph[star] H -1 ‖ ∞ ≤ glyph[epsilon1] . This proves the base case.

Our inductive hypothesis is that ‖ ̂ Q h +1 -Q glyph[star] h +1 ‖ ∞ ≤ ( H -h -1) glyph[epsilon1] . Note that:

<!-- formula-not-decoded -->

which concludes the proof for the first claim.

For the second claim, Starting from time step H -1 , for any s , we have:

<!-- formula-not-decoded -->

where the third equality uses the fact that Q ̂ π H -1 ( s, a ) = Q glyph[star] H -1 ( s, a ) = r ( s, a ) , the first inequality uses the definition of ̂ π H -1 and ̂ Q H -1 ( s, ̂ π H -1 ( s )) ≥ ̂ Q H -1 ( s, a ) , ∀ a . This concludes the base case. For time step h +1 , our inductive hypothesis is defined as V ̂ π h +1 ( s ) -V glyph[star] h +1 ( s ) ≥ -2( H -h -1) Hglyph[epsilon1] . For time step h , we have:

<!-- formula-not-decoded -->

This concludes that for any s , at time step h = 0 , we have V ̂ π 0 ( s ) -V glyph[star] 0 ( s ) ≥ -2 H 2 glyph[epsilon1] .

To conclude the proof, what we left to show is that the ̂ Q h that is returned by LSVI satisfies the condition in the above lemma. Note that ̂ Q h ( s, a ) = ˆ θ glyph[latticetop] h φ ( s, a ) , and T h ̂ Q h +1 is indeed the Bayes optimal solution of the least squares in Eq. 0.1, and via the Bellman completion assumption, we have T h ̂ Q h +1 ( s, a ) = T h ( ˆ θ h +1 ) glyph[latticetop] φ ( s, a ) , ∀ s, a . With the constructed D h based on the D-optimal design, Linear regression immediately gives us a generalization bound on ‖ ̂ Q h -T h ̂ Q h +1 ‖ ∞ . The following theorem formalize the above statement.

Lemma 3.5. Fix δ ∈ (0 , 1) and glyph[epsilon1] ∈ (0 , 1) , set N = ⌈ 16 H 2 d 2 ln( H/δ ) glyph[epsilon1] 2 ⌉ . With probability at least 1 -δ , for all h , we have:

<!-- formula-not-decoded -->

Proof: The proof uses a standard result for ordinary least squares (see Theorem A.10 in the Appendix). Consider a time step h . In time step h , we perform linear regression from φ ( s, a ) to r ( s, a ) + max a ′ ˆ θ glyph[latticetop] h +1 φ ( s ′ , a ′ ) . Note that the Bayes optimal here is

<!-- formula-not-decoded -->

Also we can show that the noise glyph[epsilon1] s,a := ( r ( s, a ) + max a ′ ̂ Q h +1 ( s ′ , a ′ ) -T h ̂ Q h ( s, a ) ) is bounded by 2 H , and also note that the noises are independent, and has mean zero: E s ′ ∼ P h ( s,a ) [ r ( s, a ) + max a ′ ̂ Q h +1 ( s ′ , a ′ ) -T h ̂ Q h ( s, a ) ] = 0 . Using Theorem A.10, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Now for any s, a , we must have:

<!-- formula-not-decoded -->

which implies that:

<!-- formula-not-decoded -->

Set the right hand side to be glyph[epsilon1] , we have N = 16 H 2 d 2 ln(1 /δ ) glyph[epsilon1] 2 .

Finally, apply union bound over all h = 0 , . . . , H -1 , we conclude the proof.

Now we are ready to conclude the proof for the main theorem.

Proof: [Proof of Theorem 3.3] Using Lemma 3.5, with N = ⌈ 64 H 6 d 2 ln( H/δ ) glyph[epsilon1] 2 ⌉ , we have that ‖ ̂ Q h - T h ̂ Q h +1 ‖ ∞ ≤ glyph[epsilon1]/ (2 H 2 ) for all h , with probability at least 1 -δ . Now apply Lemma 3.4, we immediately have that | V ˆ π -V glyph[star] | ≤ glyph[epsilon1] . Regarding sample complexity, we have:

<!-- formula-not-decoded -->

where the second inequality use the support size upper bound of the D-optimal design ρ (see Lemma 3.2). This completes the proof.

## 3.4 How Strong is Bellman Completion as a Modeling?

To be added.

## 3.5 Offline Reinforcement Learning

One notable observation about LSVI is that the algorithm is non-adaptive, in the sense that it works using a fixed set of observed transitions and rewards. Hence, it is applicable to the offline setting discussed in Chapter 1.

We now discuss two offline objectives: that of learning a near optimal policy and that of policy evaluation (i.e. evaluating the quality of a given policy).

We consider the setting where have a H datasets of the form D h = { ( s i , a i , s ′ i , r ( s i , a i )) } N i =1 , where we assume for each i and h , that we have independent samples s ′ i ∼ P h ( ·| s i , a i ) . In other words, D h is a dataset of observed transitions corresponding to stage h , where each next state has been sampled independently. Note that here we have not made explicit distributional assumption about how the ( s i , a i ) 's have been generated. The results we present can also be extended to where we have a fixed data generation distribution over these quadruples.

It is straight forward to extend the LSVI guarantees to these setting provided our dataset has coverage in the following sense:

Assumption 3.6. (Coverage) Suppose that for each h ∈ [ H ] , we have that:

<!-- formula-not-decoded -->

where Σ is the D-optimal design covariance (see Equation 0.2).

## 3.5.1 Offline Learning

A minor modification of the proof in Theorem 3.3 leads to the following guarantee:

Theorem 3.7 (Sample Complexity of LSVI) . Suppose that Assumption 3.6 holds and that our features satisfy the linear Bellman completion property. Fix δ ∈ (0 , 1) and glyph[epsilon1] ∈ (0 , 1) . Set parameter N := ⌈ cκH 6 d 2 ln(1 /δ ) glyph[epsilon1] 2 ⌉ , where c is an appropriately chosen absolute constant. With probability at least 1 -δ , Algorithm 1 outputs ̂ π such that:

<!-- formula-not-decoded -->

## 3.5.2 Offline Policy Evaluation

Here we are interested in question of evaluating some given policy π using the offline data. For this, we will make a completeness assumption with respect to π as follows:

Definition 3.8 (Linear Completeness for π ) . We say the features φ satisfy the policy completeness property for π if for all θ ∈ R d and ( s, a, h ) ∈ S × A × [ H ] , there exists w ∈ R d such that:

<!-- formula-not-decoded -->

For this case, Algorithm 2, Least Squares Policy Evaluation (LSPE), is a modified version of LSVI for the purposes of estimation; note the algorithm no longer estimates V h ( s ) using the greedy policy. Again, a nearly identical argument to the proof in Theorem 3.3 leads to the following guarantee. Here, we set:

<!-- formula-not-decoded -->

where θ 0 is the parameter returned by LSVI.

## Algorithm 2 Least Squares Policy Evaluation

- 1: Input : π , D 0 , . . . , D H -1
- 3: for h = H -1 → 0 do
- 2: Set V H ( s ) = 0 for all s ∈ S
- 4: Solve least squares
- 6: end for

<!-- formula-not-decoded -->

- 5: Set V h ( s ) = ̂ θ glyph[latticetop] h φ ( s, π ( s )) , ∀ s
- 7: Return: { ̂ θ h } H -1 h =0 .

Theorem 3.9 (Sample Complexity of LSPE) . Suppose that Assumption 3.6 holds and that our features satisfy the linear policy completion property (with respect to π ). Fix δ ∈ (0 , 1) and glyph[epsilon1] ∈ (0 , 1) . Set parameter N := ⌈ cκH 4 d 2 ln(1 /δ ) glyph[epsilon1] 2 ⌉ , where c is an appropriately chosen absolute constant. With probability at least 1 -δ , Algorithm 2 outputs ̂ θ 0 such that for all s ,

<!-- formula-not-decoded -->

Note that the above theorem has a sample complexity improvement by a factor of H . This is due to that the analysis is improvable, as we only care about value accuracy (the first claim in Lemma 3.4, rather than the second, is what is relevant here). We should note that the sample size bounds presented in this chapter have not been optimized with regards to their H dependencies.

## 3.6 Bibliographic Remarks and Further Readings

The idea of Bellman completion under general function class was introduced in [Munos, 2005] under the setting of batch RL. For the episodic online learning setting, Zanette et al. [2020] provided a statistically efficient algorithm under the linear Bellman completion condition, and [Jin et al., 2021] proposes a statistically efficient algorithms under the Bellman completion condition with general function approximation.

We refer readers to to [Lattimore and Szepesv´ ari, 2020] for a proof of the D-optimal design; the idea directly follows from John's theorem (e.g. see [Ball, 1997, Todd, 2016]).

## Chapter 4

## Fitted Dynamic Programming Methods

Let us again consider the question of learning in large MDPs, when the underlying MDPs is unknown. In the previous chapter, we relied on the linear Bellman completeness assumption to provide strong guarantees with only polynomial dependence on the dimension of the feature space and the horizon H . This chapter considers the approach of using function approximation methods with iterative dynamic programming approaches.

In particular, we now consider approaches which rely on (average-case) supervised learning methods (namely regression), where we use regression to approximate the target functions in both the value iteration and policy iteration algorithms. We refer to these algorithms as fitted Q -iteration (FQI) and fitted policy iteration (FPI). The FQI algorithm can be implemented in an offline RL sampling model while the FPI algorithm requires the sampling access to an episodic model (see Chapter 1 for review of these sampling models).

This chapter focuses on obtaining of average case function approximation error bounds, provided we have a somewhat stringent condition on how the underlying MDP behaves, quantified by the concentrability coefficient . This notion was introduced in [Munos, 2003, 2005]. While the notion is somewhat stringent, we will see that it is not avoidable without further assumptions. The next chapter more explicitly considers lower bounds, while Chapters 13 and 14 seek to relax the concentrability notion.

## 4.1 Fitted Q -Iteration (FQI) and Offline RL

We consider infinite horizon discounted MDP M = {S , A , γ, P, r, µ } where µ is the initial state distribution. We assume reward is bounded, i.e., sup s,a r ( s, a ) ∈ [0 , 1] . For notation simplicity, we denote V max := 1 / (1 -γ ) .

Given any f : S × A ↦→ R , we denote the Bellman operator T f : S × A ↦→ R as follows. For all s, a ∈ S × A ,

<!-- formula-not-decoded -->

We assume that we have a distribution ν ∈ ∆( S × A ) . We collect a dataset D := { ( s i , a i , r i , s ′ i ) } n i =1 where s i , a i ∼ ν, r i = r ( s i , a i ) , s ′ i ∼ P ( ·| s i , a i ) . Given D , our goal is to output a near optimal policy for the MDP, that is we would like our algorithm to produce a policy ˆ π such that, with probability at least 1 -δ , V (ˆ π ) ≥ V glyph[star] -glyph[epsilon1] , for some ( glyph[epsilon1], δ ) pair. As usual, the number of samples n will depend on the accuracy parameters ( glyph[epsilon1], δ ) and we would like n to scale favorably with these. Given any distribution ν ∈ S × A , and any function f : S × A ↦→ R , we write ‖ f ‖ 2 2 ,ν := E s,a ∼ ν f 2 ( s, a )

Denote a function class F = { f : S × A ↦→ [0 , V max ] } .

We require the data distribution ν is exploratory enough.

Assumption 4.1 (Concentrability) . There exists a constant C such that for any policy π (including non-stationary policies), we have:

<!-- formula-not-decoded -->

Note that concentrability does not require that the state space is finite, but it does place some constraints on the system dynamics. Note that the above assumption requires that ν to cover all possible policies's state-action distribution, even including non-stationary policies.

In additional to the above two assumptions, we need an assumption on the representational condition of class F .

Assumption 4.2 (Inherent Bellman Error) . We assume the following error bound holds:

<!-- formula-not-decoded -->

We refer to glyph[epsilon1] approx,ν as the inherent Bellman error with respect to the distribution ν .

## 4.1.1 The FQI Algorithm

Fitted Q Iteration (FQI) simply performs the following iteration. Start with some f 0 ∈ F , FQI iterates:

<!-- formula-not-decoded -->

After k many iterations, we output a policy π k ( s ) := argmax a f k ( s, a ) , ∀ s .

To get an intuition why this approach can work, let us assume the inherent Bellman error glyph[epsilon1] approx = 0 which is saying that for any f ∈ F , we have T f ∈ F (i.e., Bellman completion). Note that the Bayes optimal solution is T f t -1 . Due to the Bellman completion assumption, the Bayes optimal solution T f t -1 ∈ F . Thus, we should expect that f t is close to the Bayes optimal T f t -1 under the distribution ν , i.e., with a uniform convergence argument, for the generalization bound, we should expect that:

<!-- formula-not-decoded -->

Thus in high level, f t ≈ T f t -1 as our data distribution ν is exploratory, and we know that based on value iteration, T f k -1 is a better approximation of Q glyph[star] than f k , i.e., ‖T f t -1 -Q glyph[star] ‖ ∞ ≤ γ ‖ f t -1 -Q glyph[star] ‖ ∞ , we can expect FQI to converge to the optimal solution when n →∞ , t →∞ . We formalize the above intuition below.

## 4.1.2 Performance Guarantees of FQI

We first state the performance guarantee of FQI.

Theorem 4.3 (FQI guarantee) . Fix K ∈ N + . Fitted Q Iteration guarantees that with probability 1 -δ ,

<!-- formula-not-decoded -->

We start from the following lemma which shows that if for all t , ‖ f t +1 -T f t ‖ 2 2 ,ν ≤ ε , then the greedy policy with respect to f t is approaching to π glyph[star] .

Lemma 4.4. Assume that for all t , we have ‖ f t +1 -T f t ‖ 2 ,ν ≤ ε , then for any k , we have:

<!-- formula-not-decoded -->

Proof: We start from the Performance Difference Lemma:

<!-- formula-not-decoded -->

where the first inequality comes from the fact that π k is a greedy policy of f k , i.e., f k ( s, π k ( s )) ≥ f k ( s, a ) for any other a including π glyph[star] ( s ) . Now we bound each term on the RHS of the above inequality. We do this by consider a state-action distribution β that is induced by some policy. We have:

<!-- formula-not-decoded -->

where in the last inequality, we use the fact that ( E [ x ]) 2 ≤ E [ x 2 ] , (max x f ( x ) -max x g ( x )) 2 ≤ max x ( f ( x ) -g ( x )) 2 for any two functions f and g , and assumption 4.1.

Denote β ′ ( s ′ , a ′ ) = ∑ s,a β ( s, a ) P ( s ′ | s, a ) 1 { a ′ = argmax a ( Q glyph[star] ( s ′ , a ) -f k -1 ( s ′ , a )) 2 } , the above inequality becomes:

<!-- formula-not-decoded -->

We can recursively repeat the same process for ‖ Q glyph[star] -f k -1 ‖ 2 ,ν ′ till step k = 0 :

<!-- formula-not-decoded -->

where ν is some valid state-action distribution.

˜

Note that for the first term on the RHS of the above inequality, we can bound it as follow:

<!-- formula-not-decoded -->

For the second term, we have:

Thus, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all β , including β = d π k ◦ π glyph[star] , and β = d π k ◦ π k . This concludes the proof.

What left is to show that least squares control the error ε . We will use the generalization bound for least squares shown in Lemma A.11.

Lemma 4.5 (Bellman error) . With probability at least 1 -δ , for all t = 0 , . . . , K , we have:

<!-- formula-not-decoded -->

## Proof:

Let us fix a function g ∈ F , and consider the regression problem on dataset { φ ( s, a ) , r ( s, a ) + γ max a ′ g ( s ′ , a ′ ) } . Denote ˆ f g = argmin f ∈F ∑ n i =1 ( f ( s i , a i ) -r i -γ max a ′ g ( s ′ , a ′ )) , as the least square solution.

Note that in this regression problem, we have | r i + γ max a ′ g ( s ′ , a ′ ) | ≤ 1 + γV max ≤ 2 V max , and for our Bayes optimal solution, we have |T g ( s, a ) | = | r ( s, a ) + γ E s ′ ∼ P ( ·| s,a ) max a ′ g ( s ′ , a ′ ) | ≤ 1 + γV max ≤ 2 V max . Also note that our inherent Bellman error condition implies that min f ∈F E s,a ∼ ν ( f ( s, a ) - T g ( s, a )) 2 ≤ glyph[epsilon1] approx,ν . Thus, we can apply Lemma A.11 directly here. With probability at least 1 -δ , we get:

<!-- formula-not-decoded -->

Note that the above inequality holds for the fixed g . We apply a union bound over all possible g ∈ F , we get that with probability at least 1 -δ :

<!-- formula-not-decoded -->

Consider iteration t , set g = f t +1 in Eq. 0.2, and by the definition of f t , we have ˆ f g = f t in this case, which means that above inequality in Eq. 0.2 holds for the pair ( ˆ f g , g ) = ( f t , f t +1 ) . Now apply a union bound over all t = 0 , . . . , K , we conclude the proof.

Now we are ready to prove the main theorem.

Proof: [Proof of Theorem 4.3]

From Lemma 4.5, we see that with probability at least 1 -δ , we have for all t ≤ K :

<!-- formula-not-decoded -->

Set ε in Lemma 4.4 to be √ 22 V 2 max ln( |F| 2 K/δ ) n + √ 20 glyph[epsilon1] approx,ν , we conclude the proof.

## 4.2 Fitted Policy-Iteration (FPI)

to be added..

## 4.3 Failure Cases Without Assumption 4.1

## 4.4 FQI for Policy Evaluation

## 4.5 Bibliographic Remarks and Further Readings

The notion of concentrability was developed in [Munos, 2003, 2005] in order to permitting sharper bounds in terms of average case function approximation error, provided that the concentrability coefficient is bounded. These methods also permit sample based fitting methods, with sample size and error bounds, provided there is a data collection policy that induces a bounded concentrability coefficient [Munos, 2005, Szepesv´ ari and Munos, 2005, Antos et al., 2008, Lazaric et al., 2016]. Chen and Jiang [2019] provide a more detailed discussion on this quantity.

## Chapter 5

## Statistical Limits of Generalization

In reinforcement learning, we seek to have learnability results which are applicable to cases where number of states is large (or, possibly, countably or uncountably infinite). This is a question of generalization, which, more broadly, is one of the central challenges in machine learning.

The previous two chapters largely focussed on sufficient conditions under which we can obtain sample complexity results which do not explicitly depend on the size of the state (or action) space. This chapter focusses on what are necessary conditions for generalization. Here, we can frame our questions by examining the extent to which generalization in RL is similar to (or different from) that in supervised learning. Two most basic settings in supervised learning are: (i) agnostic learning (i.e. finding the best classifier or hypothesis in some class) and (ii) learning with linear models (i.e. learning the best linear regressor or the best linear classifier). This chapter will focus on lower bounds with regards to the analogues of these two questions for reinforcement learning:

- (Agnostic learning) Given some hypothesis class (of policies, value functions, or models), what is the sample complexity of finding (nearly) the best hypothesis in this class?
- (Linearly realizable values or policies) Suppose we are given some d -dimensional feature mapping where we are guaranteed that either the optimal value function is linear in these given features or that the optimal policy has a linear parameterization. Are we able to obtain sample complexity guarantees that are polynomial in d , with little to no explicit dependence on the size of the state or action spaces? We will consider this question in both the offline setting (for the purposes of policy evaluation, as in Chapter 3) and for in online setting where our goal is to learn a near optimal optimal policy.

Observe that supervised learning can be viewed as horizon one, H = 1 , RL problem (where the learner only receives feedback for the 'label', i.e. the action, chosen). We can view the second question above as the analogue of linear regression or classification with halfspaces. In supervised learning, both of these settings have postive answers, and they are fundamental in our understanding of generalization. Perhaps surprisingly, we will see negative answers to these questions in the RL setting. The significance of this provides insights as to why our study of generalization in reinforcement learning is substantially more subtle than in supervised learning. Importantly, the insights we develop here will also help us to motivate the more refined assumptions and settings that we consider in subsequent chapters (see Section 5.3 for discussion).

This chapter will work with finite horizon MDPs, where we consider both the episodic setting and the generative model setting. With regards to the first question on agnostic learning, this chapter follows the ideas first introduced in [Kearns et al., 2000]. With regards to the second question on linear realizability, this chapter follows the results in [Du et al., 2019, Wang et al., 2020a, Weisz et al., 2021a, Wang et al., 2021].

## 5.1 Agnostic Learning

Suppose we have a hypothesis class H (either finite or infinite), where for each f ∈ H we have an associated policy π f : S → A , which, for simplicity, we assume is deterministic. Here, we could have that:

- H itself is a class of policies.
- H is a set of state-action values, where for f ∈ H , we associate it with the greedy policy π f ( s, h ) = argmax a f h ( s, a ) , where s ∈ S and h ∈ [ H ] .
- H could be a class of models (i.e. each f ∈ H is an MDP itself). Here, for each f ∈ H , we can let π f be the optimal policy in the MDP f .

We let Π denote the induced set of policies from our hypothesis class H , i.e. Π = { π f | f ∈ H} .

The goal of agnostic learning can be formulated by the following optimization problem:

<!-- formula-not-decoded -->

where we are interested in the number of samples required to approximately solve this optimization problem. We will work both in the episodic setting and the generative model setting, and, for simplicity, we will restrict ourselves to finite hypothesis classes.

Binary classification as an H = 1 RL problem: Observe that the problem of binary classification can be viewed as learning in an MDP with a horizon of one. In particular, take H = 1 ; take |A| = 2 ; let the distribution over starting states s 0 ∼ µ correspond to the input distribution; and take the reward function as r ( s, a ) = 1 ( label ( s ) = a ) . In other words, we equate our action with the prediction of the binary class membership, and the reward function is determined by if our prediction is correct or not.

## 5.1.1 Review: Binary Classification

One of the most important concepts for learning binary classifiers is that it is possible to generalize even when the state space is infinite. Here note that the domain of our classifiers, often denoted by X , is analogous to the state space S . We now briefly review some basics of supervised learning before we turn to the question of generalization in reinforcement learning.

glyph[negationslash]

Consider the problem of binary classification with N labeled examples of the form ( x i , y i ) N i =1 , with x i ∈ X and y i ∈ { 0 , 1 } . Suppose we have a (finite or infinte) set H of binary classifiers where each h ∈ H is a mapping of the form h : X → { 0 , 1 } . Let 1 ( h ( x ) = y ) be an indicator which takes the value 0 if h ( x ) = y and 1 otherwise. We assume that our samples are drawn i.i.d. according to a fixed joint distribution D over ( x, y ) .

Define the empirical error and the true error as:

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

For a given h ∈ H , Hoeffding's inequality implies that with probability at least 1 -δ :

<!-- formula-not-decoded -->

This and the union bound give rise to what is often referred to as the 'Occam's razor' bound:

Proposition 5.1. (The 'Occam's razor' bound) Suppose H is finite. Let ̂ h = arg min h ∈H ̂ err ( h ) . With probability at least 1 -δ :

Hence, provided that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then with probability at least 1 -δ , we have that:

<!-- formula-not-decoded -->

A key observation here is that the our regret - the regret is the left hand side of the above inequality - has no dependence on the size of X (i.e. S ) which may be infinite and is only logarithmic in the number of hypothesis in our class.

## 5.1.2 Importance Sampling and a Reduction to Supervised Learning

Now let us return to the agnostic learning question in Equation 0.1. We will see that, provided we are willing to be exponential in the horizon H , then agnostic learning is possible. Furthermore, it is a straightforward argument to see that we are not able to do better.

## An Occam's Razor Bound for RL

We now provide a reduction of RL to the supervised learning problem, given only sampling access in the episodic setting. The key issue is how to efficiently reuse data. The idea is that we will simply collect N trajectories by executing a policy which chooses actions uniformly at random; let Unif A denote this policy.

The following shows how we can obtain an unbiased estimate of the value of any policy π using this uniform policy Unif A :

Lemma 5.2. (Unbiased estimation of V π 0 ( µ ) ) Let π be any deterministic policy. We have that:

<!-- formula-not-decoded -->

where Pr Unif A specifies the distribution over trajectories τ = ( s 0 , a 0 , r 0 , . . . s H -1 , a H -1 , r H -1 ) under the policy Unif A .

The proof follows from a standard importance sampling argument (applied to the distribution over the trajectories). Proof: We have that:

<!-- formula-not-decoded -->

where the last step follows due to that the probability ratio is only nonzero when Unif A choose actions identical to that of π .

Crudely, the factor of |A| H is due to that the estimated reward of π on a trajectory is nonzero only when π takes exactly identical actions to those taken by Unif A on the trajectory, which occurs with probability 1 / |A| H .

Now, given sampling access in the episodic model, we can use Unif A to get any estimate of any other policy π ∈ Π . Note that the factor of |A| H in the previous lemma will lead to this approach being a high variance estimator. Suppose we draw N trajectories under Unif A . Denote the n -th sampled trajectory by ( s n 0 , a n 0 , r n 1 , s n 1 , . . . , s n H -1 , a n H -1 , r n H -1 ) . We can then use following to estimate the finite horizon reward of any given policy π :

<!-- formula-not-decoded -->

Proposition 5.3. (An 'Occam's razor bound' for RL) Let δ ≥ 0 . Suppose Π is a finite and suppose we use the aforementioned estimator, ̂ V π 0 ( µ ) , to estimate the value of every π ∈ Π . Let ̂ π = arg max π ∈ Π ̂ V π 0 ( µ ) . We have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof: Note that, fixing π , our estimator is sum of i.i.d. random variables, where each independent estimator, |A| H 1 ( π ( s n 0 ) = a n 0 , . . . π ( s n H -1 ) = a n H -1 ) ∑ H -1 t =0 r ( s n t , a n t ) , is bounded by H |A| H . The remainder of the argument is identical to that used in Proposition 5.1.

Hence, provided that

<!-- formula-not-decoded -->

then with probability at least 1 -δ , we have that:

<!-- formula-not-decoded -->

This is the analogue of the Occam's razor bound for RL.

Importantly, the above shows that we can avoid dependence on the size of the state space, though this comes at the price of an exponential dependence on the horizon. As we see in the next section, this dependence is unavoidable (without making further assumptions).

Infinite Policy Classes. In the supervised learning setting, a crucial observation is that even though a hypothesis set H of binary classifiers may be infinite, we may still be able to obtain non-trivial generalization bounds. A crucial observation here is that even though the set H may be infinite, the number of possible behaviors of on a finite set of states is not necessarily exhaustive. The Vapnik-Chervonenkis (VC) dimension of H , V C ( H ) , is formal way to characterize this intuition, and, using this concept, we are able to obtain generalization bounds in terms of the V C ( H ) .

With regards to infinite hypothesis classes of policies (say for the case where |A| = 2 ), extending our Occam's razor bound can be done with precisely the same approach. In particular, when |A| = 2 , each π ∈ Π can be viewed as Boolean function, and this gives rise to the VC dimension VC (Π) of our policy class. The bound in Proposition 5.3 can be replaced with a VC-dimension bound that is analogous to that of binary classification (see Section 5.4).

## Lower Bounds

Clearly, the drawback of this agnostic learning approach is that we would require a number of samples that is exponential in the problem horizon. We now see that if we desire a sample complexity that scales with O (log | Π | ) , then an

exponential dependence on the effective horizon is unavoidable, without making further assumptions.

Here, our lower bound permits a (possibly randomized) algorithm to utilize a generative model (which is a more powerful sampling model than the episodic one).

Proposition 5.4. (Lower Bound with a Generative Model) Suppose algorithm A has access to a generative model. There exists a policy class Π , where | Π | = |A| H such that if algorithm A returns any policy π (not necessarily in Π ) such that

<!-- formula-not-decoded -->

with probability greater than 1 / 2 , then A must make a number of number calls N to the generative model where:

<!-- formula-not-decoded -->

(where c is an absolute constant).

Proof: The proof is simple. Consider a |A| -ary balanced tree, with |A| H states and |A| actions, where states correspond nodes and actions correspond to edges; actions always move the agent from the root towards a leaf node. We make only one leaf node rewarding, which is unknown to the algorithm. We consider the policy class to be all |A| H policies. The theorem now immediately follows since the algorithm gains no knowledge of the rewarding leaf node unless it queries that node.

Note this immediately rules out the possibility that any algorithm which can obtain a log | Π | dependence without paying a factor of |A| H in the sample size due to that log | Π | = H log |A| in the example above.

## 5.2 Linear Realizability

In supervised learning, two of the most widely studied settings are those of linear regression and binary classification with halfspaces. In both settings, we are able to obtain sample complexity results that are polynomial in the feature dimension. We now consider the analogue of these assumptions for RL, starting with the analogue of linear regression.

When the state space is large or infinite, we may hope that linearly realizability assumptions may permit a more sample efficient approach. We will start with linear realizability on Q π and consider the offline policy evaluation problem. Then we will consider the problem of learning with only a linearly realizability assumption on Q glyph[star] (along with access to either a generative model or sampling access in the episodic setting).

## 5.2.1 Offline Policy Evaluation with Linearly Realizable Values

In Chapter 3, we observed that the LSVI and LSPE algorithm could be used with an offline dataset for the purposes of policy evaluation. Here, we made the linear Bellman completeness assumption on our features. Let us now show that with only a linear realizability assumption, then not only is LSPE sample inefficient but we will also see that, information theoretically, every algorithm is sample inefficient, in a minmax sense.

This section is concerned with the offline RL setting. In this setting, the agent does not have direct access to the MDP and instead is given access to data distributions { µ h } H -1 h =0 where for each h ∈ [ H ] , µ h ∈ ∆( S h × A ) . The inputs of the agent are H datasets { D h } H -1 h =0 , and for each h ∈ [ H ] , D h consists i.i.d. samples of the form ( s, a, r, s ′ ) ∈ S h ×A× R ×S h +1 tuples, where ( s, a ) ∼ µ h , r ∼ r ( s, a ) , s ′ ∼ P ( s, a ) .

We now focus on the offline policy evaluation problem: given a policy π : S → ∆( A ) and a feature mapping φ : S × A → R d , the goal is to output an accurate estimate of the value of π (i.e., V π ) approximately, using the collected datasets { D h } H -1 h =0 , with as few samples as possible.

We will make the following linear realizability assumption with regards to a feature mapping φ : S ×A → R d , which we can think of as either being hand-crafted or coming from a pre-trained neural network that transforms a stateaction pair to a d -dimensional embedding, and the Q -functions can be predicted by linear functions of the features. In particular, this section will assume the following linear realizability assumption with regards to every policy π .

Assumption 5.5 (Realizable Linear Function Approximation) . For every policy π : S → ∆( A ) , there exists θ π 0 , . . . θ π H -1 ∈ R d such that for all ( s, a ) ∈ S × A and h ∈ [ H ] ,

<!-- formula-not-decoded -->

Note that this assumption is substantially stronger than assuming realizability with regards to a single target policy π (say the policy that we wish to evaluate); this assumption imposes realizability for all policies.

We will also assume a coverage assumption, analogous to Assumption 3.6. It should be evident that without feature coverage in our dataset, realizability alone is clearly not sufficient for sample-efficient estimation. Note that , we will make the strongest possible assumption, with regards to the conditioning of the feature covariance matrix; in particular, this assumption is equivalent to µ being a D -optimal design.

Assumption 5.6 (Coverage) . For all ( s, a ) ∈ S × A , assume our feature map is bounded such that ‖ φ ( s, a ) ‖ 2 ≤ 1 . Furthermore, suppose for each h ∈ [ H ] , the data distributions µ h satisfies the following:

<!-- formula-not-decoded -->

Note that the minimum eigenvalue of the above matrix is 1 /d , which is the largest possible minimum eigenvalue over all data distributions ˜ µ h , since σ min ( E ( s,a ) ∼ ˜ µ h [ φ ( s, a ) φ ( s, a ) glyph[latticetop] ]) is less than or equal to 1 /d (due to that ‖ φ ( s, a ) ‖ 2 ≤ 1 for all ( s, a ) ∈ S × A ). Also, it is not difficult to see that this distribution satisfies the D -optimal design property.

Clearly, for the case where H = 1 , the realizability assumption (Assumption 5.5), and coverage assumption (Assumption 5.6) imply that the ordinary least squares estimator will accurately estimate θ π 0 . The following shows these assumptions are not sufficient for offline policy evaluation for long horizon problems.

Theorem 5.7. Suppose Assumption 5.6 holds. Fix an algorithm that takes as input both a policy and a feature mapping. There exists a (deterministic) MDP satisfying Assumption 5.5, such that for any policy π : S → ∆( A ) , the algorithm requires Ω(( d/ 2) H ) samples to output the value of π up to constant additive approximation error with probability at least 0 . 9 .

Although we focus on offline policy evaluation, this hardness result also holds for finding near-optimal policies under Assumption 5.5 in the offline RL setting with linear function approximation. Below we give a simple reduction. At the initial state, if the agent chooses action a 1 , then the agent receives a fixed reward value (say 0 . 5 ) and terminates. If the agent chooses action a 2 , then the agent transits to the hard instance. Therefore, in order to find a policy with suboptimality at most 0 . 5 , the agent must evaluate the value of the optimal policy in the hard instance up to an error of 0 . 5 , and hence the hardness result holds.

Least-Squares Policy Evaluation (LSPE) has exponential variance. For offline policy evaluation with linear function approximation, it is not difficult to see that LSPE, Algorithm 2, will provide an unbiased estimate (provided the feature covariance matrices are full rank, which will occur with high probability). Interestingly, as a direct corollary, the above theorem implies that LSPE has exponential variance in H . More generally, this theorem implies that there is no estimator that can avoid such exponential dependence in the offline setting.

Figure 0.1: An illustration of the hard instance. Recall that ˆ d = d/ 2 . States on the top are those in the first level ( h = 0 ), while states at the bottom are those in the last level ( h = H -1) . Solid line (with arrow) corresponds to transitions associated with action a 1 , while dotted line (with arrow) corresponds to transitions associated with action a 2 . For each level h ∈ [ H ] , reward values and Q -values associated with s 1 h , s 2 h , . . . , s ˆ d h are marked on the left, while reward values and Q -values associated with s ˆ d +1 h are mark on the right. Rewards and transitions are all deterministic, except for the reward distributions associated with s 1 H -1 , s 2 H -1 , . . . , s ˆ d H -1 . We mark the expectation of the reward value when it is stochastic. For each level h ∈ [ H ] , for the data distribution µ h , the state is chosen uniformly at random from those states in the dashed rectangle, i.e., { s 1 h , s 2 h , . . . , s ˆ d h } , while the action is chosen uniformly at random from { a 1 , a 2 } . Suppose the initial state is s ˆ d +1 1 . When r ∞ = 0 , the value of the policy is 0 . When r ∞ = ˆ d -H/ 2 , the value of the policy is r ∞ · ˆ d H/ 2 = 1 .

<!-- image -->

## A Hard Instance for Offline Policy Evaluation

Wenowprovide the hard instance construction and the proof of Theorem 5.7. We use d to denote the feature dimension, and we assume d is even for simplicity. We use ˆ d to denote d/ 2 for convenience. We also provide an illustration of the construction in Figure 0.1.

State Space, Action Space and Transition Operator. The action space A = { a 1 , a 2 } . For each h ∈ [ H ] , S h contains ˆ d +1 states s 1 h , s 2 h , . . . , s ˆ d h and s ˆ d +1 h . For each h ∈ { 0 , 1 , . . . , H -2 } , for each c ∈ { 1 , 2 , . . . , ˆ d +1 } , we have

<!-- formula-not-decoded -->

Reward Distributions. Let 0 ≤ r ∞ ≤ ˆ d -H/ 2 be a parameter to be determined. For each ( h, c ) ∈ { 0 , 1 , . . . , H -2 } × [ ˆ d ] and a ∈ A , we set r ( s c h , a ) = 0 and r ( s ˆ d +1 h , a ) = r ∞ · ( ˆ d 1 / 2 -1) · ˆ d ( H -h -1) / 2 . For the last level, for each c ∈ [ ˆ d ] and a ∈ A , we set

<!-- formula-not-decoded -->

so that E [ r ( s c H -1 , a )] = r ∞ . Moreover, for all actions a ∈ A , r ( s ˆ d +1 H -1 , a ) = r ∞ · ˆ d 1 / 2 .

Feature Mapping. Let e 1 , e 2 , . . . , e d be a set of orthonormal vectors in R d . Here, one possible choice is to set e 1 , e 2 , . . . , e d to be the standard basis vectors. For each ( h, c ) ∈ [ H ] × [ ˆ d ] , we set φ ( s c h , a 1 ) = e c , φ ( s c h , a 2 ) = e c + ˆ d , and for all a ∈ A .

Verifying Assumption 5.5. Now we verify that Assumption 5.5 holds for this construction.

Lemma 5.8. For every policy π : S → ∆( A ) , for each h ∈ [ H ] , for all ( s, a ) ∈ S h × A , we have Q π h ( s, a ) = ( θ π h ) glyph[latticetop] φ ( s, a ) for some θ π h ∈ R d .

Proof: We first verify Q π is linear for the first H -1 levels. For each ( h, c ) ∈ { 0 , 1 , . . . , H -2 } × [ ˆ d ] , we have

<!-- formula-not-decoded -->

Moreover, for all a ∈ A ,

<!-- formula-not-decoded -->

Therefore, if we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then Q π h ( s, a ) = ( θ π h ) glyph[latticetop] φ ( s, a ) for all ( s, a ) ∈ S h ×A .

Now we verify that the Q -function is linear for the last level. Clearly, for all c ∈ [ ˆ d ] and a ∈ A , Q π H -1 ( s c H -1 , a ) = r ∞ and Q π H -1 ( s ˆ d +1 H -1 , a ) = r ∞ · √ ˆ d . Thus by defining θ π H -1 = ∑ d c =1 r ∞ · e c , we have Q π H -1 ( s, a ) = ( θ π H -1 ) glyph[latticetop] φ ( s, a ) for all ( s, a ) ∈ S H -1 ×A .

The Data Distributions. For each level h ∈ [ H ] , the data distribution µ h is a uniform distribution over the set { ( s 1 h , a 1 ) , ( s 1 h , a 2 ) , ( s 2 h , a 1 ) , ( s 2 h , a 2 ) , . . . , ( s ˆ d h , a 1 ) , ( s ˆ d h , a 2 ) } . Notice that ( s ˆ d +1 h , a ) is not in the support of µ h for all a ∈ A . It can be seen that,

<!-- formula-not-decoded -->

## The Information-Theoretic Argument

We show that it is information-theoretically hard for any algorithm to distinguish the case r ∞ = 0 and r ∞ = ˆ d -H/ 2 . We fix the initial state to be s ˆ d +1 0 , and consider any policy π : S → ∆( A ) . When r ∞ = 0 , all reward values will be zero, and thus the value of π would be zero. On the other hand, when r ∞ = ˆ d -H/ 2 , the value of π would be r ∞ · ˆ d H/ 2 = 1 . Thus, if the algorithm approximates the value of the policy up to an error of 1 / 2 , then it must distinguish the case that r ∞ = 0 and r ∞ = ˆ d -H/ 2 .

We first notice that for the case r ∞ = 0 and r ∞ = ˆ d -H/ 2 , the data distributions { µ h } H -1 h =0 , the feature mapping φ : S ×A → R d , the policy π to be evaluated and the transition operator P are the same. Thus, in order to distinguish the case r ∞ = 0 and r ∞ = ˆ d -H/ 2 , the only way is to query the reward distribution by using sampling taken from the data distributions.

For all state-action pairs ( s, a ) in the support of the data distributions of the first H -1 levels, the reward distributions will be identical. This is because for all s ∈ S h \ { s ˆ d +1 h } and a ∈ A , we have r ( s, a ) = 0 . For the case r ∞ = 0 and r ∞ = ˆ d -H/ 2 , for all state-action pairs ( s, a ) in the support of the data distribution of the last level,

<!-- formula-not-decoded -->

Therefore, to distinguish the case that r ∞ = 0 and r ∞ = ˆ d -H/ 2 , the agent needs to distinguish two reward distributions and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is standard argument that in order to distinguish r (1) and r (2) with probability at least 0 . 9 , any algorithm requires Ω( ˆ d H ) samples.

The key in this construction is the state s ˆ d +1 h in each level, whose feature vector is defined to be ∑ c ∈ ˆ d e c / ˆ d 1 / 2 . In each level, s ˆ d +1 h amplifies the Q -values by a ˆ d 1 / 2 factor, due to the linearity of the Q -function. After all the H levels, the value will be amplified by a ˆ d H/ 2 factor. Since s ˆ d +1 h is not in the support of the data distribution, the only way for the agent to estimate the value of the policy is to estimate the expected reward value in the last level. This construction forces the estimation error of the last level to be amplified exponentially and thus implies an exponential lower bound.

## 5.2.2 Linearly Realizable Q glyph[star]

Specifically, we will assume access to a feature map φ : S × A → R d , and we will assume that a linear function of φ can accurately represent the Q glyph[star] -function. Specifically,

Assumption 5.9 (Linear Q glyph[star] Realizability) . For all h ∈ [ H ] , assume there exists θ ∗ h ∈ R d such that for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

The hope is that this assumption may permit a sample complexity that is polynomial in d and H , with no explicit |S| or |A| dependence.

Another assumption that we will use is that the minimum suboptimality gap is lower bounded.

<!-- image -->

-

Figure 0.2: The Leaking Complete Graph Construction. Illustration of a hard MDP. There are m +1 states in the MDP, where f is an absorbing terminal state. Starting from any non-terminal state a , regardless of the action, there is at least α = 1 / 6 probability that the next state will be f .

Assumption 5.10 (Constant Sub-optimality Gap) . For any state s ∈ S , a ∈ A , the suboptimality gap is defined as ∆ h ( s, a ) := V ∗ h ( s ) -Q ∗ h ( s, a ) . We assume that

<!-- formula-not-decoded -->

1 The hope is that with a 'large gap', the identification of the optimal policy itself (as opposed to just estimating its value accurately) may be statistically easier, thus making the problem easier.

We now present two hardness results. We start with the case where we have access to a generative model.

Theorem 5.11. (Linear Q glyph[star] ; Generative Model Case) Consider any algorithm A which has access to a generative model and which takes as input the feature mapping φ : S ×A → R d . There exists an MDP with a feature mapping φ satisfying Assumption 5.9 and where the size of the action space is |A| = c 1 glyph[ceilingleft] min { d 1 / 4 , H 1 / 2 }glyph[ceilingright] such that if A (when given φ as input) finds a policy π such that

<!-- formula-not-decoded -->

with probability 0 . 1 , then A requires min { 2 c 2 d , 2 c 2 H } samples ( c 1 and c 2 are absolute constants).

Note that theorem above uses an MDP whose size of the action space is only of moderate size (actually sublinear in both d and H ). Of course, in order to prove such a result, we must rely on a state space which is exponential in d or H (else, we could apply a tabular algorithm to obtain a polynomial result). The implications of the above show that the linear Q glyph[star] assumption, alone, is not sufficient for sample efficient RL, even with access to a generative model.

We now see that in the (online) episodic setting, even if we have a constant gap in the MDP, the hardness of learning is still unavoidable.

Theorem 5.12. (Linear Q glyph[star] Realizability + Gap; Episodic Setting) Consider any algorithm A which has access to the episodic sampling model and which takes as input the feature mapping φ : S × A → R d . There exists an MDP with a feature mapping φ satisfying Assumption 5.9 and Assumption 5.10 (where ∆ min is an absolute constant) such that if A (using φ as input) finds a policy π such that

<!-- formula-not-decoded -->

with probability 0 . 1 , then A requires min { 2 cd , 2 cH } samples (where c is an absolute constant).

The theorem above also holds with an MDP whose action space is of size |A| = Ω( glyph[ceilingleft] min { d 1 / 4 , H 1 / 2 }glyph[ceilingright] ) . However, for ease of exposition, we will present a proof of this theorem that uses a number of actions that is exponential in d (see Section 5.4).

Linear Q glyph[star] Realizability + Gap; Generative Model Case. For this case, it is not difficult to see that, with a generative model, it is possible to exactly learn an optimal policy with a number of samples that is poly ( d, H, 1 / ∆ min ) (see Section 5.4). The key idea is that, at the last stage, the large gap permits us to learn the optimal policy itself (using linear regression). Furthermore, if we have an optimal policy from timestep h onwards, then we are able to obtain unbiased estimates of Q glyph[star] h -1 ( s, a ) , at any state action pair, using the generative model. We leave the proof as an exercise to the interested reader.

## A Hard Instance with Constant Suboptimality Gap

We now prove Theorem 5.12, and we do not prove Theorem 5.11 (see Section 5.4). The remainder of this section provides the construction of a hard family of MDPs where Q ∗ is linearly realizable and has constant suboptimality gap and where it takes exponential samples to learn a near-optimal policy. Each of these hard MDPs can roughly be seen as a 'leaking complete graph' (see Fig. 0.2). Information about the optimal policy can only be gained by: (1) taking the optimal action; (2) reaching a non-terminal state at level H . We will show that when there are exponentially many actions, either events happen with negligible probability unless exponentially many trajectories are played.

Let m be an integer to be determined. The state space is { ¯ 1 , · · · , ¯ m,f } . The special state f is called the terminal state . The action space is simply A = { 1 , 2 , . . . , m } . Each MDP in this family is specified by an index a ∗ ∈ { 1 , 2 , . . . , m } and denoted by M a ∗ . In other words, there are m MDPs in this family. We will make use of a (large) set of approximately orthogonal vectors, which exists by the Johnson-Lindenstrauss lemma. We state the following lemma without proof:

Lemma 5.13 (Johnson-Lindenstrauss) . For any α &gt; 0 , if m ≤ exp( 1 8 α 2 d ′ ) , there exists m unit vectors { v 1 , · · · , v m } in R d ′ such that ∀ i, j ∈ { 1 , 2 , . . . , m } such that i = j , |〈 v i , v j 〉| ≤ α .

glyph[negationslash]

We will set α = 1 6 and m = glyph[floorleft] exp( 1 8 α 2 d ) glyph[floorright] . By Lemma 5.13, we can find such a set of d -dimensional unit vectors { v 1 , · · · , v m } . For the clarity of presentation, we will use v i and v ( i ) interchangeably. The construction of M a ∗ is specified below. Note that in this construction, the features, the rewards and the transitions are defined for all a 1 , a 2 with a 1 , a 2 ∈ { 1 , 2 , . . . , m } and a 1 = a 2 . In particular, this construction is properly defined even when a 1 = a ∗ .

glyph[negationslash]

Features. The feature map, which maps state-action pairs to d +1 dimensional vectors, is defined as follows.

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Here 0 is the zero vector in R d . Note that the feature map is independent of a ∗ and is shared across the MDP family.

Rewards. For h ∈ [ H ] , the rewards are defined as

<!-- formula-not-decoded -->

For h = H -1 , r H -1 ( s, a ) := 〈 φ ( s, a ) , (1 , v ( a ∗ )) 〉 for every state-action pair.

Transitions. The initial state distribution µ is set as a uniform distribution over { ¯ 1 , · · · , ¯ m } . The transition probabilities are set as follows.

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

After taking action a 2 , the next state is either a 2 or f . Thus this MDP looks roughly like a 'leaking complete graph' (see Fig. 0.2): starting from state a , it is possible to visit any other state (except for a ∗ ); however, there is always at least 1 -3 α probability of going to the terminal state f . The transition probabilities are indeed valid, because

<!-- formula-not-decoded -->

We now verify that realizability, i.e. Assumption 5.9, is satisfied.

Lemma 5.14. (Linear realizability) In the MDP M a ∗ , ∀ h ∈ [ H ] , for any state-action pair ( s, a ) , Q ∗ h ( s, a ) = 〈 φ ( s, a ) , θ ∗ 〉 with θ ∗ = (1 , v ( a ∗ )) .

Proof: We first verify the statement for the terminal state f . Observe that at the terminal state f , the next state is always f and the reward is either 0 (if action 1 is chosen) or -1 (if an action other than 1 is chosen). Hence, we have

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

and

This implies Q ∗ h ( f, · ) = 〈 φ ( f, · ) , (1 , v ( a ∗ )) 〉 .

We now verify realizability for other states via induction on h = H -1 , · · · , 0 . The induction hypothesis is that for all a 1 , a 2 ∈ { 1 , 2 , . . . , m } , we have glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

and glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

Note that (0.2) implies that realizability is satisfied. In the remaining part of the proof we verify Eq. (0.2) and (0.3).

When h = H -1 , (0.2) holds by the definition of rewards. Next, note that for all h ∈ [ H ] , (0.3) follows from (0.2). This is because for all a 1 = a ∗ , for all a 2 / ∈ { a 1 , a ∗ } .

<!-- formula-not-decoded -->

Moreover, for all a 1 = a ∗ , glyph[negationslash]

<!-- formula-not-decoded -->

Furthermore, for all a 1 = a ∗ , glyph[negationslash]

<!-- formula-not-decoded -->

In other words, (0.2) implies that a ∗ is always the optimal action for all state a 1 with a 1 = a ∗ . Now, for state a ∗ , for all a = a ∗ , we have glyph[negationslash]

<!-- formula-not-decoded -->

Hence, (0.2) implies that a ∗ is always the optimal action for all states a with a ∈ { 1 , 2 , . . . , m } .

glyph[negationslash]

Thus, it remains to show that (0.2) holds for h assuming (0.3) holds for h + 1 . Here we only consider the case that a 2 = a 1 and a 2 = a ∗ , since otherwise Pr[ f | a 1 , a 2 ] = 1 and thus (0.2) holds by the definition of the rewards and the fact that V ∗ h ( f ) = 0 . When a 2 / ∈ { a 1 , a ∗ } , we have glyph[negationslash]

<!-- formula-not-decoded -->

This is exactly (0.2) for h . Hence both (0.2) and (0.3) hold for all h ∈ [ H ] .

We now verify the constant sub-optimality gap, i.e. Assumption 5.10, is satisfied.

Lemma 5.15. (Constant Gap) Assumption 5.10 is satisfied with ∆ min = 1 / 24 .

Proof: From Eq. (0.2) and (0.3), it is easy to see that at state a 1 = a ∗ , for a 2 = a ∗ , the suboptimality gap is glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

Moreover, at state a ∗ , for a = a ∗ , the suboptimality gap is

<!-- formula-not-decoded -->

Finally, for the terminal state f , the suboptimality gap is obviously 1 . Therefore ∆ min ≥ 1 24 for all MDPs under consideration.

glyph[negationslash]

glyph[negationslash]

## The Information-Theoretic Argument

We provide a proof sketch for the lower bound below in Theorem 5.12.

Proof sketch. We will show that for any algorithm, there exists a ∗ ∈ { 1 , 2 , . . . , m } such that in order to output π with

<!-- formula-not-decoded -->

with probability at least 0 . 1 for M a ∗ , the number of samples required is 2 Ω(min { d,H } ) .

Observe that the feature map of M a ∗ does not depend on a ∗ , and that for h &lt; H -1 and a 2 = a ∗ , the reward R h ( a 1 , a 2 ) also contains no information about a ∗ . The transition probabilities are also independent of a ∗ , unless the action a ∗ is taken. Moreover, the reward at state f is always 0 . Thus, to receive information about a ∗ , the agent either needs to take the action a ∗ , or be at a non-game-over state at the final time step ( h = H -1 ).

glyph[negationslash]

However, note that the probability of remaining at a non-terminal state at the next layer is at most glyph[negationslash]

<!-- formula-not-decoded -->

Thus for any algorithm, Pr[ s H -1 = f ] ≤ ( 3 4 ) H , which is exponentially small.

glyph[negationslash]

In other words, any algorithm that does not know a ∗ either needs to 'be lucky' so that s H -1 = f , or needs to take a ∗ 'by accident'. Since the number of actions is m = 2 Θ( d ) , either event cannot happen with constant probability unless the number of episodes is exponential in min { d, H } .

In order to make this claim rigorous, we can construct a reference MDP M 0 as follows. The state space, action space, and features of M 0 are the same as those of M a . The transitions are defined as follows:

glyph[negationslash]

<!-- formula-not-decoded -->

The rewards are defined as follows:

<!-- formula-not-decoded -->

glyph[negationslash]

Note that M 0 is identical to M a ∗ , except when a ∗ is taken, or when an trajectory ends at a non-terminal state. Since the latter event happens with an exponentially small probability, we can show that for any algorithm the probability of taking a ∗ in M a ∗ is close to the probability of taking a ∗ in M 0 . Since M 0 is independent of a ∗ , unless an exponential number of samples are used, for any algorithm there exists a ∗ ∈ { 1 , 2 , . . . , m } such that the probability of taking a ∗ in M 0 is o (1) . It then follows that the probability of taking a ∗ in M a ∗ is o (1) . Since a ∗ is the optimal action for every state, such an algorithm cannot output a near-optimal policy for M a ∗ .

## 5.2.3 Linearly Realizable π glyph[star]

Similar to value-based learning, a natural assumption for policy-based learning is that the optimal policy is realizable by a halfspace.

Assumption 5.16 (Linear π ∗ Realizability) . For any h ∈ [ H ] , there exists θ glyph[star] h ∈ R d that satisfies for any s ∈ S , we have

<!-- formula-not-decoded -->

As before (and as is natural in supervised learning), we will also consider the case where there is a large margin (which is analogous to previous large gap assumption).

Assumption 5.17 (Constant Margin) . Without loss of generality, assume the scalings are such that for all ( s, a, h ) ∈ S × A × [ H ] , ‖ φ ( s, a ) ‖ 2 ≤ 1 and ‖ θ glyph[star] h ‖ ≤ 1 Now suppose that for all s ∈ S and any a / ∈ π ∗ ( s ) ,

<!-- formula-not-decoded -->

Here we restrict the linear coefficients and features to have unit norm for normalization.

We state the following without proof (see Section 5.4):

Theorem 5.18. (Linear π glyph[star] Realizability + Margin) Consider any algorithm A which has access to a generative model and which takes as input the feature mapping φ : S ×A → R d . There exists an MDP with a feature mapping φ satisfying Assumption 5.16 and Assumption 5.17 (where ∆ min is an absolute constant) such that if A (with φ as input) finds a policy π such that

<!-- formula-not-decoded -->

with probability 0 . 1 , then A requires min { 2 cd , 2 cH } samples, where c is an absolute constant.

## 5.3 Discussion: Studying Generalization in RL

The previous lower bounds shows that: (i) without further assumptions, agnostic learning (in the standard supervised learning sense) is not possible in RL, unless we can tolerate an exponential dependence on the horizon, and (ii) simple realizability assumptions are also not sufficient for sample efficient RL.

This motivates the study of RL to settings where we either make stronger assumptions or have a means in which the agent can obtain side information. Three examples of these approaches that are considered in this book are:

- Structural (and Modelling) Assumptions: By making stronger assumptions about the how the hypothesis class relates to the underlying MDP, we can move away from agnostic learning lower bound. We have seen one example of this with the stronger Bellman completeness assumption that we considered in Chapter 3. We will see more examples of this in Part 2.
- Distribution Dependent Results (and Distribution Shift): We have seen one example of this approach when we consider the approximate dynamic programming approach in Chapter 4. When we move to policy gradient methods (in Part 3), we will also consider results which depend on the given distribution (i.e. where we suppose our starting state distribution µ has some reasonable notion of coverage).
- Imitation Learning and Behavior Cloning: here we will consider settings where the agent has input from, effectively, a teacher, and we see how this can circumvent statistical hardness results.

## 5.4 Bibliographic Remarks and Further Readings

The reduction from reinforcement learning to supervised learning was first introduced in [Kearns et al., 2000], which used a different algorithm (the 'trajectory tree' algorithm), as opposed to the importance sampling approach presented

here. [Kearns et al., 2000] made the connection to the VC dimension of the policy class. The fundamental sample complexity tradeoff - between polynomial dependence on the size of the state space and exponential dependence on the horizon - was discussed in depth in [Kakade, 2003].

Utilizing linear methods for dynamic programming goes back to, at least, [Shannon, 1959, Bellman and Dreyfus, 1959]. Formally considering the linear Q glyph[star] assumption goes back to at least [Wen and Van Roy, 2017]. Resolving the learnability under this assumption was an important open question discussed in [Du et al., 2019], which is now resolved. In the offline (policy evaluation) setting, the lower bound in Theorem 5.7 is due to [Wang et al., 2020a, Zanette, 2021]. With a generative model, the breakthrough result of [Weisz et al., 2021a] established the impossibility result with the linear Q glyph[star] assumption. Furthermore, Theorem 5.12, due to [Wang et al., 2021], resolves this question in the online setting with the additional assumption of having a constant sub-optimality gap. Also, [Weisz et al., 2021b] extends the ideas from [Weisz et al., 2021a], so that the lower bound is applicable with action spaces of polynomial size (in d and H ); this is the result we use for Theorem 5.11.

Theorem 5.18, which assumed a linearly realizable optimal policy, is due to [Du et al., 2019].

## Part 2

## Strategic Exploration

## Chapter 6

## Multi-Armed &amp; Linear Bandits

For the case, where γ = 0 (or H = 1 in the undiscounted case), the problem of learning in an unknown MDP reduce to the multi-armed bandit problem. The basic algorithms and proof methodologies here are important to understand in their own right, due to that we will have to extend these with more sophisticated variants to handle the explorationexploitation tradeoff in the more challenging reinforcement learning problem.

This chapter follows analysis of the LinUCB algorithm from the original proof in [Dani et al., 2008], with a simplified concentration analysis due to [Abbasi-Yadkori et al., 2011].

Throughout this chapter, we assume reward is stochastic.

## 6.1 The K -Armed Bandit Problem

The setting is where we have K decisions (the 'arms'), where when we play arm a ∈ { 1 , 2 , . . . K } we obtain a random reward r a ∈ [ -1 , 1] from R ( a ) ∈ ∆([ -1 , 1]) which has mean reward:

<!-- formula-not-decoded -->

where it is easy to see that µ a ∈ [ -1 , 1] .

Every iteration t , the learner will pick an arm I t ∈ [1 , 2 , . . . K ] . Our cumulative regret is defined as:

<!-- formula-not-decoded -->

We denote a glyph[star] = argmax i µ i as the optimal arm. We define gap ∆ a = µ a glyph[star] -µ a for any arm a .

Theorem 6.1. There exists an algorithm such that with probability at least 1 -δ , we have:

glyph[negationslash]

<!-- formula-not-decoded -->

## 6.1.1 The Upper Confidence Bound (UCB) Algorithm

We summarize the upper confidence bound (UCB) algorithm in Alg. 3. For simplicity, we allocate the first K rounds to pull each arm once.

| Algorithm 3   | UCB                                                                              |
|---------------|----------------------------------------------------------------------------------|
| 1:            | Play each arm once and denote received reward as r a for all a ∈ { 1 , 2 ,...K } |
| 2:            | for t = 1 → T - K do )                                                           |
| 3:            | Execute arm I t = argmax i ∈ [ K ] ( ˆ µ t i + √ log( TK/δ ) N t i               |
| 4:            | Observe r t := r I t                                                             |
| 5:            | end for                                                                          |

where every iteration t , we maintain counts of each arm:

<!-- formula-not-decoded -->

where I t is the index of the arm that is picked by the algorithm at iteration t . We main the empirical mean for each arm as follows:

<!-- formula-not-decoded -->

Recall that r a is the reward of arm a we got during the first K rounds.

We also main the upper confidence bound for each arm as follows:

<!-- formula-not-decoded -->

The following lemma shows that this is a valid upper confidence bound with high probability.

Lemma 6.2 (Upper Confidence Bound) . For all t ∈ [1 , . . . , T ] and a ∈ [1 , 2 , . . . K ] , we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The proof of the above lemma uses Azuma-Hoeffding's inequality (Theorem A.5) for each arm a and iteration t and then apply a union bound over all T iterations and K arms. The reason one needs to use Azuma-Hoeffding's inequality is that here the number of trials of arm a , i.e. N t a , itself is a random variable while the classic Hoeffding's inequality only applies to the setting where the number of trials is a fixed number.

Proof: We first consider a fixed arm a . Let us define the following random variables, X 0 = r a -µ a , X 1 = 1 { I 1 = a } ( r 1 -µ a ) , X 2 = 1 { I 2 = a } ( r 2 -µ a ) , . . . , X i = 1 { I i = a } ( r i -µ a ) , . . . .

Regarding the boundedness on these random variables, we have that for i where 1 { I i = a } = 1 , we have | X i | ≤ 1 , and for i where 1 { I i = a } = 0 , we have | X i | = 0 . Now, consider E [ X i |H &lt;i ] , where H &lt;i is all history up to but not including iteration i . Note that we have E [ X i |H &lt;i ] = 0 since condition on history H &lt;i , 1 { I i = a } is a deterministic quantity (i.e., UCB algorithm determines whether or not to pull arm a at iteration i only based on the information from H &lt;i ). Thus, we can conclude that { X i } i =0 is a Martingale difference sequence. Via Azuma-Hoeffding's inequality, we have that with probability at least 1 -δ , for any fixed t , we have:

<!-- formula-not-decoded -->

where we use the fact that ∑ t -1 i =0 | X i | 2 ≤ N t a . Divide N t a on both side, we have:

<!-- formula-not-decoded -->

Apply union bound over all 0 ≤ t ≤ T and a ∈ { 1 , . . . , K } , we have that with probability 1 -δ , for all t, a :

<!-- formula-not-decoded -->

This concludes the proof.

Now we can conclude the proof of the main theorem.

Proof: Below we conditioned on the above inequality 0.1 holds. This gives us the following optimism:

<!-- formula-not-decoded -->

Thus, we can upper bound the regret as follows:

<!-- formula-not-decoded -->

Sum over all iterations, we get:

<!-- formula-not-decoded -->

Note that our algorithm has regret K at the first K rounds.

On the other hand, if for each arm a , the gap ∆ a &gt; 0 , then, we must have:

<!-- formula-not-decoded -->

which is because after the UCB of an arm a is below µ a glyph[star] , UCB algorithm will never pull this arm a again (the UCB of a glyph[star] is no smaller than µ a glyph[star] ).

Thus for the regret calculation, we get:

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

Together with the fact that Inequality 0.1 holds with probability at least 1 -δ , we conclude the proof.

## 6.2 Linear Bandits: Handling Large Action Spaces

Let D ⊂ R d be a compact (but otherwise arbitrary) set of decisions. On each round, we must choose a decision x t ∈ D . Each such choice results in a reward r t ∈ [ -1 , 1] .

## Algorithm 4 The Linear UCB algorithm

Input: λ , β

$$t 1: for t = 0 , 1 . . . do 2: Execute x t = argmax x ∈ D max µ ∈ BALL t µ · x and observe the reward r t . 3: Update BALL t +1 (as specified in Equation 0.2). 4: end for$$

We assume that, regardless of the history H of decisions and observed rewards, the conditional expectation of r t is a fixed linear function, i.e. for all x ∈ D ,

<!-- formula-not-decoded -->

where x ∈ D is arbitrary. Here, observe that we have assumed the mean reward for any decision is bounded in [ -1 , 1] . Under these assumptions, the noise sequence ,

<!-- formula-not-decoded -->

is a martingale difference sequence.

The is problem is essentially a bandit version of a fundamental geometric optimization problem, in which the agent's feedback on each round t is only the observed reward r t and where the agent does not know µ glyph[star] apriori.

If x 0 , . . . x T -1 are the decisions made in the game, then define the cumulative regret by

<!-- formula-not-decoded -->

where x glyph[star] ∈ D is an optimal decision for µ glyph[star] , i.e.

which exists since D is compact. Observe that if the mean µ glyph[star] were known, then the optimal strategy would be to play x glyph[star] every round. Since the expected loss for each decision x equals µ glyph[star] · x , the cumulative regret is just the difference between the expected loss of the optimal algorithm and the expected loss for the actual decisions x t . By the HoeffdingAzuma inequality (see Lemma A.5), the observed reward ∑ T -1 t =0 r t will be close to their (conditional) expectations ∑ T -1 t =0 µ glyph[star] · x t .

<!-- formula-not-decoded -->

Since the sequence of decisions x 1 , . . . , x T -1 may depend on the particular sequence of random noise encountered, R T is a random variable. Our goal in designing an algorithm is to keep R T as small as possible.

## 6.2.1 The LinUCB algorithm

LinUCB is based on 'optimism in the face of uncertainty,' which is described in Algorithm 4. At episode t , we use all previous experience to define an uncertainty region (an ellipse) BALL t . The center of this region, ̂ µ t , is the solution of the following regularized least squares problem:

<!-- formula-not-decoded -->

where λ is a parameter and where

<!-- formula-not-decoded -->

The shape of the region BALL t is defined through the feature covariance Σ t .

Precisely, the uncertainty region, or confidence ball, is defined as:

<!-- formula-not-decoded -->

where β t is a parameter of the algorithm.

Computation. Suppose that we have an efficient linear optimization oracle, i.e. that we can efficiently solve the problem:

<!-- formula-not-decoded -->

for any ν . Even with this, Step 2 of LinUCB may not be computationally tractable. For example, suppose that D is provided to us as a polytope, then the above oracle can be efficiently computed using linear programming, while LinUCB is an NP-hard optimization. Here, we can actually use a wider confidence region, where we can keep track of glyph[lscript] 1 ball which contains BALL t . See Section 6.4 for further reading.

## 6.2.2 Upper and Lower Bounds

Our main result here is that we have sublinear regret with only a polynomial dependence on the dimension d and, importantly, no dependence on the cardinality of the decision space D , i.e. on | D | .

Theorem 6.3. Suppose that the expected costs are bounded, in magnitude, by 1 , i.e. that | µ glyph[star] · x | ≤ 1 for all x ∈ D ; that ‖ µ glyph[star] ‖ ≤ W and ‖ x ‖ ≤ B for all x ∈ D ; and that the noise η t is σ 2 sub-Gaussian 1 . Set

<!-- formula-not-decoded -->

We have that with probability greater than 1 -δ , that (simultaneously) for all T ≥ 0 ,

<!-- formula-not-decoded -->

where c is an absolute constant. In other words, we have that R T is ˜ O ( d √ T ) with high probability.

The following shows that no algorithm can (uniformly) do better.

Theorem 6.4. (Lower bound) There exists a distribution over linear bandit problems (i.e. a distribution over µ ) with the rewards being bounded by 1 , in magnitude, and σ 2 ≤ 1 , such that for every (randomized) algorithm, we have for n ≥ max { 256 , d 2 / 16 } ,

<!-- formula-not-decoded -->

where the inner expectation is with respect to randomness in the problem and the algorithm.

We will only prove the upper bound (See Section bib:bandits).

1 Roughly, this assumes the tail probabilities of η t decay no more slowly than a Gaussian distribution. See Definition A.2.

LinUCB and D -optimal design. Let us utilize the D -optimal design to improve the dependencies in Theorem 6.3. Let Σ D denote the D -optimal design matrix from Theorem 3.2. Consider the coordinate transformation:

<!-- formula-not-decoded -->

Observe that ˜ µ glyph[star] · ˜ x = µ glyph[star] · x , so we still have a linear expected reward function in this new coordinate systems. Also, observe that ‖ ˜ x ‖ = ‖ x ‖ Σ D ≤ d , which is a property of the D -optimal design, and that

<!-- formula-not-decoded -->

where the last step uses our assumption that the rewards are bounded, in magnitude, by 1 .

The following corollary shows that, without loss of generality, we we can remove the dependencies on B and W from the previous theorem, due to that B ≤ √ d and W ≤ 1 when working under this coordinate transform.

Corollary 6.5. Suppose that the expected costs are bounded, in magnitude, by 1 , i.e. that | ˜ µ glyph[star] · ˜ x | ≤ 1 for all x ∈ D and that the noise η t is σ 2 sub-Gaussian. Suppose linUCB is implemented in the ˜ x coordinate system, as described above, with the following settings of the parameters.

<!-- formula-not-decoded -->

We have that with probability greater than 1 -δ , that (simultaneously) for all T ≥ 0 ,

<!-- formula-not-decoded -->

where c is an absolute constant.

## 6.3 LinUCB Analysis

In establishing the upper bounds there are two main propositions from which the upper bounds follow. The first is in showing that the confidence region is appropriate.

Proposition 6.6. (Confidence) Let δ &gt; 0 . We have that

<!-- formula-not-decoded -->

Section 6.3.2 is devoted to establishing this confidence bound. In essence, the proof seeks to understand the growth of the quantity ( ̂ µ t -µ glyph[star] ) glyph[latticetop] Σ t ( ̂ µ t -µ glyph[star] ) .

The second main step in analyzing LinUCB is to show that as long as the aforementioned high-probability event holds, we have some control on the growth of the regret. Let us define

<!-- formula-not-decoded -->

which denotes the instantaneous regret.

The following bounds the sum of the squares of instantaneous regret.

Proposition 6.7. (Sum of Squares Regret Bound) Suppose that ‖ x ‖ ≤ B for x ∈ D . Suppose β t is increasing and that β t ≥ 1 . For LinUCB, if µ glyph[star] ∈ BALL t for all t , then

<!-- formula-not-decoded -->

This is proven in Section 6.3.1. The idea of the proof involves a potential function argument on the log volume (i.e. the log determinant) of the 'precision matrix' Σ t (which tracks how accurate our estimates of µ glyph[star] are in each direction). The proof involves relating the growth of this volume to the regret.

Using these two results we are able to prove our upper bound as follows:

Proof: [Proof of Theorem 6.3] By Propositions 6.6 and 6.7 along with the Cauchy-Schwarz inequality, we have, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The remainder of the proof follows from using our chosen value of β T and algebraic manipulations (that 2 ab ≤ a 2 + b 2 ).

We now provide the proofs of these two propositions.

## 6.3.1 Regret Analysis

In this section, we prove Proposition 6.7, which says that the sum of the squares of the instantaneous regrets of the algorithm is small, assuming the evolving confidence balls always contain the true mean µ glyph[star] . An important observation is that on any round t in which µ glyph[star] ∈ BALL t , the instantaneous regret is at most the 'width' of the ellipsoid in the direction of the chosen decision. Moreover, the algorithm's choice of decisions forces the ellipsoids to shrink at a rate that ensures that the sum of the squares of the widths is small. We now formalize this.

Unless explicitly stated, all norms refer to the glyph[lscript] 2 norm.

Lemma 6.8. Let x ∈ D . If µ ∈ BALL t and x ∈ D . Then

<!-- formula-not-decoded -->

Proof: By Cauchy-Schwarz, we have:

<!-- formula-not-decoded -->

where the last inequality holds since µ BALL

Define

<!-- formula-not-decoded -->

which we interpret as the 'normalized width' at time t in the direction of the chosen decision. We now see that the width, 2 √ β t w t , is an upper bound for the instantaneous regret.

Lemma 6.9. Fix t ≤ T . If µ glyph[star] ∈ BALL t and β t ≥ 1 , then

<!-- formula-not-decoded -->

Proof: Let ˜ µ ∈ BALL t denote the vector which minimizes the dot product ˜ µ glyph[latticetop] x t . By the choice of x t , we have

<!-- formula-not-decoded -->

∈ t .

where the inequality used the hypothesis µ glyph[star] ∈ BALL t . Hence,

<!-- formula-not-decoded -->

where the last step follows from Lemma 6.8 since ˜ µ and µ glyph[star] are in BALL t . Since r t ∈ [ -1 , 1] , regret t is always at most 2 and the first inequality follows. The final inequality is due to that β t is increasing and larger than 1 .

The following two lemmas prove useful in showing that we can treat the log determinant as a potential function, where can bound the sum of widths independently of the choices made by the algorithm.

Lemma 6.10. We have:

Proof: By the definition of Σ t +1 , we have

<!-- formula-not-decoded -->

where v t := Σ -1 / 2 t x t . Now observe that v glyph[latticetop] t v t = w 2 t and

<!-- formula-not-decoded -->

Hence (1 + w 2 t ) is an eigenvalue of I + v t v glyph[latticetop] t . Since v t v glyph[latticetop] t is a rank one matrix, all other eigenvalues of I + v t v glyph[latticetop] t equal 1. Hence, det( I + v t v glyph[latticetop] t ) is (1 + w 2 t ) , implying det Σ t +1 = (1 + w 2 t ) det Σ t . The result follows by induction.

Lemma 6.11. ('Potential Function' Bound) For any sequence x 0 , . . . x T -1 such that, for t &lt; T , ‖ x t ‖ 2 ≤ B , we have:

<!-- formula-not-decoded -->

Proof: Denote the eigenvalues of ∑ T -1 t =0 x t x glyph[latticetop] t as σ 1 , . . . σ d , and note:

<!-- formula-not-decoded -->

Using the AM-GM inequality,

<!-- formula-not-decoded -->

which concludes the proof.

Finally, we are ready to prove that if µ glyph[star] always stays within the evolving confidence region, then our regret is under control.

<!-- formula-not-decoded -->

Proof: [Proof of Proposition 6.7] Assume that µ glyph[star] ∈ BALL t for all t . We have that:

<!-- formula-not-decoded -->

where the first inequality follow from By Lemma 6.9; the second from that β t is an increasing function of t ; the third uses that for 0 ≤ y ≤ 1 , ln(1 + y ) ≥ y/ 2 ; the final two inequalities follow by Lemmas 6.10 and 6.11.

## 6.3.2 Confidence Analysis

Proof: [Proof of Proposition 6.6] Since r τ = x τ · µ glyph[star] + η τ , we have:

<!-- formula-not-decoded -->

For any 0 &lt; δ t &lt; 1 , using Lemma A.9, it holds with probability at least 1 -δ t ,

<!-- formula-not-decoded -->

where we have also used the triangle inequality and that ‖ Σ -1 t ‖ ≤ 1 /λ .

We seek to lower bound Pr( ∀ t, µ glyph[star] ∈ BALL t ) . Note that at t = 0 , by our choice of λ , we have that BALL 0 contains W glyph[star] , so Pr( µ glyph[star] / ∈ BALL 0 ) = 0 . For t ≥ 1 , let us assign failure probability δ t = (3 /π 2 ) /t 2 δ for the t -th event, which, using the above, gives us an upper bound on the sum failure probability as

<!-- formula-not-decoded -->

This along with Lemma 6.11 completes the proof.

## 6.4 Bibliographic Remarks and Further Readings

The orignal multi-armed bandit model goes to back to [Robbins, 1952]. The linear bandit model was first introduced in [Abe and Long, 1999]. Our analysis of the LinUCB algorithm follows from the original proof in [Dani et al., 2008], with a simplified concentration analysis due to [Abbasi-Yadkori et al., 2011]. The first sublinear regret bound here was due to [Auer et al., 2002], which used a more complicated algorithm.

The lower bound we present is also due to [Dani et al., 2008], which also shows that LinUCB is minimax optimal.

## Chapter 7

## Strategic Exploration in Tabular MDPs

We now turn to how an agent acting in an unknown MDP can obtain a near-optimal reward over time. Compared with the previous setting with access to a generative model, we no longer have easy access to transitions at each state, but only have the ability to execute trajectories in the MDP. The main complexity this adds to the learning process is that the agent has to engage in exploration, that is, plan to reach new states where enough samples have not been seen yet, so that optimal behavior in those states can be learned.

Wewill work with finite-horizon MDPs with a fixed start state s 0 (see Section 1.2), and we will assume the agent learns in an episodic setting, as introduced in Chapter 2. Here, in in every episode k , the learner acts for H step starting from a fixed starting state s 0 and, at the end of the H -length episode, the state is reset to s 0 . It is straightforward to extend this setting where the starting state is sampled from a distribution, i.e. s 0 ∼ µ .

The goal of the agent is to minimize her expected cumulative regret over K episodes:

<!-- formula-not-decoded -->

where the expectation is with respect to the randomness of the MDP environment and, possibly, any randomness of the agent's strategy.

This chapter considers tabular MDPs, where S and A are discrete. We will now present a (nearly) sublinear regret algorithm, UCB-Value Iteration. This chapter follows the proof in [Azar et al., 2017]. We first give a simplified analysis of UCB-VI that gives a regret bound in the order of H 2 S √ AK , followed by a more refined analysis that gives a H 2 √ SAK + H 3 S 2 A regret bound.

## 7.1 On The Need for Strategic Exploration

Consider the chain MDP of length H +2 shown in Figure 0.1. The starting state of interest is state s 0 . We consider a uniformly random policy π h ( s ) that at any time step h and any state s , selects an action from all A many actions uniform randomly. It is easy to see that starting from s 0 , the probability of such uniform random policy hitting the rewarding state s H +1 is exponentially small, i.e., O ( A H ) , since it needs to select the right action that moves one step right at every time step h. This example demonstrates that we need to perform strategic exploration in order to avoid the exponential sample complexity.

a

Figure 0.1: A deterministic, chain MDP of length H +2 . Rewards are 0 everywhere other than r ( s H +1 , a 1 ) = 1 . In this example we have A = 4 .

<!-- image -->

## Algorithm 5 UCBVI

Input: reward function r (assumed to be known), confidence parameters

- 1: for k = 0 . . . K -1 do
- 3: Compute reward bonus b k h for all h (Eq. 0.2)
- 2: Compute ̂ P k h as the empirical estimates, for all h (Eq. 0.1)
- 4: Run Value-Iteration on { ̂ P k h , r + b k h } H -1 h =0 (Eq. 0.3)
- 5: Set π k as the returned policy of VI.
- 6: end for

## 7.2 The UCB-VI algorithm

If the learner then executes π k in the underlying MDP to generate a single trajectory τ k = { s k h , a k h } H -1 h =0 with a h = π k h ( s k h ) and s k h +1 ∼ P h ( ·| s k h , a k h ) . We first define some notations below. Consider the very beginning of episode k . We use the history information up to the end of episode k -1 (denoted as H &lt;k ) to form some statistics. Specifically, we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Namely, we maintain counts of how many times s, a, s ′ and s, a are visited at time step h from the beginning of the learning process to the end of the episode k -1 . We use these statistics to form an empirical model:

<!-- formula-not-decoded -->

We will also use the counts to define a reward bonus , denoted as b h ( s, a ) for all h, s, a . Denote L := ln ( SAHK/δ ) ( δ as usual represents the failure probability which we will define later). We define reward bonus as follows:

<!-- formula-not-decoded -->

With reward bonus and the empirical model, the learner uses Value Iteration on the empirical transition ̂ P k h and the combined reward r h + b k h . Starting at H (note that H is a fictitious extra step as an episode terminates at H -1 ), we

perform dynamic programming all the way to h = 0 :

<!-- formula-not-decoded -->

Note that when using ̂ V k h +1 to compute ̂ Q k h , we truncate the value by H . This is because we know that due to the assumption that r ( s, a ) ∈ [0 , 1] , no policy's Q value will ever be larger than H .

Denote π k = { π k 0 , . . . , π k H -1 } . Learner then executes π k in the MDP to get a new trajectory τ k .

UCBVI repeats the above procedure for K episodes.

## 7.3 Analysis

We will prove the following theorem.

Theorem 7.1 (Regret Bound of UCBVI) . UCBVI achieves the following regret bound:

<!-- formula-not-decoded -->

Remark While the above regret is sub-optimal, the algorithm we presented here indeed achieves a sharper bound in the leading term ˜ O ( H 2 √ SAK ) [Azar et al., 2017], which gives the tight dependency bound on S, A, K . In Section 7.4, we will present a refined analysis that indeed achieves H 2 √ SAK regret. The dependency on H is not tight and tightening the dependency on H requires modifications to the reward bonus (use Bernstein inequality rather than Hoeffding's inequality for reward bonus design).

We prove the above theorem in this section.

We start with bounding the error from the learned model ̂ P k h .

Lemma 7.2 (State-action wise model error) . Fix δ ∈ (0 , 1) . For all k ∈ [0 , . . . , K -1] , s ∈ S , a ∈ A , h ∈ [0 , . . . , H -1] , with probability at least 1 -δ , we have that for any f : S ↦→ [0 , H ] :

<!-- formula-not-decoded -->

Remark The proof of the above lemma requires a careful argument using Martingale difference sequence mainly because N k h ( s, a ) itself is a random quantity. We give a proof the above lemma at the end of this section (Sec. 7.3.1). Using the above lemma will result a sub-optimal regret bound in terms of the dependence on S . In Section 7.4, we will give a different approach to bound ∣ ∣ ∣ ∣ ( ̂ P k h ( ·| s, a ) -P glyph[star] h ( ·| s, a ) ) glyph[latticetop] f ∣ ∣ ∣ ∣ which leads to an improved regret bound.

The following lemma is still about model error, but this time we consider an average model error with respect to V glyph[star] -adeterministic quantity.

Lemma 7.3 (State-action wise average model error under V glyph[star] ) . Fix δ ∈ (0 , 1) . For all k ∈ [1 , . . . , K -1] , s ∈ S , a ∈ A , h ∈ [0 , . . . , H -1] , and consider V glyph[star] h : S → [0 , H ] , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Note that we can use Lemma 7.2 to bound ∣ ∣ ∣ ̂ P k h ( ·| s, a ) · V glyph[star] h +1 -P glyph[star] h ( ·| s, a ) · V glyph[star] h +1 ∣ ∣ ∣ . However, the above lemma shows that by leveraging the fact that V glyph[star] is a deterministic quantity (i.e., independent of the data collected during learning), we can get a tighter upper bound which does not scale polynomially with respect to S .

Proof: Consider a fixed tuple s, a, k, h first. By the definition of ̂ P k h , we have:

<!-- formula-not-decoded -->

Now denote H h,i as the entire history from t = 0 to iteration t = i where in iteration i , H h,i includes history from time step 0 up to and including time step h . We define random variables: X i = 1 { ( s i h , a i h ) = ( s, a ) } V glyph[star] h +1 ( s i h +1 ) -E [ 1 { ( s i h , a i h ) = ( s, a ) } V glyph[star] h +1 ( s ′ h +1 ) |H h,i ] for i = 0 , . . . , K -1 . These random variables have the following properties. First | X i | ≤ H if 1 { ( s i h , a i h ) = ( s, a ) } = 1 , else | X i | = 0 , which implies that ∑ k i =0 | X i | 2 ≤ N k h ( s, a ) . Also we have E [ X i |H h,i ] = 0 for all i . Thus, { X i } i ≥ 0 is a Martingale difference sequence. Using Azuma-Hoeffding's inequality, we get that for any fixed k , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Divide N k h ( s, a ) on both sides and use the definition of ̂ P h ( ·| s, a ) glyph[latticetop] V glyph[star] h +1 , we get:

<!-- formula-not-decoded -->

With a union bound over all s ∈ S , a ∈ A , k ∈ [ N ] , h ∈ [ H ] , we conclude the proof.

We denote the two inequalities in Lemma 7.2 and Lemma 7.3 as event E model . Note that the failure probability of E model is at most 2 δ . Below we condition on E model being true (we deal with failure event at the very end).

Now we study the effect of reward bonus. Similar to the idea in multi-armed bandits, we want to pick a policy π k , such that the value of π k in under the combined reward r h + b k h and the empirical model ̂ P k h is optimistic, i.e., we want ̂ V k 0 ( s 0 ) ≥ V glyph[star] 0 ( s 0 ) for all s 0 . The following lemma shows that via reward bonus, we are able to achieve this optimism.

Lemma 7.4 (Optimism) . Assume E model is true. For all episode k , we have:

<!-- formula-not-decoded -->

where ̂ V k h is computed based on VI in Eq. 0.3.

Proof: We prove via induction. At the additional time step H we have ̂ V k H ( s ) = V glyph[star] H ( s ) = 0 for all s .

Starting at h +1 , and assuming we have ̂ V k h +1 ( s ) ≥ V glyph[star] h +1 ( s ) for all s , we move to h below.

Consider any s, a ∈ S × A . First, if Q k h ( s, a ) = H , then we have Q k h ( s, a ) ≥ Q glyph[star] h ( s, a ) .

<!-- formula-not-decoded -->

where the first inequality is from the inductive hypothesis, and the last inequality uses Lemma 7.3 and the definition of bonus.

From ̂ Q k h , one can finish the proof by showing ̂ V k h ( s ) ≥ V glyph[star] h ( s ) , ∀ s .

Before we prove our final result, one more technical lemma will be helpful:

Lemma 7.5. Consider arbitrary K sequence of trajectories τ k = { s k h , a k h } H -1 h =0 for k = 0 , . . . , K -1 . We have

<!-- formula-not-decoded -->

Proof: We swap the order of the two summation above:

<!-- formula-not-decoded -->

where in the first inequality we use the fact that ∑ N i =1 1 / √ i ≤ 2 √ N , and in the second inequality we use CS inequality.

The proof of our main theorem now follows:

## Proof: [Proof of Theorem 7.1]

Let us consider episode k and denote H &lt;k as the history up to the end of episode k -1 . We consider bounding V glyph[star] -V π k . Using optimism and the simulation lemma, we can get the following result:

<!-- formula-not-decoded -->

We prove the above two inequalities in the lecture. We leave the proof of the above inequality (Eq 0.4) as an exercise for readers. Note that this is slightly different from the usual simulation lemma, as here we truncate ̂ V by H during VI.

<!-- formula-not-decoded -->

where recall L := ln( SAKH/δ ) . Hence, for the per-episode regret V glyph[star] ( s 0 ) -V π k ( s 0 ) , we obtain:

<!-- formula-not-decoded -->

where in the last term the expectation is taken with respect to the trajectory { s k h , a k h } (which is generated from π k ) while conditioning on all history H &lt;k up to and including the end of episode k -1 .

Now we sum all episodes together and take the failure event into consideration:

<!-- formula-not-decoded -->

where the last inequality uses the law of total expectation.

We can bound the double summation term above using lemma 7.5. We can conclude that:

<!-- formula-not-decoded -->

Now set δ = 1 /KH , we get:

<!-- formula-not-decoded -->

This concludes the proof of Theorem 7.1.

## 7.3.1 Proof of Lemma 7.2

Here we include a proof of Lemma 7.2. The proof consists of two steps, first we using Martingale concentration to bound | ( ̂ P k h ( ·| s, a ) -P glyph[star] h ( ·| s, a )) glyph[latticetop] f | for all s, a, h, k , but for a fixed f : S ↦→ [0 , H ] . Note that there are infinitely many such f that maps from S to [0 , H ] . Thus, the second step is to use a covering argument (i.e., glyph[epsilon1] -net on all { f : S ↦→ [0 , H ] } ) to show that | ( ̂ P k h ( ·| s, a ) -P glyph[star] h ( ·| s, a )) glyph[latticetop] f | is bounded for all s, a, h, k , and all f : S ↦→ [0 , H ] .

Proof: [Proof of Lemma 7.2] We consider a fixed tuple s, a, k, h, f first. Recall the definition of ̂ P k h ( s, a ) , and we have: ̂ P k h ( ·| s, a ) glyph[latticetop] f = ∑ k -1 i =0 1 { s i h , a i h = s, a } f ( s i h +1 ) /N k h ( s, a ) .

glyph[negationslash]

Define H h,i as the history starting from the beginning of iteration 0 all the way up to and include time step h at iteration i . Define random variables X i as X i = 1 { s i h , a i h = s, a } f ( s i h +1 ) -1 { s i h , a i h = s, a } E s ′ ∼ P glyph[star] h ( s,a ) f ( s ′ ) . We now show that { X i } is a Martingale difference sequence. First we have E [ X i |H h,i ] = 0 , since 1 { s i h , a i h = s, a } is a deterministic quantity given H h,i . Second, we have | X i | = 0 for ( s i h , a i h ) = ( s, a ) , and | X i | ≤ H when ( s i h , a i h ) = ( s, a ) . Thus, we have shown that it is a Martingale difference sequence.

Now we can apply Azuma-Hoeffding's inequality (Theorem A.5). With probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Now apply union bound over all s ∈ S , a ∈ A , h ∈ [0 , . . . , H -1] , k ∈ [0 , . . . , K -1] , we have that with probability at least 1 -δ , for all s, a, h, k , we have:

<!-- formula-not-decoded -->

Note that the above result only holds for the fixed f that we specified at the beginning of the proof. We are left to perform a covering argument over all functions that map from S to [0 , H ] . We use a standard glyph[epsilon1] -net argument for that. We abuse notation a bit and denote f as a vector in [0 , H ] S .

Note that ‖ f ‖ 2 ≤ H √ S for any f : S ↦→ [0 , H ] . A standard glyph[epsilon1] -net argument shows that we can construct an glyph[epsilon1] -net N glyph[epsilon1] over [0 , H ] S with |N glyph[epsilon1] | ≤ (1 + 2 H √ S/glyph[epsilon1] ) S , such that for any f ∈ [0 , H ] S , there exists a point f ′ ∈ N glyph[epsilon1] , such that ‖ f -f ′ ‖ 2 ≤ glyph[epsilon1] . Note that the glyph[epsilon1] -net N glyph[epsilon1] is independent of the training data during the learning process.

Now using Eq. 0.6 and a union bound over all f in N glyph[epsilon1] , we have that with probability at least 1 -δ , for all s, a, k, h, f ∈ N glyph[epsilon1] , we have:

<!-- formula-not-decoded -->

Finally, for any f ∈ [0 , H ] S , denote its closest point in N glyph[epsilon1] as f ′ , we have:

<!-- formula-not-decoded -->

where in the last inequality we use the fact that if ‖ f -f ′ ‖ 2 ≤ glyph[epsilon1] , then | f ( s ) -f ′ ( s ) | ≤ glyph[epsilon1], ∀ s ∈ S .

Now we can set glyph[epsilon1] = 1 /K and use the fact that N k h ( s, a ) ≤ K , we have:

<!-- formula-not-decoded -->

## 7.4 An Improved Regret Bound

In this section, we show that UCB-VI also enjoys the regret bound ˜ O ( H 2 √ SAK + H 3 S 2 A ) .

Theorem 7.6 (Improved Regret Bound for UCB-VI) . The total expected regret is bounded as follows:

<!-- formula-not-decoded -->

Note that comparing to the regret bound we had in the previous section, this new regret improves the S dependence in the leading order term (i.e., the term that contains √ K ), but at the cost of introducing a lower order term H 3 S 2 A .

We first give an improved version of the model error presented in Lemma 7.2.

Lemma 7.7. With probability at least 1 -δ , for all h, k, s, a , and all f : S ↦→ [0 , H ] , we have:

<!-- formula-not-decoded -->

Below we prove this lemma. Recall the definition of ̂ P k h ( s ′ | s, a ) in Eq. 0.1. We can use Azume-Bernstein's inequality to bound | ̂ P k h ( s ′ | s, a ) -P glyph[star] h ( s ′ | s, a ) | .

Lemma 7.8. With probability at least 1 -δ , for all s, a, k, h , we have:

<!-- formula-not-decoded -->

where L := ln( SAKH/δ ) .

Proof: Consider a fixed tuple ( s, a, h, k ) . Define H h,i as the history starting from the beginning of iteration 0 to time step h at iteration i (including step h , i.e., up to s i h , a i h ). Define random variables { X i } i ≥ 0 as X i = 1 { s i h , a i h , s i h +1 = s, a, s ′ } -1 { s i h , a i h = s, a } P glyph[star] h ( s ′ | s, a ) . We now show that { X i } is a Martingale difference sequence. Note that E [ X i |H h,i ] = 0 . We have | X i | ≤ 1 . To use Azuma-Bernstein's inequality, we note that E [ X 2 i |H h,i ] is bounded as:

<!-- formula-not-decoded -->

where we use the fact that the variance of a Bernoulli with parameter p is p (1 -p ) . This means that ∑ k -1 i =0 E [ X 2 i |H h,i ] = P glyph[star] h ( s ′ | s, a )(1 -P glyph[star] h ( s ′ | s, a )) N k h ( s, a ) . Now apply Bernstein's inequality on the martingale difference sequence { X i } , we have:

<!-- formula-not-decoded -->

Recall the definition of L := ln( SAHK/δ ) . Apply a union bound over all s ∈ S , a ∈ A , h ∈ [0 , . . . , H -1] , k ∈ [0 , . . . , K -1] , we conclude the proof.

Now we are ready to prove Lemma 7.7.

Proof: [Proof of Lemma 7.7] Let us condition on the event in Lemma 7.8 being true, of which the probability is at least 1 -δ .

Take any function f : S ↦→ [0 , H ] , we have:

<!-- formula-not-decoded -->

where the third inequality uses that condition that ‖ f ‖ ∞ ≤ H , the fourth inequality uses Cauchy-Schwarz inequality, the fifth inequality uses the inequality that ab ≤ ( a 2 + b 2 ) / 2 for any real numbers a and b , and the last inequality uses the condition that ‖ f ‖ ∞ ≤ H again. This concludes the proof.

So far in Lemma 7.7, we have not specify what function f we will use. The following lemma instantiate f to be ̂ V k h +1 -V glyph[star] h +1 . Recall the bonus definition, b k h ( s, a ) := 2 H √ L/N k h ( s, a ) where L := ln( SAHK/δ ) .

Lemma 7.9. Assume the events in Lemma 7.7 and Lemma 7.3 are true. For all s, a, k, h , set f := ̂ V k h +1 -V glyph[star] h +1 , we have:

<!-- formula-not-decoded -->

where ξ i τ ( s, a ) = 3 H 2 SL N i τ ( s,a ) , for any τ ∈ [0 , . . . , H -1] , i ∈ [0 , . . . , K -1] , and the expectation is with respect to the randomness from the event of following policy π k from ( s, a ) starting from time step h .

Proof: Using Lemma 7.7 and the fact that f := ̂ V k h +1 -V glyph[star] h +1 , we have:

<!-- formula-not-decoded -->

Now consider ̂ V k h +1 ( s ′ ) -V glyph[star] h +1 ( s ′ )) , we have:

<!-- formula-not-decoded -->

where the first inequality uses the fact that V glyph[star] h +1 ( s ′ ) = max a ′ ( r ( s ′ , a ′ ) + P glyph[star] h +1 ( ·| s ′ , a ′ ) glyph[latticetop] V glyph[star] h +2 ) , the second inequality uses the definition of ̂ V k h +1 and π k . Note that the second term in the RHS of the above inequality can be upper bounded exactly by the bonus using Lemma 7.3, and the first term can be further upper bounded using the same operation that we had in Lemma 7.7, i.e.,

<!-- formula-not-decoded -->

Combine the above inequalities together, we arrive at:

<!-- formula-not-decoded -->

We can recursively apply the same operation on ̂ V k h +2 ( s ′′ ) -V glyph[star] h +2 ( s ′′ ) till step H , we have:

<!-- formula-not-decoded -->

where the last inequality uses the fact that (1 + 1 /H ) x ≤ e for any x ∈ N + . Substitute the above inequality into Eq. 0.7, we get:

<!-- formula-not-decoded -->

Now we are ready to derive a refined upper bound for per-episode regret.

Lemma 7.10. Assume the events in Lemma 7.9 and Lemma 7.4 (optimism) hold. For all k ∈ [0 , K -1] , we have:

<!-- formula-not-decoded -->

Proof: Via optimism in Lemma 7.4, we have:

<!-- formula-not-decoded -->

where the first inequality uses optimism, and the fourth inequality uses Lemma 7.9, and the last inequality rearranges by grouping same terms together.

To conclude the final regret bound, let us denote the event E model as the joint event in Lemma 7.3 and Lemma 7.8, which holds with probability at least 1 -2 δ .

Proof: [Proof of Theorem 7.6] Following the derivation we have in Eq. 0.5, we decompose the expected regret bound as follows:

<!-- formula-not-decoded -->

where the last inequality uses Lemma 7.10. Using Lemma 7.5, we have that:

<!-- formula-not-decoded -->

For ∑ k ∑ h ξ k h ( s k h , a k h ) , we have:

<!-- formula-not-decoded -->

where in the last inequality, we use the fact that N K h ( s, a ) ≤ K for all s, a, h , and the inequality ∑ t i =1 1 /i ≤ ln( t ) . Putting things together, and setting δ = 1 / ( KH ) , we conclude the proof.

## 7.5 Phased Q -learning

to be added

## 7.6 Bibliographic Remarks and Further Readings

The first provably correct PAC algorithm for reinforcement learning (which finds a near optimal policy) was due to Kearns and Singh [2002], which provided the E 3 algorithm; it achieves polynomial sample complexity in tabular MDPs. Brafman and Tennenholtz [2002] presents the Rmax algorithm which provides a refined PAC analysis over E 3 . Both are model based approaches [Kakade, 2003] improves the sample complexity to be O ( S 2 A ) . Both E 3 and Rmax uses the concept of absorbing MDPs to achieve optimism and balance exploration and exploitation.

Jaksch et al. [2010] provides the first O ( √ ( T )) regret bound, where T is the number of timesteps in the MDP ( T is proportional to K in our setting); this dependence on T is optimal. Subsequently, Azar et al. [2017], Dann et al. [2017] provide algorithms that, asymptotically, achieve minimax regret bound in tabular MDPs. By this, we mean that for sufficiently large T (for T ≥ Ω( |S| 2 ) ), the results in Azar et al. [2017], Dann et al. [2017] obtain optimal dependencies on |S| and |A| . The requirement that T ≥ Ω( |S| 2 ) before these bounds hold is essentially the requirement that nontrivial model accuracy is required. It is an opent question to remove this dependence.

Lower bounds are provided in [Dann and Brunskill, 2015, Osband and Van Roy, 2016, Azar et al., 2017].

Further exploration strategies. Refs and discussion for Q -learning, reward free, and thompson sampling to be added...

## Chapter 8

## Linearly Parameterized MDPs

In this chapter, we consider learning and exploration in linearly parameterized MDPs-the linear MDP. Linear MDP generalizes tabular MDPs into MDPs with potentially infinitely many state and action pairs.

This chapter largely follows the model and analysis first provided in [Jin et al., 2020].

## 8.1 Setting

We consider episodic finite horizon MDP with horizon H , M = {S , A , { r h } h , { P h } h , H, s 0 } , where s 0 is a fixed initial state, r h : S × A ↦→ [0 , 1] and P h : S × A ↦→ ∆( S ) are time-dependent reward function and transition kernel. Note that for time-dependent finite horizon MDP, the optimal policy will be time-dependent as well. For simplicity, we overload notations a bit and denote π = { π 0 , . . . , π H -1 } , where each π h : S ↦→ A . We also denote V π := V π 0 ( s 0 ) , i.e., the expected total reward of π starting at h = 0 and s 0 .

Wedefine the learning protocol below. Learning happens in an episodic setting. Every episode k , learner first proposes a policy π k based on all the history information up to the end of episode k -1 . The learner then executes π k in the underlying MDP to generate a single trajectory τ k = { s k h , a k h } H -1 h =0 with a h = π k h ( s k h ) and s k h +1 ∼ P h ( ·| s k h , a k h ) . The goal of the learner is to minimize the following cumulative regret over N episodes:

<!-- formula-not-decoded -->

where the expectation is with respect to the randomness of the MDP environment and potentially the randomness of the learner (i.e., the learner might make decisions in a randomized fashion).

## 8.1.1 Low-Rank MDPs and Linear MDPs

Note that here we do not assume S and A are finite anymore. Indeed in this note, both of them could be continuous. Without any further structural assumption, the lower bounds we saw in the Generalization Lecture forbid us to get a polynomially regret bound.

The structural assumption we make in this note is a linear structure in both reward and the transition.

Definition 8.1 (Linear MDPs) . Consider transition { P h } and { r h } h . Alinear MDP has the following structures on r h

and P h :

<!-- formula-not-decoded -->

where φ is a known state-action feature map φ : S ×A ↦→ R d , and µ glyph[star] h ∈ R |S|× d . Here φ, θ glyph[star] h are known to the learner, while µ glyph[star] is unknown. We further assume the following norm bound on the parameters: (1) sup s,a ‖ φ ( s, a ) ‖ 2 ≤ 1 , (2) ‖ v glyph[latticetop] µ glyph[star] h ‖ 2 ≤ √ d for any v such that ‖ v ‖ ∞ ≤ 1 , and all h , and (3) ‖ θ glyph[star] h ‖ 2 ≤ W for all h . We assume r h ( s, a ) ∈ [0 , 1] for all h and s, a .

The model essentially says that the transition matrix P h ∈ R |S|×|S||A| has rank at most d , and P h = µ glyph[star] h Φ . where Φ ∈ R d ×|S||A| and each column of Φ corresponds to φ ( s, a ) for a pair s, a ∈ S × A .

Linear Algebra Notations For real-valued matrix A , we denote ‖ A ‖ 2 = sup x : ‖ x ‖ 2 =1 ‖ Ax ‖ 2 which denotes the maximum singular value of A . We denote ‖ A ‖ F as the Frobenius norm ‖ A ‖ 2 F = ∑ i,j A 2 i,j where A i,j denotes the i, j 'th entry of A . For any Positive Definite matrix Λ , we denote x glyph[latticetop] Λ x = ‖ x ‖ 2 Λ . We denote det( A ) as the determinant of the matrix A . For a PD matrix Λ , we note that det(Λ) = ∏ d i =1 σ i where σ i is the eigenvalues of Λ . For notation simplicity, during inequality derivation, we will use glyph[lessorsimilar] , glyph[equalorsimilar] to suppress all absolute constants. We will use ˜ O to suppress all absolute constants and log terms.

## 8.2 Planning in Linear MDPs

We first study how to do value iteration in linear MDP if µ is given.

We start from Q glyph[star] H -1 ( s, a ) = θ glyph[star] H -1 · φ ( s, a ) , and π glyph[star] H -1 ( s ) = argmax a Q glyph[star] H -1 ( s, a ) = argmax a θ glyph[star] H -1 · φ ( s, a ) , and V glyph[star] H -1 ( s ) = argmax a Q glyph[star] H -1 ( s, a ) .

Now we do dynamic programming from h +1 to h :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we denote w h := θ glyph[star] h +( µ glyph[star] h ) glyph[latticetop] V glyph[star] h +1 . Namely we see that Q glyph[star] h ( s, a ) is a linear function with respect to φ ( s, a ) ! We can continue by defining π glyph[star] h ( s ) = argmax a Q glyph[star] h ( s, a ) and V glyph[star] h ( s ) = max a Q glyph[star] h ( s, a ) .

At the end, we get a sequence of linear Q glyph[star] , i.e., Q glyph[star] h ( s, a ) = w h · φ ( s, a ) , and the optimal policy is also simple, π glyph[star] h ( s ) = argmax a w h · φ ( s, a ) , for all h = 0 , . . . , H -1 .

One key property of linear MDP is that a Bellman Backup of any function f : S ↦→ R is a linear function with respect to φ ( s, a ) . We summarize the key property in the following claim.

Claim 8.2. Consider any arbitrary function f : S ↦→ [0 , H ] . At any time step h ∈ [0 , . . . H -1] , there must exist a w ∈ R d , such that, for all s, a ∈ S × A :

<!-- formula-not-decoded -->

The proof of the above claim is essentially the Eq. 0.1.

## 8.3 Learning Transition using Ridge Linear Regression

In this section, we consider the following simple question: given a dataset of state-action-next state tuples, how can we learn the transition P h for all h ?

Note that µ glyph[star] ∈ R |S|× d . Hence explicitly writing down and storing the parameterization µ glyph[star] takes time at least |S| . We show that we can represent the model in a non-parametric way.

We consider a particular episode n . Similar to Tabular-UCBVI, we learn a model at the very beginning of the episode n using all data from the previous episodes (episode 1 to the end of the episode n -1 ). We denote such dataset as:

<!-- formula-not-decoded -->

We maintain the following statistics using D n h :

<!-- formula-not-decoded -->

where λ ∈ R + (it will be set to 1 eventually, but we keep it here for generality).

To get intuition of Λ n , think about the tabular setting where φ ( s, a ) is a one-hot vector (zeros everywhere except that the entry corresponding to ( s, a ) is one). Then Λ n h is a diagonal matrix and the diagonal entry contains N n ( s, a ) -the number of times ( s, a ) has been visited.

We consider the following multi-variate linear regression problem. Denote δ ( s ) as a one-hot vector that has zero everywhere except that the entry corresponding to s is one. Denote glyph[epsilon1] i h = P ( ·| s i h , a i h ) -δ ( s i h +1 ) . Conditioned on history H i h (history H i h denotes all information from the very beginning of the learning process up to and including ( s i h , a i h ) ), we have:

<!-- formula-not-decoded -->

simply because s i h +1 is sampled from P h ( ·| s i h , a i h ) conditioned on ( s i h , a i h ) . Also note that ‖ glyph[epsilon1] i h ‖ 1 ≤ 2 for all h, i .

Since µ glyph[star] h φ ( s i h , a i h ) = P h ( ·| s i h , a i h ) , and δ ( s i h +1 ) is an unbiased estimate of P h ( ·| s i h , a i h ) conditioned on s i h , a i h , it is reasonable to learn µ glyph[star] via regression from φ ( s i h , a i h ) to δ ( s i h +1 ) . This leads us to the following ridge linear regression:

<!-- formula-not-decoded -->

Ridge linear regression has the following closed-form solution:

<!-- formula-not-decoded -->

Note that ̂ µ n h ∈ R |S|× d , so we never want to explicitly store it. Note that we will always use ̂ µ n h together with a specific s, a pair and a value function V (think about value iteration case), i.e., we care about ̂ P n h ( ·| s, a ) · V := ( ̂ µ n h φ ( s, a )) · V , which can be re-written as:

<!-- formula-not-decoded -->

where we use the fact that δ ( s ) glyph[latticetop] V = V ( s ) . Thus the operator ̂ P n h ( ·| s, a ) · V simply requires storing all data and can be computed via simple linear algebra and the computation complexity is simply poly ( d, n ) -no poly dependency on |S| .

Let us calculate the difference between ̂ µ n h and µ glyph[star] h .

Lemma 8.3 (Difference between ̂ µ h and µ glyph[star] h ) . For all n and h , we must have:

<!-- formula-not-decoded -->

Proof: We start from the closed-form solution of ̂ µ n h :

<!-- formula-not-decoded -->

Rearrange terms, we conclude the proof.

Lemma 8.4. Fix V : S ↦→ [0 , H ] . For all n and s, a ∈ S × A , and h , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Proof: We first check the noise terms { V glyph[latticetop] glyph[epsilon1] i h } h,i . Since V is independent of the data (it's a pre-fixed function), and by linear property of expectation, we have:

<!-- formula-not-decoded -->

Hence, this is a Martingale difference sequence. Using the Self-Normalized vector-valued Martingale Bound (Lemma A.9), we have that for all n , with probability at least 1 -δ :

<!-- formula-not-decoded -->

Apply union bound over all h ∈ [ H ] , we get that with probability at least 1 -δ , for all n, h :

<!-- formula-not-decoded -->

## 8.4 Uniform Convergence via Covering

Now we take a detour first and consider how to achieve a uniform convergence result over a function class F that contains infinitely many functions. Previously we know how to get uniform convergence if F is finite-we simply do a union bound. However, when F contains infinitely many functions, we cannot simply apply a union bound. We will use the covering argument here.

Consider the following ball with radius R : Θ = { θ ∈ R d : ‖ θ ‖ 2 ≤ R ∈ R + } . Fix an glyph[epsilon1] . An glyph[epsilon1] -net N glyph[epsilon1] ⊂ Θ is a set such that for any θ ∈ Θ , there exists a θ ′ ∈ N glyph[epsilon1] , such that ‖ θ -θ ′ ‖ 2 ≤ glyph[epsilon1] . We call the smallest glyph[epsilon1] -net as glyph[epsilon1] -cover. Abuse notations a bit, we simply denote N glyph[epsilon1] as the glyph[epsilon1] -cover.

The glyph[epsilon1] -covering number is the size of glyph[epsilon1] -cover N glyph[epsilon1] . We define the covering dimension as ln ( |N glyph[epsilon1] | )

Lemma 8.5. The glyph[epsilon1] -covering number of the ball Θ = { θ ∈ R d : ‖ θ ‖ 2 ≤ R ∈ R + } is upper bounded by (1 + 2 R/glyph[epsilon1] ) d .

We can extend the above definition to a function class. Specifically, we look at the following function. For a triple of ( w,β, Λ) where w ∈ R d and ‖ w ‖ 2 ≤ L , β ∈ [0 , B ] , and Λ such that σ min (Λ) ≥ λ , we define f w,β, Λ : S ↦→ [0 , R ] as follows:

<!-- formula-not-decoded -->

We denote the function class F as:

<!-- formula-not-decoded -->

Note that F contains infinitely many functions as the parameters are continuous. However we will show that it has finite covering number that scales exponentially with respect to the number of parameters in ( w,β, Λ) .

Why do we look at F ? As we will see later in this chapter F contains all possible ̂ Q h functions one could encounter during the learning process.

Lemma 8.6 ( glyph[epsilon1] -covering dimension of F ) . Consider F defined in Eq. 0.6. Denote its glyph[epsilon1] -cover as N glyph[epsilon1] with the glyph[lscript] ∞ norm as the distance metric, i.e., d ( f 1 , f 2 ) = ‖ f 1 -f 2 ‖ ∞ for any f 1 , f 2 ∈ F . We have that:

<!-- formula-not-decoded -->

Note that the glyph[epsilon1] -covering dimension scales quadratically with respect to d .

Proof: We start from building a net over the parameter space ( w,β, Λ) , and then we convert the net over parameter space to an glyph[epsilon1] -net over F under the glyph[lscript] ∞ distance metric.

We pick two functions that corresponding to parameters ( w,β, Λ) and ( ˆ w, ˆ β, ̂ Λ) .

<!-- formula-not-decoded -->

Note that Λ -1 is a PD matrix with σ max (Λ -1 ) ≤ 1 /λ .

<!-- formula-not-decoded -->

Now we consider the glyph[epsilon1]/ 3 -Net N glyph[epsilon1]/ 3 ,w over { w : ‖ w ‖ 2 ≤ L } , √ λglyph[epsilon1]/ 3 -net N √ λglyph[epsilon1]/ 3 ,β over interval [0 , B ] for β , and glyph[epsilon1] 2 / (9 B 2 ) -net N glyph[epsilon1] 2 / (9 B ) , Λ over { Λ : ‖ Λ ‖ F ≤ √ d/λ } . The product of these three nets provide a glyph[epsilon1] -cover for F , which means that that size of the glyph[epsilon1] -net N glyph[epsilon1] for F is upper bounded as:

Remark Covering gives a way to represent the complexity of function class (or hypothesis class). Relating to VC, covering number is upper bound roughly by exp( d ) with d being the VC-dimension. However, there are cases where VC-dimensional is infinite, but covering number if finite.

Now we can build a uniform convergence argument for all f ∈ F .

<!-- formula-not-decoded -->

Lemma 8.7 (Uniform Convergence Results) . Set λ = 1 . Fix δ ∈ (0 , 1) . For all n, h , all s, a , and all f ∈ F , with probability at least 1 -δ , we have:

Proof: Recall Lemma 8.4, we have with probability at least 1 -δ , for all n, h , for a pre-fixed V (independent of the random process):

<!-- formula-not-decoded -->

where we have used the fact that ‖ φ ‖ 2 ≤ 1 , λ = 1 , and ‖ Λ n h ‖ 2 ≤ N +1 .

Denote the glyph[epsilon1] -cover of F as N glyph[epsilon1] . With an application of a union bound over all functions in N glyph[epsilon1] , we have that with probability at least 1 -δ , for all V ∈ N glyph[epsilon1] , all n, h , we have:

<!-- formula-not-decoded -->

Recall Lemma 8.6, substitute the expression of ln |N glyph[epsilon1] | into the above inequality, we get:

<!-- formula-not-decoded -->

Now consider an arbitrary f ∈ F . By the definition of glyph[epsilon1] -cover, we know that for f , there exists a V ∈ N glyph[epsilon1] , such that ‖ f -V ‖ ∞ ≤ glyph[epsilon1] . Thus, we have:

<!-- formula-not-decoded -->

where in the second inequality we use the fact that ‖ ∑ n -1 i =1 φ ( s i h , a i h )( V -f ) glyph[latticetop] glyph[epsilon1] i h ‖ 2 (Λ n h ) -1 ≤ 4 glyph[epsilon1] 2 N , which is from

<!-- formula-not-decoded -->

Set glyph[epsilon1] = 1 / √ N , we get:

<!-- formula-not-decoded -->

where we recall glyph[lessorsimilar] ignores absolute constants.

Now recall that we can express ( ̂ P n h ( ·| s, a ) -P ( ·| s, a ) ) · f = φ ( s, a ) glyph[latticetop] ( ̂ µ n h -µ glyph[star] h ) glyph[latticetop] f . Recall Lemma 8.3, we have:

<!-- formula-not-decoded -->

## 8.5 Algorithm

Our algorithm, Upper Confidence Bound Value Iteration (UCB-VI) will use reward bonus to ensure optimism. Specifically, we will the following reward bonus, which is motivated from the reward bonus used in linear bandit:

<!-- formula-not-decoded -->

where β contains poly of H and d , and other constants and log terms. Again to gain intuition, please think about what this bonus would look like when we specialize linear MDP to tabular MDP.

## Algorithm 6 UCBVI for Linear MDPs

- 1: Input: parameters β, λ
- 2: for n = 1 . . . N do
- 3: Compute ̂ P n h for all h (Eq. 0.3)
- 4: Compute reward bonus b n h for all h (Eq. 0.7)
- 5: Run Value-Iteration on { ̂ P n h , r h + b n h } H -1 h =0 (Eq. 0.8)
- 6: Set π n as the returned policy of VI.
- 7: end for

With the above setup, now we describe the algorithm. Every episode n , we learn the model ̂ µ n h via ridge linear regression. We then form the quadratic reward bonus as shown in Eq. 0.7. With that, we can perform the following truncated Value Iteration (always truncate the Q function at H ):

<!-- formula-not-decoded -->

Note that above ̂ Q n h contains two components: a quadratic component and a linear component. And ̂ V n h has the format of f w,β, Λ defined in Eq. 0.5.

The following lemma bounds the norm of linear weights in ̂ Q n h .

Lemma 8.8. Assume β ∈ [0 , B ] . For all n, h , we have ̂ V n h is in the form of Eq. 0.5, and ̂ V n h falls into the following class:

<!-- formula-not-decoded -->

Proof: We just need to show that θ glyph[star] +( ̂ µ n h ) glyph[latticetop] ̂ V n h +1 has its glyph[lscript] 2 norm bounded. This is easy to show as we always have ‖ ̂ V n h +1 ‖ ∞ ≤ H as we do truncation at Value Iteration:

<!-- formula-not-decoded -->

Now we use the closed-form of ̂ µ n h from Eq. 0.3:

<!-- formula-not-decoded -->

where we use the fact that ‖ ̂ V n h +1 ‖ ∞ ≤ H , σ max (Λ -1 ) ≤ 1 /λ , and sup s,a ‖ φ ( s, a ) ‖ 2 ≤ 1 .

## 8.6 Analysis of UCBVI for Linear MDPs

In this section, we prove the following regret bound for UCBVI.

Theorem 8.9 (Regret Bound) . Set β = ˜ O ( Hd ) , λ = 1 . UCBVI (Algorithm 6) achieves the following regret bound:

<!-- formula-not-decoded -->

The main steps of the proof are similar to the main steps of UCBVI in tabular MDPs. We first prove optimism via induction, and then we use optimism to upper bound per-episode regret. Finally we use simulation lemma to decompose the per-episode regret.

In this section, to make notation simple, we set λ = 1 directly.

## 8.6.1 Proving Optimism

Proving optimism requires us to first bound model error which we have built in the uniform convergence result shown in Lemma 8.7, namely, the bound we get for ( ̂ P n h ( ·| s, a ) -P ( ·| s, a )) · f for all f ∈ V . Recall Lemma 8.7 but this time replacing F by V defined in Eq. 0.9. With probability at least 1 -δ , for all n, h, s, a and for all f ∈ V ,

<!-- formula-not-decoded -->

Denote the above inequality as event E model . Below we are going to condition on E model being hold. Note that here for notation simplicity, we denote

<!-- formula-not-decoded -->

remark Note that in the definition of V (Eq. 0.9), we have β ∈ [0 , B ] . And in the above formulation of β , note that B appears inside a log term. So we need to set B such that β ≤ B and we can get the correct B by solving the inequality β ≤ B for B .

Lemma 8.10 (Optimism) . Assume event E model is true. for all n and h ,

<!-- formula-not-decoded -->

Proof: We consider a fixed episode n . We prove via induction. Assume that ̂ V n h +1 ( s ) ≥ V glyph[star] h +1 ( s ) for all s . For time step h , we have:

<!-- formula-not-decoded -->

where in the last inequality we use the inductive hypothesis that ̂ V n h +1 ( s ) ≥ V glyph[star] h +1 ( s ) , and µ glyph[star] h φ ( s, a ) is a valid distribution (note that ̂ µ n h φ ( s, a ) is not necessarily a valid distribution). We need to show that the bonus is big enough to offset the model error φ ( s, a ) glyph[latticetop] ( ̂ µ n h -µ glyph[star] h ) glyph[latticetop] ̂ V n h +1 . Since we have event E model being true, we have that:

<!-- formula-not-decoded -->

as by the construction of V , we know that ̂ V n h +1 ∈ V .

This concludes the proof.

## 8.6.2 Regret Decomposition

Now we can upper bound the per-episode regret as follows:

<!-- formula-not-decoded -->

We can further bound the RHS of the above inequality using simulation lemma. Recall Eq. 0.4 that we derived in the note for tabular MDP (Chapter 7:

<!-- formula-not-decoded -->

(recall that the simulation lemma holds for any MDPs-it's not specialized to tabular).

In the event E model , we already know that for any s, a, h, n , we have ( ̂ P n h ( ·| s, a ) -P ( ·| s, a ) ) · ̂ V n h +1 glyph[lessorsimilar] β ‖ φ ( s, a ) ‖ (Λ n h ) -1 = b n h ( s, a ) . Hence, under E model , we have:

<!-- formula-not-decoded -->

Sum over all episodes, we have the following statement.

Lemma 8.11 (Regret Bound) . Assume the event E model holds. We have:

<!-- formula-not-decoded -->

## 8.6.3 Concluding the Final Regret Bound

We first consider the following elliptical potential argument, which is similar to what we have seen in the linear bandit lecture.

Lemma8.12 (Elliptical Potential) . Consider an arbitrary sequence of state action pairs s i h , a i h . Assume sup s,a ‖ φ ( s, a ) ‖ 2 ≤ 1 . Denote Λ n h = I + ∑ n -1 i =0 φ ( s i h , a i h ) φ ( s i h , a i h ) glyph[latticetop] . We have:

<!-- formula-not-decoded -->

Proof: By the Lemma 3.7 and 3.8 in the linear bandit lecture note,

<!-- formula-not-decoded -->

where the first inequality uses that for 0 ≤ y ≤ 1 , ln(1 + y ) ≥ y/ 2 .

Now we use Lemma 8.11 together with the above inequality to conclude the proof.

Proof: [Proof of main Theorem 8.9]

We split the expected regret based on the event E model .

<!-- formula-not-decoded -->

Note that:

<!-- formula-not-decoded -->

Recall that β = ˜ O ( Hd ) . This concludes the proof.

## 8.7 Bibliographic Remarks and Further Readings

There are number of ways to linearly parameterize an MDP such that it permits for efficient reinforcement learning (both statistically and computationally). The first observation that such assumptions lead to statistically efficient algorithms was due to [Jiang et al., 2017] due to that these models have low Bellman rank (as we shall see in Chapter 9). The first statistically and computationally efficient algorithm for a linearly parameterized MDP model was due to [Yang and Wang, 2019a,b]. Subsequently, [Jin et al., 2020] provided a computationally and statistically efficient algorithm for simplified version of this model, which is the model we consider here. The model of [Modi et al., 2020b, Jia et al., 2020, Ayoub et al., 2020b, Zhou et al., 2020] provides another linearly parameterized model, which can viewed as parameterizing P ( s ′ | s, a ) as a linear combination of feature functions φ ( s, a, s ′ ) . One notable aspect of the model we choose to present here, where P h ( ·| s, a ) = µ glyph[star] h φ ( s, a ) , is that this model has a number of free parameters that is |S| · d (note that µ is unknown and is of size |S| · d ), and yet the statistical complexity does not depend on |S| . Notably, this implies that accurate model estimation request O ( |S| ) samples, while the regret for reinforcement learning is only polynomial in d . The linearly parameterized models of [Modi et al., 2020b, Jia et al., 2020, Ayoub et al., 2020b, Zhou et al., 2020] are parameterized by O ( d ) parameters, and, while O ( d ) free parameters suggests lower model capacity (where accurate model based estimation requires only polynomial in d samples), these models are incomparable to the linearly parameterized models presented in this chapter;

It is worth observing that all of these models permit statistically efficient estimation due to that they have bounded Bellman rank [Jiang et al., 2017] (and bounded Witness rank [Sun et al., 2019a]), a point which we return to in the next Chapter.

The specific linear model we consider here was originally introduced by [Jin et al., 2020]. The non-parametric modelbased algorithm we study here was first introduced by [Lykouris et al., 2019] (but under the context of adversarial attacks).

The analysis we present here does not easily extend to infinite dimensional feature φ (e.g., RBF kernel); here, [Agarwal et al., 2020a] provide an algorithm and an analysis that extends to infinite dimensional φ , i.e. where we have a Reproducing Kernel Hilbert Space (RKHS) and the regret is based on the concept of Information Gain.

## Chapter 9

## Generalization with Bounded Bilinear Rank

Our previous lecture on linear bandits and linear MDPs gave examples of statistically efficient reinforcement learning, where we can obtain a sublinear regret with no dependence on the number of states or actions in the MDP. Obtaining sample size results which are independent of the size of the state space (and possibly the action space) is essentially a question of generalization , which is the focus of this chapter. In particular, we now present a more general underlying assumption which permit sample efficient reinforcement learning in various cases.

Recall our lower bounds from Chapter 5 which suggest that assumptions which would permit sample efficient RL may be subtle due to that: (i) agnostic RL is not possible and (ii) any assumption which permits sample efficient requires stronger, and possibly more subtle, conditions than simply assuming Q glyph[star] is a linear in some known features. This chapter introduces the Bellman rank condition along with a generalization, the Bilinear rank condition, which permit sample efficient RL.

This chapter follows the ideas in the original introduction of the Bellman rank [Jiang et al., 2017], along with a simplified analysis and a generalized concept, the Bilinear class, provided by [Du et al., 2021].

Setting and notation and setting. This chapter focuses on PAC RL (see Section 1.4), where our goal is be to find a near optimal policy. We work in the episodic setting where we can obtain trajectories under some fixed s 0 (see Section 1.4), and we work in the finite horizon setting. We use the notation a 0: h ∼ d π to denote the sampling of actions a 0 to a h from trajectories under π . We also overload notation where we write s h , a h ∼ d π h for the sampled state-action pair at timestep h under π , and s h ∼ d π h as the sampled state at time h under π .

## 9.1 Hypothesis Classes

We assume access to a hypothesis class H = H 0 × . . . × H H -1 , which can be abstract sets that permit for both model-based and value-based hypotheses. The only restriction we make is that for all f ∈ H , we have an associated state-action value function Q h,f and a value function V h,f . Furthermore, we assume the hypothesis class is constrained so that V h,f ( s ) = max a Q h,f ( s, a ) for all f ∈ H , h ∈ [ H ] , and s ∈ S , which is always possible as we can remove hypothesis for which this is not true. We let π h,f be the greedy policy with respect to Q h,f , i.e., π h,f ( s ) = argmax a ∈A Q h,f ( s, a ) , and π f as the sequence of time-dependent policies { π h,f } H -1 h =0 .

The following provide some examples of value based and model based hypothesis classes:

- An example of value-based hypothesis class H is an explicit set of state-action value Q and value functions V i.e.

<!-- formula-not-decoded -->

Note that in this case, for any hypothesis f := (( Q 0 , V 0 ) , ( Q 1 , V 1 ) , . . . , ( Q H -1 , V H -1 )) ∈ H , we can take the associated Q h,f = Q h and associated V h,f = V h .

- Another example of value-based hypothesis class H is when H is just a set of state-action value Q functions i.e.

<!-- formula-not-decoded -->

In this case, for any hypothesis f := ( Q 0 , Q 1 , . . . , Q H -1 ) ∈ H , we can take the associated Q h,f = Q h and the associated V h,f function to be greedy with respect to the Q h,f function i.e. V h,f ( · ) = max a ∈A Q h,f ( · , a ) .

- An example of model-based hypothesis class is when H h is a set of transition models and reward functions, i.e.

<!-- formula-not-decoded -->

In this case, for any hypothesis f := (( P 0 , r 0 ) , ( P 1 , r 1 ) , . . . , ( P H -1 , r H -1 )) ∈ H , we can take the associated Q h,f and V h,f functions to be the optimal value functions corresponding to the transition models { P h } H -1 h =0 and reward functions { r h } H -1 h =0 .

## 9.2 The Bellman Rank

We now make assumptions on the Bellman residual error. For the 'Q-version' of the Bellman rank, we assume that the Bellman residual error, Q -T Q , has a simple parametric form. In particular,

Definition 9.1 ( Q -Bellman Rank) . For a given MDP, a hypothesis class H has a Q -Bellman rank of dimension d if for all h ∈ [ H ] , there exist functions W h : H → R d and X h : H → R d with sup f ∈H ‖ W h ( f ) ‖ 2 ≤ B W and sup f ∈H ‖ X h ( f ) ‖ 2 ≤ B X , such that for all f, g ∈ H

<!-- formula-not-decoded -->

Importantly, the functions W h and X h need not be known to the learner.

Instead, of assuming this parametric assumption on the Bellman residual error of the state-action value, we consider a variant which is stated in terms of the values. Note that we can write the Bellman optimality equations (see Equation 0.7) in terms of the value functions as follows:

<!-- formula-not-decoded -->

This motivates the following definition:

Definition 9.2 ( V -Bellman Rank) . For a given MDP, a hypothesis class H has a V -Bellman rank of dimension d if for all h ∈ [ H ] , there exist functions W h : H → R d and X h : H → R d with sup f ∈H ‖ W h ( f ) ‖ 2 ≤ B W and sup f ∈H ‖ X h ( f ) ‖ 2 ≤ B X , such that for all f, g ∈ H :

<!-- formula-not-decoded -->

Note that the action a h is chosen by π g , and recall that for every g , there corresponds a Q h,g , where both V h,g and π g are the corresponding greedy value function and policy.

Let us remark on a few subtle differences in these two definitions, in terms their usage of functions V h,f vs Q h,f . Roughly speaking, the former definition corresponds to an assumption on the Bellman residual error of the stateaction value function, while the latter definition corresponds to an assumption on the Bellman residual error of the value functions. A more subtle distinction is that, for the Q -version, the expectation is with respect to a 0: h ∼ π f , while for the V -version, the expectation is with respect to a 0: h -1 ∼ π f and a h ∼ π g . The use of π g is subtly different than the expectation being taken directly on the Bellman residual error of V h,g due to that π g may not correspond to the true argmax in Equation 0.1. However, as we shall see, we can find estimates of both of the relevant quantities in the Q and the V versions.

Realizability. We say that H is realizable for an MDP M if, for all h ∈ [ H ] , there exists a hypothesis f glyph[star] ∈ H such that Q glyph[star] h ( s, a ) = Q h,f glyph[star] ( s, a ) , where Q glyph[star] h is the optimal state-action value at time step h in the ground truth MDP M . For instance, for the model-based perspective, the realizability assumption is implied if the ground truth transitions { P h } H -1 h =0 and reward function { r h } H -1 h =0 belong to our hypothesis class H .

## 9.3 Examples

## 9.3.1 Examples that have small Q -Bellman rank

## Contextual Linear Bandits

Let us start with the H = 1 case and return to the linear bandit model from Chapter 6. Let us consider a contextual model where there is a distribution over states s 0 ∼ µ and where we assume:

<!-- formula-not-decoded -->

Here, our hypothesis of state-action values is H = { θ glyph[latticetop] φ ( s, a ) : ‖ θ ‖ 2 ≤ W } .

Take two hypothesis f, g where we denote Q 0 ,f ( s, a ) := θ glyph[latticetop] φ ( s, a ) , Q 0 ,g ( s, a ) = w glyph[latticetop] φ ( s, a ) (we take h = 0 due to that H = 1 ). We have:

<!-- formula-not-decoded -->

Namely, we can set W 0 ( g ) = w , and X 0 ( f ) = E s ∼ µ,a = π f ( s ) [ φ ( s, a )] . Thus, we have the following:

Proposition 9.3. In the contextual linear bandit model, we have that H , as defined above, has a Q -Bellman rank of d .

## Linear Bellman Completion

Let us now consider the linear Bellman completion setting (Definition 3.1). We show that linear Bellman completion models have Q -Bellman rank equal to d . In this setting, since Q glyph[star] h ( s, a ) = ( θ glyph[star] h ) glyph[latticetop] φ ( s, a ) , ∀ s, a, h , we use linear functions for our hypothesis class, i.e., H h = { θ glyph[latticetop] φ ( s, a ) : ‖ θ ‖ 2 ≤ W, ‖ θ glyph[latticetop] φ ( · , · ) ‖ ∞ ≤ H } for some W ∈ R + where ‖ θ glyph[star] h ‖ 2 ≤ W ; as in Section 9.1, the set of values are those that are greedy with respect to these Q functions.

Recall from Definition 3.1, given θ , we use T h ( θ ) ∈ R d to represent the linear function resulting from the Bellman backup, i.e.,

<!-- formula-not-decoded -->

We further assume feature is normalized, i.e., sup s,a ‖ φ ( s, a ) ‖ 2 ≤ 1 , and sup h,θ ∈H h ‖T h ( θ ) ‖ 2 ≤ W . Note that this condition holds in linear MDP with W = 2 H √ d .

Take two hypothesis f, g where we denote Q h,f ( s, a ) := θ glyph[latticetop] h φ ( s, a ) , Q h,g ( s, a ) = w glyph[latticetop] h φ ( s, a ) , ∀ h . Then for any time step h ∈ [0 , H -2] , we have:

<!-- formula-not-decoded -->

Taking W h ( g ) := w h -T h ( w h +1 ) (and note that W h ( f glyph[star] ) = 0 due to the Bellman optimality condition) and X h ( f ) := E s h ,a h ∼ d π f h [ φ ( s h , a h )] , we have shown the following:

Proposition 9.4. Under the linear Bellman completion assumption, we have that H , as defined above, has a Q -Bellman rank of d . Further more, we have sup g ∈H ‖ W h ( g ) ‖ 2 ≤ B W := 2 W , and sup f ∈H ‖ X h ( f ) ‖ 2 ≤ B X := 1 .

There are a number of natural models which satisfy the Bellman completeness property, including contextual linear bandits, linear MDPs and Linear Quadratic Regulators (we will cover LQRs in Chapter 16).

## Linear Q glyph[star] and V glyph[star]

Let us now suppose that Q glyph[star] h and V glyph[star] h are functions which are in the span of some given features features φ : S × A ↦→ R d , ψ : S ↦→ R d , i.e. that, for all h ∈ [0 , . . . , H -1] , there exist θ glyph[star] h ∈ R d and w glyph[star] h ∈ R d , such that for all s, a , Q glyph[star] h ( s, a ) = ( θ glyph[star] h ) glyph[latticetop] φ ( s, a ) and V glyph[star] h ( s ) = ( w glyph[star] h ) glyph[latticetop] ψ ( s ) .

Our hypothesis class class H h is defined as H h = { ( θ, w ) : ‖ θ ‖ 2 ≤ W 1 , ‖ w ‖ 2 ≤ W 2 , ∀ s, max a θ glyph[latticetop] φ ( s, a ) = w glyph[latticetop] ψ ( s ) } . Take any f, g where we denote Q h,g ( s, a ) = θ glyph[latticetop] h φ ( s, a ) , V h,g ( s ) = w glyph[latticetop] h ψ ( s ) , we have that:

<!-- formula-not-decoded -->

Taking W h ( g ) := [ θ h w h ] , and X h ( f ) := [ φ ( s, a ) E s ′ ∼ P h ( ·| s,a ) [ ψ ( s ′ )] ] , we have shown that:

Proposition 9.5. Under the above linear Q glyph[star] and V glyph[star] assumption, our hypothesis class H , as defined above, has a Q -Bellman rank of 2 d .

The linear Q glyph[star] assumption vs. the linear Q glyph[star] /V glyph[star] assumption. As we saw in Chapter 5, simply assuming that Q glyph[star] is linear is not sufficient for sample efficient RL. Thus it may seem surprising that assuming both Q glyph[star] and V glyph[star] are linear (in different features) permits a sample efficient algorithm (as we shall see). However, it is worthwhile to observe how this is a substantially strong assumption. In particular, note that our hypothesis class must enforce the constraint that max a θ glyph[latticetop] φ ( s, a ) = w glyph[latticetop] ψ ( s ) for all ( θ, w ) ∈ H ; this eliminates functions from the set of all Q which are linear in φ . Furthermore, note that this pruning is done without collecting any data.

## Q glyph[star] -state abstraction

The Q glyph[star] -state abstraction mode is defined as follows. There exists some abstraction function ξ : S ↦→ Z where |Z| &lt; |S| , such that for all h , we have:

<!-- formula-not-decoded -->

We show that this example has low Q -bellman rank by showing that it is a special case of the linear Q glyph[star] &amp; V glyph[star] model. Specifically, let us define φ : S × A ↦→ R |Z||A| and ψ : S ↦→ R |Z| as follows:

<!-- formula-not-decoded -->

With this feature setup, we have:

<!-- formula-not-decoded -->

Thus, similar to linear Q glyph[star] /V glyph[star] model above, we can set H such that H h := { ( θ, w ) : ‖ θ ‖ 2 ≤ W 1 , ‖ w ‖ 2 ≤ W 2 , ∀ s, max s θ glyph[latticetop] φ ( s, a ) = w glyph[latticetop] ψ ( s ) } . we demonstrate that this model has Q -Bellman rank at most |Z||A| + |Z| .

Proposition 9.6. Under the above Q glyph[star] -state abstraction assumption, our hypothesis class H has a Q -Bellman rank of |Z||A| + |Z| .

## Low-occupancy measure

An MDP has low (state-action) occupancy measure if the following is true. For all h , there exist two functions β h : H ↦→ R d , and φ h : S × A ↦→ R d , such that for any f ∈ H , we have:

<!-- formula-not-decoded -->

We show that this model has Q -Bellman rank at most d . Consider any f, g and h , we have:

<!-- formula-not-decoded -->

which shows that W h ( g ) := ∑ s,a φ h ( s, a ) [ Q h,g ( s, a ) -r ( s, a ) -E s ′ ∼ P h ( ·| s,a ) [ V h +1 ,g ( s ′ )] ] (note that W h ( f glyph[star] ) = 0 due to the Bellman optimality condition), and X h ( f ) := β h ( f ) . Note that in this model we can use any hypothesis class H as long as it contains f glyph[star] .

Proposition 9.7. Any hypothesis class H , together with the MDP that has a low-occupancy measure, has Q -Bellman rank d .

Note that if the state distribution has low-occupancy, i.e.

<!-- formula-not-decoded -->

for some φ h : S ↦→ R d , then it satisfies the V -Bellman rank condition, with rank d . In this latter case, we refer to the model as having low (state) occupancy measure .

## 9.3.2 Examples that have small V -Bellman rank

## Contextual Bandit

Contextual bandit is a finite horizon MDP with H = 1 and finite number of actions. The contexts are states s 0 ∼ µ . Here our hypothesis class H = H 0 = { f : S × A ↦→ [0 , 1] } with r ∈ H . We show that the V -Bellman rank is at most A . We use the notation V 1 ( s ) = 0 , ∀ s (recall that H = 1 again). Consider any function pair f, g ∈ H , note that since H = 1 , the state distribution at h = 0 is independent of the policy π f . Thus we have:

<!-- formula-not-decoded -->

This means that we can write the Bellman error of g averaged over µ as an inner product between two vectors - an all one vector, and a vector whose a -th entry is E s ∼ µ [ 1 { π g ( s ) = a } [ Q 0 ,g ( s, a ) -r ( s, a )]] , i.e., we have X 0 ( f ) = 1 A where 1 A represents a A-dim vector will all one , and W 0 ( g ) ∈ R A where the a -th entry of W 0 ( g ) is E s ∼ ν 1 { π g ( s ) = a } [ Q 0 ,g ( s, a ) -r ( s, a )] . Note that W 0 ( f glyph[star] ) = 0 where f glyph[star] is the ground truth reward function here.

Proposition 9.8. The hypothesis class H and the contextual bandit model together has a V -Bellman rank A .

## Feature Selection in Low-rank MDPs (Representation Learning)

We consider low-rank MDP where for all h , there exist µ glyph[star] : S ↦→ R d , φ glyph[star] : S × A ↦→ R d , such that for all s, s, a ′ , we have P h ( s ′ | s, a ) = µ glyph[star] ( s ′ ) glyph[latticetop] φ glyph[star] ( s, a ) , r ( s, a ) = ( θ glyph[star] ) glyph[latticetop] φ glyph[star] ( s, a ) . We denote Φ ⊂ S × A ↦→ R d as our feature class such that φ glyph[star] ∈ Φ . We assume µ glyph[star] , φ glyph[star] , θ glyph[star] are normalized, i.e., ‖ φ glyph[star] ( s, a ) ‖ 2 ≤ 1 , ∀ s, a, h , and ‖ v glyph[latticetop] µ glyph[star] ‖ 2 ≤ √ d , for all v such that ‖ v ‖ ∞ ≤ 1 , ‖ θ glyph[star] ‖ 2 ≤ √ d .

We denote the hypothesis class H h := { w glyph[latticetop] φ ( s, a ) : ‖ w ‖ 2 ≤ W,φ ∈ Φ } where we can set W = 4 H √ d since we can verify that for Q glyph[star] h ( s, a ) , it is linear with respect to φ glyph[star] , i.e., Q glyph[star] h ( s, a ) = ( w glyph[star] h ) glyph[latticetop] φ ( s, a ) for some w glyph[star] h with ‖ w glyph[star] h ‖ 2 ≤ 2 H √ d . Consider a pair of hypothesis ( f, g ) where we denote Q h,g := w glyph[latticetop] h φ , and recall V h,g ( · ) = max a Q h,g ( · , a ) , and π h,g ( s ) := argmax a Q h,g ( s, a ) .

<!-- formula-not-decoded -->

Taking W h ( g ) := ∫ s µ glyph[star] ( s ) [ V h,g ( s ) -r ( s, π h,g ( s )) -E s ′ ∼ P h ( ·| s,π h,g ( s )) [ V h +1 ,g ( s ′ )] ] (note that W h ( f glyph[star] ) = 0 due to the Bellman optimality condition) and X h ( f ) := E ˜ s, ˜ a ∼ d π f h -1 [ φ glyph[star] (˜ s, ˜ a )] , we have shown that:

Proposition 9.9. The hypothesis class H and the low-rank MDP defined above together has V -Bellman rank d . Moreover, we have ‖ X h ( f ) ‖ 2 ≤ B X = 1 , and ‖ W h ( g ) ‖ 2 ≤ B W = 2 H √ d .

Block MDPs. The Block MDP has a small size discrete latent state space Z , an emission distribution v ( · ; z ) ∈ ∆( S ) , ∀ z , a decoder ξ : S ↦→ Z which maps a state s to its corresponding latent state z , and a transition in latent state space T h ( z ′ | z, a ) , ∀ h . To show that Low-rank MDP model captures it, we define φ glyph[star] ( s, a ) ∈ { 0 , 1 } |Z||A| , where φ glyph[star] ( s, a ) is a one hot encoding vector which contains zero everywhere, except one in the entry corresponding to the latent state and action pair ( ξ ( s ) , a ) . The corresponding µ glyph[star] ( s ′ ) ∈ R |Z||A| is defined such that its entry corresponding to ( z, a ) is equal to ∑ z ′ T ( z ′ | z, a ) v ( s ′ ; z ′ ) . This shows that the Block MDP model is a special case of the Low-rank MDP model where the rank is the number of unique latent states.

Feature Selection in Sparse Linear MDP. A sparse linear MDP is a linear MDP where φ glyph[star] is known (but µ glyph[star] is unknown), and there is a subset K glyph[star] ∈ [ d ] with |K glyph[star] | = s and s &lt; d , such that P h ( s ′ | s, a ) = ∑ i ∈K glyph[star] µ glyph[star] i ( s ′ ) φ glyph[star] i ( s, a ) , where weuse x i to denote the i -th element of the vector x . Wecanre-write the transition as P h ( s ′ | s, a ) = µ glyph[star] K glyph[star] ( s ′ ) glyph[latticetop] φ glyph[star] K glyph[star] ( s, a ) , where we denote x K as the subvector of x that is indexed by elements in K . Thus, this shows that a sparse linear MDP is a special case of low-rank MDP where the rank is s instead of d

We consider the following hypothesis class. Define a representation class Φ with | Φ | = ( d s ) . Each φ ∈ Φ corresponds to a s -size subset K ∈ [ d ] . Given K , we have φ ( s, a ) ∈ R s , and φ ( s, a ) = φ glyph[star] K glyph[star] ( s, a ) , ∀ s, a . By the assumption that ‖ φ glyph[star] ( s, a ) ‖ 2 ≤ 1 , we must have ‖ φ ( s, a ) ‖ 2 ≤ 1 for all s, a, φ ∈ Φ . We define H h = { w glyph[latticetop] φ : ‖ w ‖ 2 ≤ W,w ∈ R s , φ ∈ Φ h , | w glyph[latticetop] φ ( · , · ) | ≤ H } . We set W = 2 H √ d such that Q glyph[star] h ∈ H h (note that Q glyph[star] h is a linear function with respect to φ glyph[star] K glyph[star] due to the fact that the MDP is a linear MDP with respect to feature φ glyph[star] K glyph[star] ).

## 9.4 Bilinear Classes

The Bilinear class model generalizes both the Q -Bellman rank and V -Bellman rank. We define the Bilinear class model as follows.

Definition 9.10 (Bilinear Class) . Consider an MDP and a hypothesis class H , a discrepancy function glyph[lscript] f : S × A × S × [ H ] ×H (defined for each f ∈ H ), and a set of (non-stationary) estimation policy Π est = { π est ( f ) : f ∈ H} . We say that {H , glyph[lscript] f , Π est , M} is a Bilinear class of rank d if H is realizable in M , and if there exists W h : H ↦→ R d and X h : H ↦→ R d , such that the following two properties hold for all f ∈ H and h ∈ [ H ] :

## 1. We have

<!-- formula-not-decoded -->

2. The policy π est ( f ) and the discrepancy glyph[lscript] f can be used for estimation in the following sense: for any g ∈ H , we have:

<!-- formula-not-decoded -->

We suppress the h dependence on π est,h when clear from context. Typically π est ( f ) will be either the uniform distribution over A or π f itself. In the latter case, we refer to the estimation strategy as being on-policy.

The first condition in the above definition assumes that the average Bellman error of f under π f 's state-action distribution can be upper bounded by the value from the bilinear formulation. The second condition assumes that we have a discrepancy function that allows us to evaluate the value of the bilinear formulation. Note that the second condition permits data reuse, i.e., given a set of samples where s ∼ d π f h , a ∼ π est,h ( s ; f ) , s ′ ∼ P h ( ·| s, a ) , we can use the dataset to evaluate the loss for all g ∈ H simultaneously. This data reuse is the key to generalization. Also the second condition ensures that for all f , we must have E s ∼ d π f h ,a ∼ π est,h ( s ; f ) ,s ′ ∼ P h ( ·| s,a ) [ glyph[lscript] f ( s, a, s ′ , h, f glyph[star] )] = 0 .

## 9.4.1 Examples

Let now see how the Bilinear classes naturally generalize the Bellman rank.

## Q-Bellman rank and V-Bellman rank

It is straightforward to see that the Bilinear class captures the Q -Bellman rank and V -Bellman rank models. To capture Q -Bellman rank, in Bilinear class, we will set glyph[lscript] f := glyph[lscript] for all f , with glyph[lscript] being defined in Eq. 0.6, and set

π est,h ( f ) = π f,h , ∀ f . To capture V -Bellman rank, we will set glyph[lscript] f := glyph[lscript] for all f with glyph[lscript] being defined in Eq. 0.7, and π est ( f ) := Unif A , i.e., uniform distribution over A . More precisely, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the V -Bellman rank case, the importance weighting factor 1 { a = π h,g ( s ) } / (1 /A ) ensures that given any s, g ,

<!-- formula-not-decoded -->

i.e., glyph[lscript] ( s, a, s ′ , g ) is an unbiased estimate of the V-Bellman error of g at state s , given that a ∼ Unif A , s ′ ∼ P h ( ·| s, a ) .

Q -Bellman and V -Bellman rank already assume the existence of two mapps.

Note that by our assumption that ‖ Q h,g ‖ ∞ ≤ H , we have ‖ glyph[lscript] ‖ ∞ ≤ H for the Q-Bellman rank case, and ‖ glyph[lscript] ‖ ∞ ≤ AH for the V -Bellman rank case.

## Linear mixture model

The additional f -dependent discrepancy function allows Bilinear classes to capture the following linear mixture model , which cannot be captured by Q or V version of the Bellman rank.

A linear mixture MDP model has a feature φ : S × A × S ↦→ R d , such that P h ( s ′ | s, a ) = ( θ glyph[star] h ) glyph[latticetop] φ ( s, a, s ′ ) where we assume ‖ θ glyph[star] h ‖ 2 ≤ W . We assume reward r is known.

Here, our hypothesis class H is a model based one, where H h = { ( P h , r ) : P h ( s ′ | s, a ) := θ glyph[latticetop] φ ( s, a, s ′ ) , ‖ θ ‖ 2 ≤ W } . We assume that the reward function is known, and the associated Q and V function with any hypothesis is the optimal state-action value and optimal value for the associated model (as discussed in Section 9.1).

We design the discrepancy function glyph[lscript] f as follows. Denote g ∈ H as g = { θ glyph[latticetop] h ψ ( · , · , · ) } H -1 h =0 ,

<!-- formula-not-decoded -->

Below we show that it is a Bilinear class model. For any f, g ∈ H , set π est ( f ) = π f (i.e., on-policy), we have:

<!-- formula-not-decoded -->

Thus we have shown that W h ( g ) = θ h , and X h ( f ) = E s,a ∼ d π f h [ ∑ s ′ φ ( s, a, s ′ )] .

Proposition 9.11. The hypothesis class H together with the linear mixture model is a Bilinear class.

## 9.5 PAC-RL with Bounded Bilinear Rank

We now focus on the sample complexity of finding an glyph[epsilon1] -near optimal policy. We work in the episodic setting where we can obtain trajectories under some fixed initial state s 0 (i.e., µ := δ ( s 0 ) ). Note that generalizing to any distribution µ is straightforward.

## 9.5.1 Algorithm

## Algorithm 7 BLin-UCB

- 1: Input: number of iteration T , batch size m , confidence radius R , estimation policy π est , loss glyph[lscript]
- 2: for t = 0 . . . T -1 do
- 3: Set f t as the solution of the following program:

<!-- formula-not-decoded -->

- 4: ∀ h , create batch data D t,h = { s i , a i , ( s ′ ) i } i m =1 , where s i ∼ d π f t h , a i ∼ π est,h ( s i ; f t ) , ( s i ) ′ ∼ P h ( ·| s i , a i ) (see text for discussion).
- 5: end for
- 6: return arg max π ∈{ π f 0 ,...,π f T -1 } V π .

Wepresent the Bilinear-UCB algorithm in Algorithm 7. Algorithm 7 uses the estimation policy π est,h ( · ; f t ) to generate action a t when collecting the batch D t,h at iteration t for time step h .

Note that the samples required by Bilinear-UCB algorithm can be obtained in episodic sampling model. We simply follow π until time step h , and then use π est ( · ; f t ) to sample the action at the timestep.

While the algorithm returns the best policy found up to time T , i.e. max t ∈ [ T ] V π f t , it is also straight forward to replace this step using a sample based estimate of the best policy seen so far (with a negligible increase in the sample size).

Intuitively, BLin-UCB uses optimism for exploration. Our constraints are designed such that they eliminate functions that are not f glyph[star] . Also the threshold R will be set such that f glyph[star] is always a feasible solution with high probability. Thus, the argmax procedure in the objective function ensures that it returns an optimistic estimate, i.e., V 0 ,f t ( s 0 ) ≥ V glyph[star] 0 ( s 0 ) .

## 9.5.2 Sample Complexity

We begin by stating the following assumption about the uniform convergence property of our hypothesis class H .

Assumption 9.12 (Uniform Convergence) . We assume that there exists a function ε gen ( m, H , δ ) such that for any distribution ν ∈ ∆( S ×A×S ) and f ∈ H , and any δ ∈ (0 , 1) , with probability at least 1 -δ over the m i.i.d samples D := { s i , a i , s ′ i } i m =1 ∼ ν , we have:

<!-- formula-not-decoded -->

The above assumption is the standard uniform convergence result and typically we have ε gen ( m, H , δ ) → 0 , m →∞ . Below we give three examples, one for finite hypothesis class H , one for infinite hypothesis class under the linear Bellman completion setting, and one for the infinite hypothesis class under the problem of feature selection in sparse linear MDPs.

Example 9.13 ( Finite hypothesis class case) . Let us consider the example where H is a discrete hypothesis class. In this case, for Q -Bellman rank loss, via Hoeffding's inequality and a union bound over H , we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

which means we can set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For V -Bellman rank loss, due to the importance weighting, we have an additional A term, i.e., with probability at least 1 -δ ,

Example 9.14 (Linear Bellman completion) . In the linear Bellman completion model, our hypothesis class H h = { w glyph[latticetop] φ ( s, a ) : ‖ w ‖ 2 ≤ W, ‖ w glyph[latticetop] φ ( · , · ) ‖ ∞ ≤ H } is not discrete. However, with the loss function associated with Q -Bellman rank, via a standard uniform convergence analysis using glyph[epsilon1] -net for linear functions in H , we can show (details in Lemma A.12):

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Example 9.15 (Feature selection for sparse linear MDP) . To give another example of infinite hypothesis class case, we consider the following H with H h = { w glyph[latticetop] φ : ‖ w ‖ 2 ≤ 2 H √ d, w ∈ R s , φ ∈ Φ h , ‖ w glyph[latticetop] φ ‖ ∞ ≤ H } from the example of feature selection in sparse linear MDP. Recall Φ is discrete and | Φ | = ( d s ) . We focus on the loss function associated with V -Bellman rank. Denote H h,φ = { w glyph[latticetop] φ : ‖ w ‖ 2 ≤ 2 H √ d, w ∈ R s , ‖ w glyph[latticetop] φ ‖ ∞ ≤ H } for a φ ∈ Φ . Via a standard uniform convergence analysis using glyph[epsilon1] -net for linear functions in H h,φ , we can show (details in Lemma A.12):

<!-- formula-not-decoded -->

Applying a union bound over all φ ∈ Φ and use the fact that | Φ | ≤ s d , we have:

<!-- formula-not-decoded -->

with probability at least 1 -δ . Thus we have

<!-- formula-not-decoded -->

Note the polynomial dependence on s and the poly-log dependence on d .

With the above assumption, now we are ready to state the main theorem.

Theorem 9.16 (PAC RL for BLin-UCB) . Fix δ ∈ (0 , 1) , batch size m ∈ N + . Suppose our hypothesis class H forms a Bilinear class of rank d ; that the class is realizable; and that the Bilinear-UCB algorithm has access to both H and the associated discrepancy function glyph[lscript] . Set the parameters as follows:

<!-- formula-not-decoded -->

Let π be the policy returned by the Bilinear-UCB algorithm (Algorithm 7). With probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Note the total number of samples obtained is mTH .

Before proving the main theorem, we revisit the three examples above, i.e., finite hypothesis case, linear Bellman completion, and feature selection in sparse linear MDP, and derive the sample complexity bounds for these examples.

Finite hypothesis class case (Example 9.13 continued). Plugging in specific forms of ε gen ( m, H , δ/ ( TH )) in the above general theorem allows us to instantiate PAC bounds. Consider the Q -Bellman rank example with a discrete hypothesis class H . In this case, as we show above, we have ε gen ( m, H , δ ) := 2 H √ ln( |H| /δ ) /m . To have V glyph[star] 0 ( s 0 ) -max i ∈ [0 ,...,T -1] V π f i 0 ( s 0 ) ≤ glyph[epsilon1] for some glyph[epsilon1] ∈ (0 , H ) , we just need to set m such that:

<!-- formula-not-decoded -->

Using the inequality from Equation 0.9, it is suffice to set m as:

<!-- formula-not-decoded -->

and c is an absolute constant. Using the fact that T = ⌈ 2 Hd ln ( 4 Hd ( B 2 X B 2 W ε 2 gen ( m, H ,δ/ ( TH )) +1 ))⌉ , we can reach the following corollary.

Corollary 9.17 (PAC bound for finite hypothesis H under Q -Bellman rank) . Fix glyph[epsilon1], δ ∈ (0 , 1) . With probability at least 1 -δ , BLin-UCB learns a policy π such that V glyph[star] 0 ( s 0 ) -V π 0 ( s 0 ) ≤ glyph[epsilon1] , using at most

<!-- formula-not-decoded -->

many trajectories.

Linear Bellman completion (Example 9.14, continued). Plugging in the specific form of ε gen ( m, H , δ/ ( TH )) and B X = 1 and B W = W , with some linear algebra, we get:

<!-- formula-not-decoded -->

for some absolute constant c . Thus, to have V glyph[star] 0 ( s 0 ) -max i ∈ [0 ,...,T -1] V π f i 0 ( s 0 ) ≤ glyph[epsilon1] for some glyph[epsilon1] ∈ (0 , H ) , it is suffice to set m such that:

<!-- formula-not-decoded -->

Using the fact that for any positive scalars a, b , m = 9 a ln 2 (9 ab ) is a solution to the inequality m ≥ a ln 2 ( bm ) , we can set m as:

<!-- formula-not-decoded -->

Also plugging in ε gen ( m, H , δ/ ( HT )) into T , we get:

<!-- formula-not-decoded -->

Thus, the total number of trajectories used can be upper bounded as mT . This leads to the following corollary.

Corollary 9.18 (PAC bound for linear Bellman completion) . Fix glyph[epsilon1], δ ∈ (0 , 1) . With probability at least 1 -δ , for the linear Bellman completion model, BLin-UCB finds a policy π such that V glyph[star] 0 ( s 0 ) -V π 0 ( s 0 ) ≤ glyph[epsilon1] , using at most

<!-- formula-not-decoded -->

where ˜ c is an absolute constant, and ν ′ only contains log terms:

<!-- formula-not-decoded -->

Feature Selection in Sparse linear MDP (Example 9.15, continued). Plugging in the specific form of ε gen ( m, H , δ/ ( TH )) and the fact that B X = 1 and B W = 2 H √ d , with some linear algebra, we get:

<!-- formula-not-decoded -->

for some absolute constant c . Thus, to have V glyph[star] 0 ( s 0 ) -max i ∈ [0 ,...,T -1] V π f i 0 ( s 0 ) ≤ glyph[epsilon1] for some glyph[epsilon1] ∈ (0 , H ) , it is suffice to set m such that:

<!-- formula-not-decoded -->

Using the fact that for any positive scalars a, b , m = 9 a ln 2 (9 ab ) is a solution to the inequality m ≥ a ln 2 ( bm ) , we can set m as:

<!-- formula-not-decoded -->

Also plugging in ε gen ( m, H , δ/ ( HT )) into T , we get:

<!-- formula-not-decoded -->

Note the total number of trajectories used here is mT . This leads to the following corollary.

Corollary 9.19 (PAC bound for feature selection in sparse linear MDP) . Fix δ, glyph[epsilon1] ∈ (0 , 1) . With probability at least 1 -δ , for the sparse linear MDP model, BLin-UCB finds a policy π such that V glyph[star] 0 ( s 0 ) -V π 0 ( s 0 ) ≤ glyph[epsilon1] , using total number of trajectories at most

<!-- formula-not-decoded -->

with ˜ c being an absolute constant, and ν ′ only containing log terms:

<!-- formula-not-decoded -->

Note the polynomial dependence on s instead of d ( d only appears inside log-terms) in the above PAC bound, which indicates that BLin-UCB leverages the sparsity.

## 9.5.3 Analysis

For notational convenience, let us overload notation and write ε gen := ε gen ( m, H , δ/T H ) .

Lemma 9.20. For all h ∈ [0 , . . . , H -1] and t ∈ [0 , . . . , T -1] , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Proof: Note that the ( s, a, s ′ ) triple from D i,h is sampled as s ∼ d π f t h , a ∼ π est ( s ) , s ′ ∼ P h ( ·| s, a ) , in an i.i.d fashion. Thus, using assumption 9.12 and a union bound over all h and t concludes the proof.

Our next lemma is with regards to feasibility and optimism; we show that with high probability, f glyph[star] is always a feasible solution of our constrained optimization program, which implies the optimism property holds.

Lemma 9.21 (Optimism and Feasibility of f glyph[star] ) . Assume the event in Lemma 9.20 holds and that R := √ Tε gen . We have that for all t ∈ [ T ] , f glyph[star] is a feasible solution of the constrained program in Alg. 7. Furthermore, we have V 0 ,f t ( s 0 ) ≥ V glyph[star] 0 ( s 0 ) , for all t ∈ [ T ] .

Proof: Based on the second condition in the Bilinear class (Eq. 0.5), we have that for f glyph[star] :

<!-- formula-not-decoded -->

This and our assumption that the event in Lemma 9.20 holds imply that:

<!-- formula-not-decoded -->

This implies ∑ t -1 i =0 ( E D i,h glyph[lscript] f i ( s, a, s ′ , f glyph[star] ) ) 2 ≤ Tε 2 gen , which completes the proof of the first claim.

For the second claim, the feasibility of f glyph[star] and the fact that f t is the maximizer of the constrained program immediately implies the lemma.

Now we are ready to bound the per-episode regret.

Lemma 9.22. Assume the event in Lemma 9.20 holds. For all t ∈ [0 , . . . , T -1] , we have:

<!-- formula-not-decoded -->

Proof: We upper bound the regret as follows. Using optimism, we have

<!-- formula-not-decoded -->

where the first equality uses that a 0 = π 0 ,f t ( s 0 ) , and the fact that π 0 ,f t ( s ) = argmax a Q 0 ,f t ( s, a ) , the second equality uses telescoping with the definition Q H,f 0 ( s, a ) = 0 for all s, a , and the last equality uses the first condition in Bilinear class (Eq. 0.4)

We are now ready to prove the main theorem. The proof uses the same elliptical potential argument from Chapter 6.

Proof: [Proof of Theorem 9.16] Define Σ t,h := λI + ∑ t -1 τ =0 X h ( f τ ) X glyph[latticetop] h ( f τ ) (we will set λ later). First let us show that there exists an iteration t , such that for all h ∈ [ H ] ,

<!-- formula-not-decoded -->

We now prove the first claim above. Using Lemma 6.11 (from our linear bandit analysis),

<!-- formula-not-decoded -->

which holds for all h . After multiplying by 1 /T and summing over h , leads to:

<!-- formula-not-decoded -->

Bythe intermediate value theorem, there exists a t such that ∑ H -1 h =0 ln ( 1 + ‖ X h ( f t ) ‖ 2 Σ -1 t,h ) ≤ Hd T ln ( 1 + TB 2 X / ( dλ ) ) . This further implies that for any h in this iteration t , we have ln ( 1 + ‖ X h ( f t ) ‖ 2 Σ -1 t,h ) ≤ Hd T ln ( 1 + TB 2 X / ( dλ ) ) . Rearranging terms proves the first claim.

Now let us prove our claim on the second term. Using the definition of π est ( s ) and that f t is a feasible solution, we have that

<!-- formula-not-decoded -->

Using this and the second condition in the definition of Bilinear class (Eq. 0.5), we have:

<!-- formula-not-decoded -->

which proves our claimed bound on the second term.

Now consider the iteration t where Eqs. 0.8 hold. Using Lemma 9.22, we have:

<!-- formula-not-decoded -->

Now let us set 1 /λ = B 2 W /ε 2 gen +1 /B 2 X . Using this setting of λ and that T = ⌈ 2 Hd ln ( 4 Hd ( B 2 X B 2 W ε 2 gen +1 ))⌉ ,

<!-- formula-not-decoded -->

where the last step follows since we can show that, for any positive a, b ,

<!-- formula-not-decoded -->

This implies that:

<!-- formula-not-decoded -->

where the second inequality uses that λ ≤ ε 2 gen /B 2 W , due to our choice of λ . This concludes the proof.

## 9.6 The Eluder Dimension

To be added...

## 9.7 Bibliographic Remarks and Further Readings

The Bellman rank was originally proposed by [Jiang et al., 2017]; this original version was the V -version presented here. Both [Du et al., 2021, Jin et al., 2021] noted that there are two natural versions of the Bellman rank, which we refer to as the the Q and V versions. While we focus on PAC RL (i.e. finding a good policy), the work in [Dong et al., 2020] showed how to a obtain O ( √ T ) regret bound for the case of the V -Bellman rank assumption; this algorithm utilized a careful sampling procedure (along with an 'elimination-based' algorithm); the challenge here is that, for the V version, we do not have an 'on-policy' estimation procedure, unlike for the Q version. With regards to regret bounds for the Q -version, a more direct algorithm is possible which obtains a O ( √ T ) regret; this regret bound for the Q -version was provided by [Jin et al., 2021], which essentially follows along similar lines as the original argument provided in [Zanette et al., 2020] ([Zanette et al., 2020] provided a O ( √ T ) regret for the linear Bellman completion case).

The bilinear class approach was proposed in [Du et al., 2021]. The algorithm and proof is inspired by [Jiang et al., 2017], though it provides a more simplified analysis to obtain PAC RL bounds, using a more direct elliptical function argument (which this chapter follows).

Another generalization is provided by the Eluder dimension. The Eluder dimension was first proposed in [Russo and Van Roy, 2013] in a bandit setting. It was extended to RL in [Osband and Roy, 2014, Wang et al., 2020b, Ayoub et al., 2020a, Jin et al., 2021]. The Bellman eluder dimension and bounded bilinear rank are two different concepts. We focus on the latter one here due to that it captures nearly all the now standard parametric models which permit efficient RL, and it is relatively straightforward to verify. The Bellman eluder dimension can be thought of as specifying a more relaxed condition for when the algorithm terminates. Du et al. [2021] also shows that the Bilinear class can also capture Witness rank [Sun et al., 2019a] thus capturing examples such as Factored MDPs [Kearns and Koller, 1999], where Bellman rank and Bellman Eluder dimension [Jin et al., 2021] could be exponentially larger than the bilinear rank.

We also add a few remarks on further readings for important special cases. The problem of feature selection in lowrank MDP was first studied in [Jiang et al., 2017], and [Agarwal et al., 2020b, Uehara et al., 2021] later introduced two oracle-efficient model-based algorithms, and [Modi et al., 2021] introduces a model-free algorithm. The sparse linear MDP model was proposed in [Hao et al., 2021] though the algorithm from [Hao et al., 2021] does not perform strategic exploration. The linear mixture model was introduced and studied in [Modi et al., 2020a, Ayoub et al., 2020a,

Zhou et al., 2021b,a]. The linear Q glyph[star] and V glyph[star] model and the low-occupancy model were first introduced by Du et al. [2021].

## Chapter 10

## Deterministic MDPs with Linearly Parameterized Q glyph[star]

To be added

## Part 3

## Policy Optimization

## Chapter 11

## Policy Gradient Methods and Non-Convex Optimization

For a distribution ρ over states, define:

We drop the MDP subscript in this chapter.

One immediate issue is that if the policy class { π θ } consists of deterministic policies then π θ will, in general, not be differentiable. This motivates us to consider policy classes that are stochastic, which permit differentiability.

Example 11.1 (Softmax policies) . It is instructive to explicitly consider a 'tabular' policy representation, given by the softmax policy :

<!-- formula-not-decoded -->

where the parameter space is Θ = R |S||A| . Note that (the closure of) the set of softmax policies contains all stationary and deterministic policies.

Example 11.2 (Log-linear policies) . For any state, action pair s, a , suppose we have a feature mapping φ s,a ∈ R d . Let us consider the policy class

<!-- formula-not-decoded -->

with θ ∈ R d .

Example 11.3 (Neural softmax policies) . Here we may be interested in working with the policy class

<!-- formula-not-decoded -->

where the scalar function f θ ( s, a ) may be parameterized by a neural network, with θ ∈ R d .

<!-- formula-not-decoded -->

where we slightly overload notation. Consider a class of parametric policies { π θ | θ ∈ Θ ⊂ R d } . The optimization problem of interest is:

<!-- formula-not-decoded -->

## 11.1 Policy Gradient Expressions and the Likelihood Ratio Method

Let τ denote a trajectory, whose unconditional distribution Pr π µ ( τ ) under π with starting distribution µ , is

<!-- formula-not-decoded -->

We drop the µ subscript when it is clear from context.

It is convenient to define the discounted total reward of a trajectory as:

<!-- formula-not-decoded -->

where s t , a t are the state-action pairs in τ . Observe that:

<!-- formula-not-decoded -->

Theorem 11.4. (Policy gradients) The following are expressions for ∇ θ V π θ ( µ ) :

- REINFORCE:
- Action value expression:

<!-- formula-not-decoded -->

- Advantage expression:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The alternative expressions are more helpful to use when we turn to Monte Carlo estimation.

Proof: We have:

<!-- formula-not-decoded -->

which completes the proof of the first claim.

For the second claim, for any state s 0

<!-- formula-not-decoded -->

By linearity of expectation,

<!-- formula-not-decoded -->

where the last step follows from recursion. This completes the proof of the second claim.

The proof of the final claim is left as an exercise to the reader.

## 11.2 (Non-convex) Optimization

It is worth explicitly noting that V π θ ( s ) is non-concave in θ for the softmax parameterizations, so the standard tools of convex optimization are not applicable.

Lemma 11.5. (Non-convexity) There is an MDP M (described in Figure 0.1) such that the optimization problem V π θ ( s ) is not concave for both the direct and softmax parameterizations.

Proof: Recall the MDP in Figure 0.1. Note that since actions in terminal states s 3 , s 4 and s 5 do not change the expected reward, we only consider actions in states s 1 and s 2 . Let the 'up/above' action as a 1 and 'right' action as a 2 . Note that

<!-- formula-not-decoded -->

Now consider

<!-- formula-not-decoded -->

where θ is written as a tuple ( θ a 1 ,s 1 , θ a 2 ,s 1 , θ a 1 ,s 2 , θ a 2 ,s 2 ) . Then, for the softmax parameterization, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

This gives

Figure 0.1: (Non-concavity example) A deterministic MDP corresponding to Lemma 11.5 where V π θ ( s ) is not concave. Numbers on arrows represent the rewards for each action.

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which shows that V π is non-concave.

## 11.2.1 Gradient ascent and convergence to stationary points

Let us say a function f : R d → R is β -smooth if

<!-- formula-not-decoded -->

where the norm ‖ · ‖ is the Euclidean norm. In other words, the derivatives of f do not change too quickly.

Gradient ascent, with a fixed stepsize η , follows the update rule:

<!-- formula-not-decoded -->

It is convenient to use the shorthand notation:

<!-- formula-not-decoded -->

The next lemma is standard in non-convex optimization.

Lemma 11.6. (Convergence to Stationary Points) Assume that for all θ ∈ Θ , V π θ is β -smooth and bounded below by V ∗ . Suppose we use the constant stepsize η = 1 /β . For all T , we have that

<!-- formula-not-decoded -->

## 11.2.2 Monte Carlo estimation and stochastic gradient ascent

One difficulty is that even if we know the MDP M , computing the gradient may be computationally intensive. It turns out that we can obtain unbiased estimates of π with only simulation based access to our model, i.e. assuming we can obtain sampled trajectories τ ∼ Pr π θ µ .

With respect to a trajectory τ , define:

<!-- formula-not-decoded -->

We now show this provides an unbiased estimated of the gradient:

Lemma 11.7. (Unbiased gradient estimate) We have :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from the tower property of the conditional expectations and ( b ) follows from that the Markov property implies E [ ̂ Q π θ ( s t , a t ) | s t , a t ] = Q π θ ( s t , a t ) .

Hence, the following procedure is a stochastic gradient ascent algorithm:

1. initialize θ 0 .
2. For t = 0 , 1 , . . .
3. (a) Sample τ ∼ Pr π θ µ .
4. (b) Update:

Proof: Observe:

<!-- formula-not-decoded -->

where η t is the stepsize and ̂ ∇ V π θ ( µ ) estimated with τ .

Note here that we are ignoring that τ is an infinte length sequence. It can be truncated appropriately so as to control the bias.

The following is standard result with regards to non-convex optimization. Again, with reasonably bounded variance, we will obtain a point θ t with small gradient norm.

Lemma11.8. (Stochastic Convergence to Stationary Points) Assume that for all θ ∈ Θ , V π θ is β -smooth and bounded below by V ∗ . Suppose the variance is bounded as follows:

<!-- formula-not-decoded -->

For t ≤ β ( V ∗ ( µ ) -V (0) ( µ )) /σ 2 , suppose we use a constant stepsize of η t = 1 /β , and thereafter, we use η t = √ 2 / ( βT ) . For all T , we have:

<!-- formula-not-decoded -->

## Baselines and stochastic gradient ascent

A significant practical issue is that the variance σ 2 is often large in practice. Here, a form of variance reduction is often critical in practice. A common method is as follows.

Let f : S → R .

1. Construct f as an estimate of V π θ ( µ ) . This can be done using any previous data.
2. Sample a new trajectory τ , and define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We often refer to f ( s ) as a baseline at state s .

Lemma 11.9. (Unbiased gradient estimate with Variance Reduction) For any procedure used to construct to the baseline function f : S → R , if the samples used to construct f are independent of the trajectory τ , where ̂ Q π θ ( s t , a t ) is constructed using τ , then:

<!-- formula-not-decoded -->

where the expectation is with respect to both the random trajectory τ and the random function f ( · ) .

Proof: For any function g ( s ) ,

<!-- formula-not-decoded -->

Using that f ( · ) is independent of τ , we have that for all t

<!-- formula-not-decoded -->

The result now follow froms Lemma 11.7.

## 11.3 Bibliographic Remarks and Further Readings

The REINFOCE algorithm is due to [Williams, 1992], which is an example of the likelihood ratio method for gradient estimation [Glynn, 1990].

For standard optimization results in non-convex optimization (e.g. Lemma 11.6 and 11.8), we refer the reader to [Beck, 2017]. Our results for convergence rates for SGD to approximate stationary points follow from [Ghadimi and Lan, 2013].

## Chapter 12

## Optimality

We now seek to understand the global convergence properties of policy gradient methods, when given exact gradients. Here, we will largely limit ourselves to the (tabular) softmax policy class in Example 11.1.

Given our a starting distribution ρ over states, recall our objective is:

<!-- formula-not-decoded -->

where { π θ | θ ∈ Θ ⊂ R d } is some class of parametric policies.

While we are interested in good performance under ρ , we will see how it is helpful to optimize under a different measure µ . Specifically, we consider optimizing V π θ ( µ ) , i.e.

<!-- formula-not-decoded -->

even though our ultimate goal is performance under V π θ ( ρ ) .

We now consider the softmax policy parameterization (0.1). Here, we still have a non-concave optimization problem in general, as shown in Lemma 11.5, though we do show that global optimality can be reached under certain regularity conditions. From a practical perspective, the softmax parameterization of policies is preferable to the direct parameterization, since the parameters θ are unconstrained and standard unconstrained optimization algorithms can be employed. However, optimization over this policy class creates other challenges as we study in this section, as the optimal policy (which is deterministic) is attained by sending the parameters to infinity.

This chapter will study three algorithms for this problem, for the softmax policy class. The first performs direct policy gradient ascent on the objective without modification, while the second adds a log barrier regularizer to keep the parameters from becoming too large, as a means to ensure adequate exploration. Finally, we study the natural policy gradient algorithm and establish a global optimality convergence rate, with no dependence on the dimension-dependent factors.

The presentation in this chapter largely follows the results in [Agarwal et al., 2020d].

## 12.1 Vanishing Gradients and Saddle Points

To understand the necessity of optimizing under a distribution µ that is different from ρ , let us first give an informal argument that some condition on the state distribution of π , or equivalently µ , is necessary for stationarity to imply

a

Figure 0.1: (Vanishing gradient example) A deterministic, chain MDP of length H +2 . We consider a policy where π ( a | s i ) = θ s i ,a for i = 1 , 2 , . . . , H . Rewards are 0 everywhere other than r ( s H +1 , a 1 ) = 1 . See Proposition 12.1.

<!-- image -->

optimality. For example, in a sparse-reward MDP (where the agent is only rewarded upon visiting some small set of states), a policy that does not visit any rewarding states will have zero gradient, even though it is arbitrarily suboptimal in terms of values. Below, we give a more quantitative version of this intuition, which demonstrates that even if π chooses all actions with reasonable probabilities (and hence the agent will visit all states if the MDP is connected), then there is an MDP where a large fraction of the policies π have vanishingly small gradients, and yet these policies are highly suboptimal in terms of their value.

Concretely, consider the chain MDP of length H +2 shown in Figure 0.1. The starting state of interest is state s 0 and the discount factor γ = H/ ( H + 1) . Suppose we work with the direct parameterization, where π θ ( a | s ) = θ s,a for a = a 1 , a 2 , a 3 and π θ ( a 4 | s ) = 1 -θ s,a 1 -θ s,a 2 -θ s,a 3 . Note we do not over-parameterize the policy. For this MDP and policy structure, if we were to initialize the probabilities over actions, say deterministically, then there is an MDP (obtained by permuting the actions) where all the probabilities for a 1 will be less than 1 / 4 .

The following result not only shows that the gradient is exponentially small in H , it also shows that many higher order derivatives, up to O ( H/ log H ) , are also exponentially small in H .

Proposition 12.1 (Vanishing gradients at suboptimal parameters) . Consider the chain MDP of Figure 0.1, with H +2 states, γ = H/ ( H +1) , and with the direct policy parameterization (with 3 |S| parameters, as described in the text above). Suppose θ is such that 0 &lt; θ &lt; 1 (componentwise) and θ s,a 1 &lt; 1 / 4 (for all states s ). For all k ≤ H 40 log(2 H ) -1 , we have ∥ ∥ ∇ k θ V π θ ( s 0 ) ∥ ∥ ≤ (1 / 3) H/ 4 , where ∇ k θ V π θ ( s 0 ) is a tensor of the k th order derivatives of V π θ ( s 0 ) and the norm is the operator norm of the tensor. 1 Furthermore, V glyph[star] ( s 0 ) -V π θ ( s 0 ) ≥ ( H +1) / 8 -( H +1) 2 / 3 H .

We do not prove this lemma here (see Section 12.5). The lemma illustrates that lack of good exploration can indeed be detrimental in policy gradient algorithms, since the gradient can be small either due to π being near-optimal, or, simply because π does not visit advantageous states often enough. Furthermore, this lemma also suggests that varied results in the non-convex optimization literature, on escaping from saddle points, do not directly imply global convergence due to that the higher order derivatives are small.

While the chain MDP of Figure 0.1, is a common example where sample based estimates of gradients will be 0 under random exploration strategies; there is an exponentially small in H chance of hitting the goal state under a random exploration strategy. Note that this lemma is with regards to exact gradients. This suggests that even with exact computations (along with using exact higher order derivatives) we might expect numerical instabilities.

## 12.2 Policy Gradient Ascent

Let us now return to the softmax policy class, from Equation 0.1, where:

<!-- formula-not-decoded -->

1 The operator norm of a k th -order tensor J ∈ R d ⊗ k is defined as sup u 1 ,...,u k ∈ R d : ‖ u i ‖ 2 =1 〈 J, u 1 ⊗ . . . ⊗ u d 〉 .

where the number of parameters in this policy class is |S||A| .

Observe that:

<!-- formula-not-decoded -->

where 1 [ E ] is the indicator of E being true.

Lemma 12.2. For the softmax policy class, we have:

<!-- formula-not-decoded -->

Proof: Using the advantage expression for the policy gradient (see Theorem 11.4),

<!-- formula-not-decoded -->

where the last step uses that for any policy ∑ a π ( a | s ) A π ( s, a ) = 0 .

The update rule for gradient ascent is:

<!-- formula-not-decoded -->

Recall from Lemma 11.5 that, even for the case of the softmax policy class (which contains all stationary policies), our optimization problem is non-convex. Furthermore, due to the exponential scaling with the parameters θ in the softmax parameterization, any policy that is nearly deterministic will have gradients close to 0 . Specifically, for any sequence of policies π θ t that becomes deterministic, ‖∇ V π θ t ‖ → 0 .

In spite of these difficulties, it turns out we have a positive result that gradient ascent asymptotically converges to the global optimum for the softmax parameterization.

Theorem 12.3 (Global convergence for softmax parameterization) . Assume we follow the gradient ascent update rule as specified in Equation (0.2) and that the distribution µ is strictly positive i.e. µ ( s ) &gt; 0 for all states s . Suppose η ≤ (1 -γ ) 3 8 , then we have that for all states s , V ( t ) ( s ) → V glyph[star] ( s ) as t →∞ .

The proof is somewhat technical, and we do not provide a proof here (see Section 12.5).

Afew remarks are in order. Theorem 12.3 assumed that optimization distribution µ was strictly positive, i.e. µ ( s ) &gt; 0 for all states s . We conjecture that any gradient ascent may not globally converge if this condition is not met. The concern is that if this condition is not met, then gradient ascent may not globally converge due to that d π θ µ ( s ) effectively scales down the learning rate for the parameters associated with state s (see Equation 0.1).

Furthermore, there is strong reason to believe that the convergence rate for this is algorithm (in the worst case) is exponentially slow in some of the relevant quantities, such as in terms of the size of state space. We now turn to a regularization based approach to ensure convergence at a polynomial rate in all relevant quantities.

## 12.3 Log Barrier Regularization

Due to the exponential scaling with the parameters θ , policies can rapidly become near deterministic, when optimizing under the softmax parameterization, which can result in slow convergence. Indeed a key challenge in the asymptotic analysis in the previous section was to handle the growth of the absolute values of parameters as they tend to infinity. Recall that the relative-entropy for distributions p and q is defined as:

<!-- formula-not-decoded -->

Denote the uniform distribution over a set X by Unif X , and define the following log barrier regularized objective as:

<!-- formula-not-decoded -->

where λ is a regularization parameter. The constant (i.e. the last term) is not relevant with regards to optimization. This regularizer is different from the more commonly utilized entropy regularizer, a point which we return to later.

The policy gradient ascent updates for L λ ( θ ) are given by:

<!-- formula-not-decoded -->

We now see that any approximate first-order stationary points of the entropy-regularized objective is approximately globally optimal, provided the regularization is sufficiently small.

Theorem 12.4. (Log barrier regularization) Suppose θ is such that:

<!-- formula-not-decoded -->

and glyph[epsilon1] opt ≤ λ/ (2 |S| |A| ) . Then we have that for all starting state distributions ρ :

<!-- formula-not-decoded -->

We refer to ∥ ∥ ∥ ∥ d π glyph[star] ρ µ ∥ ∥ ∥ ∥ ∞ as the distribution mismatch coefficient . The above theorem shows the importance of having an appropriate measure µ ( s ) in order for the approximate first-order stationary points to be near optimal.

Proof: The proof consists of showing that max a A π θ ( s, a ) ≤ 2 λ/ ( µ ( s ) |S| ) for all states. To see that this is sufficient, observe that by the performance difference lemma (Lemma 1.16),

<!-- formula-not-decoded -->

which would then complete the proof.

We now proceed to show that max a A π θ ( s, a ) ≤ 2 λ/ ( µ ( s ) |S| ) . For this, it suffices to bound A π θ ( s, a ) for any stateaction pair s, a where A π θ ( s, a ) ≥ 0 else the claim is trivially true. Consider an ( s, a ) pair such that A π θ ( s, a ) &gt; 0 . Using the policy gradient expression for the softmax parameterization (see Equation 0.1),

<!-- formula-not-decoded -->

The gradient norm assumption ‖∇ θ L λ ( θ ) ‖ 2 ≤ glyph[epsilon1] opt implies that:

<!-- formula-not-decoded -->

where we have used A π θ ( s, a ) ≥ 0 . Rearranging and using our assumption glyph[epsilon1] opt ≤ λ/ (2 |S| |A| ) ,

<!-- formula-not-decoded -->

Solving for A π θ ( s, a ) in (0.5), we have:

<!-- formula-not-decoded -->

where the penultimate step uses glyph[epsilon1] opt ≤ λ/ (2 |S| |A| ) and the final step uses d π θ µ ( s ) ≥ (1 -γ ) µ ( s ) . This completes the proof.

The policy gradient ascent updates for L λ ( θ ) are given by:

<!-- formula-not-decoded -->

By combining the above theorem with the convergence of gradient ascent to first order stationary points (Lemma 11.6), we obtain the following corollary.

Corollary 12.5. (Iteration complexity with log barrier regularization) Let β λ := 8 γ (1 -γ ) 3 + 2 λ |S| . Starting from any initial θ (0) , consider the updates (0.6) with λ = glyph[epsilon1] (1 -γ ) 2 ∥ ∥ ∥ ∥ d π glyph[star] ρ µ ∥ ∥ ∥ ∥ ∞ and η = 1 /β λ . Then for all starting state distributions ρ , we have

<!-- formula-not-decoded -->

The corollary shows the importance of balancing how the regularization parameter λ is set relative to the desired accuracy glyph[epsilon1] , as well as the importance of the initial distribution µ to obtain global optimality.

Proof: [ of Corollary 12.5 ] Let β λ be the smoothness of L λ ( θ ) . A valid upper bound on β λ is:

<!-- formula-not-decoded -->

where we leave the proof as an exercise to the reader.

Using Theorem 12.4, the desired optimality gap glyph[epsilon1] will follow if we set λ = glyph[epsilon1] (1 -γ ) 2 ∥ ∥ ∥ ∥ d π glyph[star] ρ µ ∥ ∥ ∥ ∥ ∞ and if ‖∇ θ L λ ( θ ) ‖ 2 ≤

λ/ (2 |S| |A| ) . In order to complete the proof, we need to bound the iteration complexity of making the gradient sufficiently small.

By Lemma 11.6, after T iterations of gradient ascent with stepsize of 1 /β λ , we have

<!-- formula-not-decoded -->

where β λ is an upper bound on the smoothness of L λ ( θ ) . We seek to ensure

<!-- formula-not-decoded -->

Choosing T ≥ 8 β λ |S| 2 |A| 2 (1 -γ ) λ 2 satisfies the above inequality. Hence,

<!-- formula-not-decoded -->

where we have used that λ &lt; 1 . This completes the proof.

Entropy vs. log barrier regularization. A commonly considered regularizer is the entropy, where the regularizer would be:

<!-- formula-not-decoded -->

Note the entropy is far less aggressive in penalizing small probabilities, in comparison to the log barrier, which is equivalent to the relative entropy. In particular, the entropy regularizer is always bounded between 0 and log |A| , while the relative entropy (against the uniform distribution over actions), is bounded between 0 and infinity, where it tends to infinity as probabilities tend to 0 . Here, it can be shown that the convergence rate is asymptotically O (1 glyph[epsilon1] ) (see Section 12.5) though it is unlikely that the convergence rate for this method is polynomial in other relevant quantities, including |S| , |A| , 1 / (1 -γ ) , and the distribution mismatch coefficient. The polynomial convergence rate using the log barrier (KL) regularizer crucially relies on the aggressive nature in which the relative entropy prevents small probabilities.

## 12.4 The Natural Policy Gradient

Observe that a policy constitutes a family of probability distributions { π θ ( ·| s ) | s ∈ S} . We now consider a preconditioned gradient descent method based on this family of distributions. Recall that the Fisher information matrix

of a parameterized density p θ ( x ) is defined as E x ∼ p θ [ ∇ log p θ ( x ) ∇ log p θ ( x ) glyph[latticetop] ] . Now we let us define F θ ρ as an (average) Fisher information matrix on the family of distributions { π θ ( ·| s ) | s ∈ S} as follows:

<!-- formula-not-decoded -->

Note that the average is under the state-action visitation frequencies. The NPG algorithm performs gradient updates in the geometry induced by this matrix as follows:

<!-- formula-not-decoded -->

where M † denotes the Moore-Penrose pseudoinverse of the matrix M .

Throughout this section, we restrict to using the initial state distribution ρ ∈ ∆( S ) in our update rule in (0.8) (so our optimization measure µ and the performance measure ρ are identical). Also, we restrict attention to states s ∈ S reachable from ρ , since, without loss of generality, we can exclude states that are not reachable under this start state distribution 2 .

For the softmax parameterization, this method takes a particularly convenient form; it can be viewed as a soft policy iteration update.

Lemma 12.6. (Softmax NPG as soft policy iteration) For the softmax parameterization (0.1) , the NPG updates (0.8) take the form:

<!-- formula-not-decoded -->

where Z t ( s ) = ∑ a ∈A π ( t ) ( a | s ) exp ( ηA ( t ) ( s, a ) / (1 -γ ) ) . Here, v is only a state dependent offset (i.e. v s,a = c s for some c s ∈ R for each state s ), and, owing to the normalization Z t ( s ) , v has no effect on the update rule.

It is important to note that while the ascent direction was derived using the gradient ∇ θ V ( t ) ( ρ ) , which depends on ρ , the NPG update rule actually has no dependence on the measure ρ . Furthermore, there is no dependence on the state distribution d ( t ) ρ , which is due to the pseudoinverse of the Fisher information cancelling out the effect of the state distribution in NPG.

Proof: By definition of the Moore-Penrose pseudoinverse, we have that ( F θ ρ ) † ∇ V π θ ( ρ ) = w glyph[star] if an only if w glyph[star] is the minimum norm solution of:

<!-- formula-not-decoded -->

Let us first evaluate F θ ρ w . For the softmax policy parameterization, Lemma 0.1 implies:

<!-- formula-not-decoded -->

where w s is not a function of a . This implies that:

<!-- formula-not-decoded -->

where the last equality uses that w s is not a function of s . Again using the functional form of derivative of the softmax policy parameterization, we have:

<!-- formula-not-decoded -->

2 Specifically, we restrict the MDP to the set of states { s ∈ S : ∃ π such that d π ρ ( s ) &gt; 0 } .

This implies:

<!-- formula-not-decoded -->

Due to that w = 1 1 -γ A π θ leads to 0 error, the above implies that all 0 error solutions are of the form w = 1 1 -γ A π θ + v , where v is only a state dependent offset (i.e. v s,a = c s for some c s ∈ R for each state s ). The first claim follows due to that the minimum norm solution is one of these solutions. The proof of the second claim now follows by the definition of the NPG update rule, along with that v has no effect on the update rule due to the normalization constant Z t ( s ) .

We now see that this algorithm enjoys a dimension free convergence rate.

Theorem 12.7 (Global convergence for NPG) . Suppose we run the NPG updates (0.8) using ρ ∈ ∆( S ) and with θ (0) = 0 . Fix η &gt; 0 . For all T &gt; 0 , we have:

<!-- formula-not-decoded -->

Note in the above the theorem that the NPG algorithm is directly applied to the performance measure V π ( ρ ) , and the guarantees are also with respect to ρ . In particular, there is no distribution mismatch coefficient in the rate of convergence.

Now setting η ≥ (1 -γ ) 2 log |A| , we see that NPG finds an glyph[epsilon1] -optimal policy in a number of iterations that is at most:

<!-- formula-not-decoded -->

which has no dependence on the number of states or actions, despite the non-concavity of the underlying optimization problem.

The proof strategy we take borrows ideas from the classical multiplicative weights algorithm (see Section!12.5).

First, the following improvement lemma is helpful:

Lemma 12.8 (Improvement lower bound for NPG) . For the iterates π ( t ) generated by the NPG updates (0.8) , we have for all starting state distributions µ

<!-- formula-not-decoded -->

Proof: First, let us show that log Z t ( s ) ≥ 0 . To see this, observe:

<!-- formula-not-decoded -->

where we have used Jensen's inequality on the concave function log x and that ∑ a π ( t ) ( a | s ) A ( t ) ( s, a ) = 0 . By the

performance difference lemma,

<!-- formula-not-decoded -->

where the last step uses that d ( t +1) µ (1 γ ) µ (by (0.9)) and that log Z t ( s ) 0

≥ -≥ .

With this lemma, we now prove Theorem 12.7.

Proof: [ of Theorem 12.7 ] Since ρ is fixed, we use d glyph[star] as shorthand for d π glyph[star] ρ ; we also use π s as shorthand for the vector of π ( ·| s ) . By the performance difference lemma (Lemma 1.16),

<!-- formula-not-decoded -->

where we have used the closed form of our updates from Lemma 12.6 in the second step.

By applying Lemma 12.8 with d glyph[star] as the starting state distribution, we have:

<!-- formula-not-decoded -->

which gives us a bound on E s ∼ d glyph[star] log Z t ( s ) .

Using the above equation and that V ( t +1) ( ρ ) ≥ V ( t ) ( ρ ) (as V ( t +1) ( s ) ≥ V ( t ) ( s ) for all states s by Lemma 12.8), we have:

<!-- formula-not-decoded -->

The proof is completed using that V ( T ) ( ρ ) ≥ V ( T -1) ( ρ ) .

## 12.5 Bibliographic Remarks and Further Readings

The natural policy gradient method was originally presented in [Kakade, 2001]; a number of arguments for this method have been provided based on information geometry [Kakade, 2001, Bagnell and Schneider, 2003, Peters and Schaal, 2008].

The convergence rates in this chapter are largely derived from [Agarwal et al., 2020d]. The proof strategy for the NPG analysis has origins in the online regret framework in changing MDPs [Even-Dar et al., 2009], which would result in a worst rate in comparison to [Agarwal et al., 2020d]. This observation that the proof strategy from [Even-Dar et al., 2009] provided a convergence rate for the NPG was made in [Neu et al., 2017]. The faster NPG rate we present here is due to [Agarwal et al., 2020d]. The analysis of the MD-MPI algorithm [Geist et al., 2019] also implies a O (1 /T ) rate for the NPG, though with worse dependencies on other parameters.

Building on ideas in [Agarwal et al., 2020d], [Mei et al., 2020] showed that, for the softmax policy class, both the gradient ascent and entropy regularized gradient ascent asymptotically converge at a O (1 /t ) ; it is unlikely these methods are have finite rate which are polynomial in other quantities (such as the |S| , |A| , 1 / (1 -γ ) , and the distribution mismatch coefficient).

[Mnih et al., 2016] introduces the entropy regularizer (also see [Ahmed et al., 2019] for a more detailed empirical investigation).

## Chapter 13

## Function Approximation and the NPG

We now analyze the case of using parametric policy classes:

<!-- formula-not-decoded -->

where Π may not contain all stochastic policies (and it may not even contain an optimal policy). In contrast with the tabular results in the previous sections, the policy classes that we are often interested in are not fully expressive, e.g. d glyph[lessmuch] |S||A| (indeed |S| or |A| need not even be finite for the results in this section); in this sense, we are in the regime of function approximation.

We focus on obtaining agnostic results, where we seek to do as well as the best policy in this class (or as well as some other comparator policy). While we are interested in a solution to the (unconstrained) policy optimization problem

<!-- formula-not-decoded -->

(for a given initial distribution ρ ), we will see that optimization with respect to a different distribution will be helpful, just as in the tabular case,

We will consider variants of the NPG update rule (0.8):

<!-- formula-not-decoded -->

Our analysis will leverage a close connection between the NPG update rule (0.8) with the notion of compatible function approximation . We start by formalizing this connection. The compatible function approximation error also provides a measure of the expressivity of our parameterization, allowing us to quantify the relevant notion of approximation error for the NPG algorithm.

The main results in this chapter establish the effectiveness of NPG updates where there is error both due to statistical estimation (where we may not use exact gradients) and approximation (due to using a parameterized function class). Wewill see an precise estimation/approximation decomposition based on the compatible function approximation error.

The presentation in this chapter largely follows the results in [Agarwal et al., 2020d].

## 13.1 Compatible function approximation and the NPG

We now introduce the notion of compatible function approximation , which both provides some intuition with regards to policy gradient methods and it will help us later on with regards to characterizing function approximation.

Lemma 13.1 (Gradients and compatible function approximation) . Let w glyph[star] denote the following minimizer:

<!-- formula-not-decoded -->

where the squared error above is referred to as the compatible function approximation . Denote the best linear predictor of A π θ ( s, a ) using ∇ θ log π θ ( a | s ) by ̂ A π θ ( s, a ) , i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: The first order optimality conditions for w glyph[star] imply

<!-- formula-not-decoded -->

Rearranging and using the definition of ̂ A π θ ( s, a ) ,

<!-- formula-not-decoded -->

which completes the proof.

The next lemma shows that the weight vector above is precisely the NPG ascent direction. Precisely,

Lemma 13.2. We have that:

where w glyph[star] is a minimizer of the following regression problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: The above is a straightforward consequence of the first order optimality conditions (see Equation 0.2). Specifically, Equation 0.2, along with the advantage expression for the policy gradient (see Theorem 11.4), imply that w glyph[star] must satisfy:

which completes the proof.

<!-- formula-not-decoded -->

This lemma implies that we might write the NPG update rule as:

<!-- formula-not-decoded -->

where w glyph[star] is minimizer of the compatible function approximation error (which depends on θ .

The above regression problem can be viewed as 'compatible' function approximation: we are approximating A π θ ( s, a ) using the ∇ θ log π θ ( ·| s ) as features. We also consider a variant of the above update rule, Q -NPG, where instead of using advantages in the above regression we use the Q -values. This viewpoint provides a methodology for approximate updates, where we can solve the relevant regression problems with samples.

We have that:

## 13.2 Examples: NPG and Q -NPG

In practice, the most common policy classes are of the form:

<!-- formula-not-decoded -->

where f θ is a differentiable function. For example, the tabular softmax policy class is one where f θ ( s, a ) = θ s,a . Typically, f θ is either a linear function or a neural network. Let us consider the NPG algorithm, and a variant Q -NPG, in each of these two cases.

## 13.2.1 Log-linear Policy Classes and Soft Policy Iteration

For any state-action pair ( s, a ) , suppose we have a feature mapping φ s,a ∈ R d . Each policy in the log-linear policy class is of the form:

with θ ∈ R d . Here, we can take f θ ( s, a ) = θ · φ s,a .

<!-- formula-not-decoded -->

With regards to compatible function approximation for the log-linear policy class, we have:

<!-- formula-not-decoded -->

that is, φ θ s,a is the centered version of φ s,a . With some abuse of notation, we accordingly also define ¯ φ π for any policy π . Here, using (0.3), the NPG update rule (0.1) is equivalent to:

<!-- formula-not-decoded -->

(We have rescaled the learning rate η in comparison to (0.1)). Note that we recompute w glyph[star] for every update of θ . Here, the compatible function approximation error measures the expressivity of our parameterization in how well linear functions of the parameterization can capture the policy's advantage function.

We also consider a variant of the NPG update rule (0.1), termed Q -NPG , where:

<!-- formula-not-decoded -->

Note we do not center the features for Q -NPG; observe that Q π ( s, a ) is also not 0 in expectation under π ( ·| s ) , unlike the advantage function.

(NPG/ Q -NPG and Soft-Policy Iteration) We now see how we can view both NPG and Q -NPG as an incremental (soft) version of policy iteration, just as in Lemma 12.6 for the tabular case. Rather than writing the update rule in terms of the parameter θ , we can write an equivalent update rule directly in terms of the (log-linear) policy π :

<!-- formula-not-decoded -->

where Z s is normalization constant. While the policy update uses the original features φ instead of φ π , whereas the quadratic error minimization is terms of the centered features φ π , this distinction is not relevant due to that we may also instead use φ π (in the policy update) which would result in an equivalent update; the normalization makes the update invariant to (constant) translations of the features. Similarly, an equivalent update for Q -NPG, where we update π directly rather than θ , is:

<!-- formula-not-decoded -->

(On the equivalence of NPG and Q -NPG) If it is the case that the compatible function approximation error is 0 , then it straightforward to verify that the NPG and Q -NPG are equivalent algorithms, in that their corresponding policy updates will be equivalent to each other.

## 13.2.2 Neural Policy Classes

Now suppose f θ ( s, a ) is a neural network parameterized by θ ∈ R d , where the policy class Π is of form in (0.4). Observe:

<!-- formula-not-decoded -->

and, using (0.3), the NPG update rule (0.1) is equivalent to:

<!-- formula-not-decoded -->

(Again, we have rescaled the learning rate η in comparison to (0.1)).

The Q -NPG variant of this update rule is:

<!-- formula-not-decoded -->

## 13.3 The NPG 'Regret Lemma'

It is helpful for us to consider NPG more abstractly, as an update rule of the form

<!-- formula-not-decoded -->

We will now provide a lemma where w ( t ) is an arbitrary (bounded) sequence, which will be helpful when specialized. Recall a function f : R d → R is said to be β -smooth if for all x, x ′ ∈ R d :

<!-- formula-not-decoded -->

and, due to Taylor's theorem, recall that this implies:

<!-- formula-not-decoded -->

The following analysis of NPG is draws close connections to the mirror-descent approach used in online learning (see Section 13.6), which motivates us to refer to it as a 'regret lemma'.

Lemma 13.3. (NPG Regret Lemma) Fix a comparison policy ˜ π and a state distribution ρ . Assume for all s ∈ S and a ∈ A that log π θ ( a | s ) is a β -smooth function of θ . Consider the update rule (0.5) , where π (0) is the uniform distribution (for all states) and where the sequence of weights w (0) , . . . , w ( T ) , satisfies ‖ w ( t ) ‖ 2 ≤ W (but is otherwise arbitrary). Define:

<!-- formula-not-decoded -->

Using η = √ 2 log |A| / ( βW 2 T ) , we have that:

<!-- formula-not-decoded -->

This lemma is the key tool in understanding the role of function approximation of various algorithms. We will consider one example in detail with regards to the log-linear policy class (from Example 11.2).

Note that when err t = 0 , as will be the case with the (tabular) softmax policy class with exact gradients, we obtain a convergence rate of O ( √ 1 /T ) using a learning rate of η = O ( √ 1 /T ) . Note that this is slower than the faster rate of O (1 /T ) , provided in Theorem 12.7. Obtaining a bound that leads to a faster rate in the setting with errors requires more complicated dependencies on err t than those stated above.

Proof: By smoothness (see (0.6)),

<!-- formula-not-decoded -->

We use ˜ d as shorthand for d ˜ π ρ (note ρ and ˜ π are fixed); for any policy π , we also use π s as shorthand for the distribution π ( ·| s ) . Using the performance difference lemma (Lemma 1.16),

<!-- formula-not-decoded -->

Rearranging, we have:

<!-- formula-not-decoded -->

Proceeding,

<!-- formula-not-decoded -->

which completes the proof.

## 13.4 Q -NPG: Performance Bounds for Log-Linear Policies

For a state-action distribution υ , define:

<!-- formula-not-decoded -->

The iterates of the Q -NPG algorithm can be viewed as minimizing this loss under some (changing) distribution υ .

We now specify an approximate version of Q -NPG. It is helpful to consider a slightly more general version of the algorithm in the previous section, where instead of optimizing under a starting state distribution ρ , we have a different starting state-action distribution ν . The motivation for this is similar in spirit to our log barrier regularization: we seek to maintain exploration (and estimation) over the action space even if the current policy does not have coverage over the action space.

Analogous to the definition of the state-action visitation measure d π s 0 (see Equation 0.8), we define another state-action visitation measure, d π s 0 ,a 0 , as follows:

<!-- formula-not-decoded -->

where Pr π ( s t = s, a t = a | s 0 , a 0 ) is the probability that s t = s and a t = a , after starting at state s 0 , taking action a 0 , and following π thereafter. Again, we overload notation and write:

<!-- formula-not-decoded -->

where ν is a distribution over S × A .

Q -NPG will be defined with respect to the on-policy state action measure starting with s 0 , a 0 ∼ ν . As per our convention, we define

<!-- formula-not-decoded -->

The approximate version of this algorithm is:

<!-- formula-not-decoded -->

where the above update rule also permits us to constrain the norm of the update direction w ( t ) (alternatively, we could use glyph[lscript] 2 regularization as is also common in practice). The exact minimizer is denoted as:

<!-- formula-not-decoded -->

Note that w ( t ) glyph[star] depends on the current parameter θ ( t ) .

Our analysis will take into account both the excess risk (often also referred to as estimation error) and the approximation error . The standard approximation-estimation error decomposition is as follows:

<!-- formula-not-decoded -->

Using a sample based approach, we would expect glyph[epsilon1] stat = O (1 / √ N ) or better, where N is the number of samples used to estimate. w ( t ) glyph[star] In constrast, the approximation error is due to modeling error, and does not tend to 0 with more samples. We will see how these two errors have strikingly different impact on our final performance bound.

Note that we have already considered two cases where glyph[epsilon1] approx = 0 . For the tabular softmax policy class, it is immediate that glyph[epsilon1] approx = 0 . A more interesting example (where the state and action space could be infinite) is provided by

the linear parameterized MDP model from Chapter 8. Here, provided that we use the log-linear policy class (see Section 13.2.1) with features corresponding to the linear MDP features, it is straightforward to see that glyph[epsilon1] approx = 0 for this log-linear policy class. More generally, we will see the effect of model misspecification in our performance bounds.

We make the following assumption on these errors:

Assumption 13.4 (Approximation/estimation error bounds) . Let w (0) , w (1) , . . . w ( T -1) be the sequence of iterates used by the Q -NPG algorithm Suppose the following holds for all t &lt; T :

1. ( Excess risk ) Suppose the estimation error is bounded as follows:

<!-- formula-not-decoded -->

2. ( Approximation error ) Suppose the approximation error is bounded as follows:

<!-- formula-not-decoded -->

We will also see how, with regards to our estimation error, we will need a far more mild notion of coverage. Here, with respect to any state-action distribution υ , define:

<!-- formula-not-decoded -->

We make a the following conditioning assumption:

Assumption 13.5 (Relative condition number) . Fix a state distribution ρ (this will be what ultimately be the performance measure that we seek to optimize). Consider an arbitrary comparator policy π glyph[star] (not necessarily an optimal policy). With respect to π glyph[star] , define the state-action measure d glyph[star] as

<!-- formula-not-decoded -->

i.e. d glyph[star] samples states from the comparators state visitation measure, d π glyph[star] ρ and actions from the uniform distribution. Define and assume that κ is finite.

We later discuss why it is reasonable to expect that κ is not a quantity related to the size of the state space.

The main result of this chapter shows how the approximation error, the excess risk, and the conditioning, determine the final performance.

Theorem 13.6. Fix a state distribution ρ ; a state-action distribution ν ; an arbitrary comparator policy π glyph[star] (not necessarily an optimal policy). Suppose Assumption 13.5 holds with respect to these choices and that ‖ φ s,a ‖ 2 ≤ B for all s, a . Suppose the Q -NPG update rule (in (0.8) ) starts with θ (0) = 0 , η = √ 2 log |A| / ( B 2 W 2 T ) , and the (random) sequence of iterates satisfies Assumption 13.4. We have that:

<!-- formula-not-decoded -->

Note when glyph[epsilon1] approx = 0 , our convergence rate is O ( √ 1 /T ) plus a term that depends on the excess risk; hence, provided we obtain enough samples, then glyph[epsilon1] stat will also tend to 0 , and we will be competitive with the comparison policy π glyph[star] .

The above also shows the striking difference between the effects of estimation error and approximation error. A few remarks are now in order.

<!-- formula-not-decoded -->

Transfer learning, distribution shift, and the approximation error. In large scale problems, the worst case distribution mismatch factor ∥ ∥ ∥ d glyph[star] ν ∥ ∥ ∥ ∞ is unlikely to be small. However, this factor is ultimately due to transfer learning. Our approximation error is with respect to the fitting distribution d ( t ) , where we assume that L ( w ( t ) glyph[star] ; θ ( t ) , d ( t ) ) ≤ glyph[epsilon1] approx . As the proof will show, the relevant notion of approximation error will be L ( w ( t ) glyph[star] ; θ ( t ) , d glyph[star] ) , where d glyph[star] is the fixed comparators measure. In others words, to get a good performance bound we need to successfully have low transfer learning error to the fixed measure d glyph[star] . Furthermore, in many modern machine learning applications, this error is often is favorable, in that it is substantially better than worst case theory might suggest.

See Section 13.6 for further remarks on this point.

Dimension dependence in κ and the importance of ν . It is reasonable to think about κ as being dimension dependent (or worse), but it is not necessarily related to the size of the state space. For example, if ‖ φ s,a ‖ 2 ≤ B , then κ ≤ B 2 σ min ( E s,a ∼ ν [ φ s,a φ glyph[latticetop] s,a ]) though this bound may be pessimistic. Here, we also see the importance of choice of ν in having a small (relative) condition number; in particular, this is the motivation for considering the generalization which allows for a starting state-action distribution ν vs. just a starting state distribution µ (as we did in the tabular case). Roughly speaking, we desire a ν which provides good coverage over the features. As the following lemma shows, there always exists a universal distribution ν , which can be constructed only with knowledge of the feature set (without knowledge of d glyph[star] ), such that κ ≤ d .

Lemma 13.7. ( κ ≤ d is always possible) Let Φ = { φ ( s, a ) | ( s, a ) ∈ S × A} ⊂ R d and suppose Φ is a compact set. There always exists a state-action distribution ν , which is supported on at most d 2 state-action pairs and which can be constructed only with knowledge of Φ (without knowledge of the MDP or d glyph[star] ), such that:

<!-- formula-not-decoded -->

Proof: The distribution can be found through constructing the minimal volume ellipsoid containing Φ , i.e. the Lo ¨ wnerJohn ellipsoid. To be added...

Direct policy optimization vs. approximate value function programming methods Part of the reason for the success of the direct policy optimization approaches is to due their more mild dependence on the approximation error. Here, our theoretical analysis has a dependence on a distribution mismatch coefficient, ∥ ∥ ∥ d glyph[star] ν ∥ ∥ ∥ ∞ , while approximate value function methods have even worse dependencies. See Chapter 4. As discussed earlier and as can be seen in the regret lemma (Lemma 13.3), the distribution mismatch coefficient is due to that the relevant error for NPG is a transfer error notion to a fixed comparator distribution, while approximate value function methods have more stringent conditions where the error has to be small under, essentially, the distribution of any other policy.

## 13.4.1 Analysis

Proof: (of Theorem 13.6) For the log-linear policy class, due to that the feature mapping φ satisfies ‖ φ s,a ‖ 2 ≤ B , then it is not difficult to verify that log π θ ( a | s ) is a B 2 -smooth function. Using this and the NPG regret lemma (Lemma 13.3), we have:

<!-- formula-not-decoded -->

where we have used our setting of η .

We make the following decomposition of err t :

<!-- formula-not-decoded -->

For the first term, using that ∇ θ log π θ ( a | s ) = φ s,a -E a ′ ∼ π θ ( ·| s ) [ φ s,a ′ ] (see Section 13.2.1), we have:

<!-- formula-not-decoded -->

where we have used the definition of d glyph[star] and L ( w ( t ) glyph[star] ; θ ( t ) , d glyph[star] ) in the last step. Using following crude upper bound,

<!-- formula-not-decoded -->

(where the last step uses the defintion of d ( t ) , see Equation 0.7), we have that:

<!-- formula-not-decoded -->

For the second term, let us now show that:

<!-- formula-not-decoded -->

To see this, first observe that a similar argument to the above leads to:

<!-- formula-not-decoded -->

where we use the notation ‖ x ‖ 2 M := x glyph[latticetop] Mx for a matrix M and a vector x . From the definition of κ ,

<!-- formula-not-decoded -->

using that (1 -γ ) ν ≤ d π ( t ) ν (see (0.7)). Due to that w ( t ) glyph[star] minimizes L ( w ; θ ( t ) , d ( t ) ) over the set W := { w : ‖ w ‖ 2 ≤ W } , for any w ∈ W the first-order optimality conditions for w ( t ) glyph[star] imply that:

<!-- formula-not-decoded -->

Therefore, for any w ∈ W ,

<!-- formula-not-decoded -->

Noting that w ( t ) ∈ W by construction in Algorithm 0.8 yields the claimed bound on the second term in (0.10).

Using the bounds on the first and second terms in (0.9) and (0.10), along with concavity of the square root function, we have that:

<!-- formula-not-decoded -->

The proof is completed by substitution and using our assumptions on glyph[epsilon1] stat and glyph[epsilon1] bias .

## 13.5 Q -NPG Sample Complexity

To be added...

## 13.6 Bibliographic Remarks and Further Readings

The notion of compatible function approximation was due to [Sutton et al., 1999], which also proved the claim in Lemma 13.1. The close connection of the NPG update rule to compatible function approximation (Lemma 0.3) was noted in [Kakade, 2001].

The regret lemma (Lemma 13.3) for the NPG analysis has origins in the online regret framework in changing MDPs [EvenDar et al., 2009]. The convergence rates in this chapter are largely derived from [Agarwal et al., 2020d]. The Q-NPG algorithm for the log-linear policy classes is essentially the same algorithm as POLITEX [Abbasi-Yadkori et al., 2019], with a distinction that it is important to use a state action measure over the initial distribution. The analysis and error decomposition of Q-NPG is from [Agarwal et al., 2020d], which has a more general analysis of NPG with function approximation under the regret lemma. This more general approach also permits the analysis of neural policy classes, as shown in [Agarwal et al., 2020d]. Also, [Liu et al., 2019] provide an analysis of the TRPO algorithm [Schulman et al., 2015] (essentially the same as NPG) for neural network parameterizations in the somewhat restrictive linearized 'neural tangent kernel' regime.

## Chapter 14

## CPI, TRPO, and More

In this chapter, we consider conservative policy iteration (CPI) and trust-region constrained policy optimization (TRPO). Both CPI and TRPO can be understood as making small incremental update to the policy by forcing that the new policy's state action distribution is not too far away from the current policy's. We will see that CPI achieves that by forming a new policy that is a mixture of the current policy and a local greedy policy, while TRPO forcing that by explicitly adding a KL constraint (over polices' induced trajectory distributions space) in the optimization procedure. We will show that TRPO gives an equivalent update procedure as Natural Policy Gradient.

Along the way, we discuss the benefit of incremental policy update, by contrasting it to another family of policy update procedure called Approximate Policy Iteration (API), which performs local greedy policy search and could potentially lead to abrupt policy change. We show that API in general fails to converge or make local improvement, unless under a much stronger concentrability ratio assumption.

The algorithm and analysis of CPI is adapted from the original one in [Kakade and Langford, 2002], and the we follow the presentation of TRPO from [Schulman et al., 2015], while making a connection to the NGP algorithm.

## 14.1 Conservative Policy Iteration

As the name suggests, we will now describe a more conservative version of the policy iteration algorithm, which shifts the next policy away from the current policy with a small step size to prevent drastic shifts in successive state distributions.

We consider the discounted MDP here {S , A , P, r, γ, ρ } where ρ is the initial state distribution. Similar to Policy Gradient Methods, we assume that we have a restart distribution µ (i.e., the µ -restart setting). Throughout this section, for any policy π , we denote d π µ as the state-action visitation starting from s 0 ∼ µ instead of ρ , and d π the state-action visitation starting from the true initial state distribution ρ , i.e., s 0 ∼ ρ . Similarly, we denote V π µ as expected discounted total reward of policy π starting at µ , while V π as the expected discounted total reward of π with ρ as the initial state distribution. We assume A is discrete but S could be continuous.

CPI is based on the concept of Reduction to Supervised Learning . Specifically we will use the Approximate Greedy Policy Selector defined in Chapter 4 (Definition ?? ). We recall the definition of the ε -approximate Greedy Policy Selector G ε ( π, Π , µ ) below. Given a policy π , policy class Π , and a restart distribution µ , denote ̂ π = G ε ( π, Π , µ ) , we have that:

<!-- formula-not-decoded -->

˜

Recall that in Chapter 4 we explained two approach to implement such approximate oracle: one with a reduction to classification oracle, and the other one with a reduction to regression oracle.

## 14.1.1 The CPI Algorithm

CPI, summarized in Alg. 8, will iteratively generate a sequence of policies π i . Note we use π α = (1 -α ) π + απ ′ to refer to a randomized policy which at any state s , chooses an action according to π with probability 1 -α and according to π ′ with probability α . The greedy policy π ′ is computed using the ε -approximate greedy policy selector G ε ( π t , Π , µ ) . The algorithm is terminate when there is no significant one-step improvement over π t , i.e., E s ∼ d π t µ A π t ( s, π ′ ( s ))) ≤ ε .

```
Algorithm 8 Conservative Policy Iteration (CPI) Input: Initial policy π 0 ∈ Π , accuracy parameter ε . 1: for t = 0 , 1 , 2 . . . do 2: π ′ = G ε ( π t , Π , µ ) 3: if E s ∼ d π t µ A π t ( s, π ′ ( s )) ≤ ε then 4: Return π t 5: end if 6: Update π t +1 = (1 -α ) π t + απ ′ 7: end for
```

The main intuition behind the algorithm is that the stepsize α controls the difference between state distributions of π t and π t +1 . Let us look into the performance difference lemma to get some intuition on this conservative update. From PDL, we have:

<!-- formula-not-decoded -->

where the last equality we use the fact that π t +1 = (1 -α ) π t + απ ′ and A π t ( s, π t ( s )) = 0 for all s . Thus, if we can search for a policy π ′ ∈ Π that maximizes E s ∼ d π t +1 µ A π t ( s, π ′ ( s )) and makes E s ∼ d π t +1 µ A π t ( s, π ′ ( s )) &gt; 0 , then we can guarantee policy improvement. However, at episode t , we do not know the state distribution of π t +1 and all we have access to is d π t µ . Thus, we explicitly make the policy update procedure to be conservative such that d π t µ and the new policy's distribution d π t +1 µ is guaranteed to be not that different. Thus we can hope that E s ∼ d π t +1 µ A π t ( s, π ′ ( s )) is close to E s ∼ d π t µ A π t ( s, π ′ ( s )) , and the latter is something that we can manipulate using the greedy policy selector.

Below we formalize the above intuition and show that with small enough α , we indeed can ensure monotonic policy improvement.

We start from the following lemma which shows that π t +1 and π t are close to each other in terms of total variation distance at any state, and d π t +1 µ and d π t µ are close as well.

Lemma 14.1 (Similar Policies imply similar state visitations) . Consider any t , we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, we have:

Proof: The first claim in the above lemma comes from the definition of policy update:

<!-- formula-not-decoded -->

Denote P π h as the state distribution resulting from π at time step h with µ as the initial state distribution. We consider bounding ‖ P π t +1 h -P π t h ‖ 1 with h ≥ 1 .

<!-- formula-not-decoded -->

Take absolute value on both sides, we get:

<!-- formula-not-decoded -->

Now use the definition of d π µ , we have:

<!-- formula-not-decoded -->

Add glyph[lscript] 1 norm on both sides, we get:

<!-- formula-not-decoded -->

It is not hard to verify that ∑ ∞ h =0 γ h h = γ (1 -γ ) 2 . Thus, we can conclude that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above lemma states that if π t +1 and π t are close in terms of total variation distance for every state, then the total variation distance between the resulting state visitations from π t +1 and π t will be small up to a effective horizon 1 / (1 -γ ) amplification.

The above lemma captures the key of the conservative policy update. Via the conservative policy update, we make sure that d π t +1 µ and d π t µ are close to each other in terms of total variation distance. Now we use the above lemma to show a monotonic policy improvement.

Theorem 14.2 (Monotonic Improvement in CPI) . Consider any episode t . Denote A = E s ∼ d π t µ A π t ( s, π ′ ( s )) . We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above lemma shows that as long as we still have positive one-step improvement, i.e., A &gt; 0 , then we guarantee that π t +1 is strictly better than π t .

Proof: Via PDL, we have:

Recall Lemma 14.1, we have:

<!-- formula-not-decoded -->

where the first inequality we use the fact that for any two distributions P 1 and P 2 and any function f , | E x ∼ P 1 f ( x ) -E x ∼ P 2 f ( x ) | ≤ sup x | f ( x ) |‖ P 1 -P 2 ‖ 1 , for the second inequality, we use the fact that | A π ( s, a ) | ≤ 1 / (1 -γ ) for any π, s, a , and the last inequality uses Lemma 14.1.

For the second part of the above theorem, note that we want to maximum the policy improvement as much as possible by choosing α . So we can pick α which maximizes α ( A -2 αγ/ (1 -γ ) 2 ) . This gives us the α we claimed in the lemma. Plug in α back into α ( A -2 αγ/ (1 -γ ) 2 ) , we conclude the second part of the theorem.

The above theorem indicates that with the right choice of α , we guarantee that the policy is making improvement as long as A &gt; 0 . Recall the termination criteria in CPI where we terminate CPI when A ≤ ε . Putting these results together, we obtain the following overall convergence guarantee for the CPI algorithm.

Theorem 14.3 (Local optimality of CPI) . Algorithm 8 terminates in at most 8 γ/glyph[epsilon1] 2 steps and outputs a policy π t satisfying max π ∈ Π E s ∼ d π t µ A π t ( s, π ( s )) ≤ 2 ε .

Proof: Note that our reward is bounded in [0 , 1] which means that V π µ ∈ [0 , 1 / (1 -γ )] . Note that we have shown in Theorem 14.2, every iteration t , we have policy improvement at least A 2 (1 -γ ) 8 γ , where recall A at episode t is defined as A = E s ∼ d π t µ A π t ( s, π ′ ( s )) . If the algorithm does not terminate at episode t , then we guarantee that:

<!-- formula-not-decoded -->

Since V π µ is upper bounded by 1 / (1 -γ ) , so we can at most make improvement 8 γ/glyph[epsilon1] 2 many iterations.

Finally, recall that ε -approximate greedy policy selector π ′ = G ε ( π t , Π , µ ) , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof.

Theorem 14.3 can be viewed as a local optimality guarantee in a sense. It shows that when CPI terminates, we cannot find a policy π ∈ Π that achieves local improvement over the returned policy more than ε . However, this does not necessarily imply that the value of π is close to V glyph[star] . However, similar to the policy gradient analysis, we can turn this local guarantee into a global one when the restart distribution µ covers d π glyph[star] . We formalize this intuition next.

Theorem 14.4 (Global optimality of CPI) . Upon termination, we have a policy π such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words, if our policy class is rich enough to approximate the policy max a ∈A A π ( s, a ) under d π µ , i.e., glyph[epsilon1] Π is small, and µ covers d π glyph[star] in a sense that ∥ ∥ ∥ d π glyph[star] µ ∥ ∥ ∥ ∞ ≤ ∞ , CPI guarantees to find a near optimal policy.

Proof: By the performance difference lemma,

<!-- formula-not-decoded -->

where the second inequality holds due to the fact that max a A π ( s, a ) ≥ 0 , the third inequality uses the fact that d π µ ( s ) ≥ (1 -γ ) µ ( s ) for any s and π , and the last inequality uses the definition glyph[epsilon1] Π and Theorem 14.3.

It is informative to contrast CPI and policy gradient algorithms due to the similarity of their guarantees. Both provide local optimality guarantees. For CPI, the local optimality always holds, while for policy gradients it requires a smooth value function as a function of the policy parameters. If the distribution mismatch between an optimal policy and the output of the algorithm is not too large, then both algorithms further yield a near optimal policy. The similarities are not so surprising. Both algorithms operate by making local improvements to the current policy at each iteration, by inspecting its advantage function. The changes made to the policy are controlled using a stepsize parameter in both the approaches. It is the actual mechanism of the improvement which differs in the two cases. Policy gradients assume that the policy's reward is a differentiable function of the parameters, and hence make local improvements through gradient ascent. The differentiability is certainly an assumption and does not necessarily hold for all policy classes. An easy example is when the policy itself is not an easily differentiable function of its parameters. For instance, if the policy is parametrized by regression trees, then performing gradient updates can be challenging.

In CPI, on the other hand, the basic computational primitive required on the policy class is the ability to maximize the advantage function relative to the current policy. Notice that Algorithm 8 does not necessarily restrict to a policy class, such as a set of parametrized policies as in policy gradients. Indeed, due to the reduction to supervised learning approach (e.g., using the weighted classification oracle CO), we can parameterize policy class via decision tree, for

instance. This property makes CPI extremely attractive. Any policy class over which efficient supervised learning algorithms exist can be adapted to reinforcement learning with performance guarantees.

A second important difference between CPI and policy gradients is in the notion of locality. Policy gradient updates are local in the parameter space, and we hope that this makes small enough changes to the state distribution that the new policy is indeed an improvement on the older one (for instance, when we invoke the performance difference lemma between successive iterates). While this is always true in expectation for correctly chosen stepsizes based on properties of stochastic gradient ascent on smooth functions, the variance of the algorithm and lack of robustness to suboptimal stepsizes can make the algorithm somewhat finicky. Indeed, there are a host of techniques in the literature to both lower the variance (through control variates) and explicitly control the state distribution mismatch between successive iterates of policy gradients (through trust region techniques). On the other hand, CPI explicitly controls the amount of perturbation to the state distribution by carefully mixing policies in a manner which does not drastically alter the trajectories with high probability. Indeed, this insight is central to the proof of CPI, and has been instrumental in several follow-ups, both in the direct policy improvement as well as policy gradient literature.

## 14.2 Trust Region Methods and Covariant Policy Search

So far we have seen policy gradient methods and CPI which all uses a small step-size to ensure incremental update in policies. Another popular approach for incremental policy update is to explicitly forcing small change in policies' distribution via a trust region constraint. More specifically, let us go back to the general policy parameterization π θ . At iteration t with the current policy π θ t , we are interested in the following local trust-region constrained optimization:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where recall Pr π µ ( τ ) is the trajectory distribution induced by π starting at s 0 ∼ µ , and KL ( P 1 || P 2 ) are KL-divergence between two distribution P 1 and P 2 . Namely we explicitly perform local policy search with a constraint forcing the new policy not being too far away from Pr π θ t µ in terms of KL divergence.

As we are interested in small local update in parameters, we can perform sequential quadratic programming here, i.e., we can further linearize the objective function at θ t and quadratize the KL constraint at θ t to form a local quadratic programming:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we denote ∇ 2 KL | θ = θ t as the Hessian of the KL constraint measured at θ t . Note that KL divergence is not a valid metric as it is not symmetric. However, its local quadratic approximation can serve as a valid local distance metric, as we prove below that the Hessian ∇ 2 KL | θ = θ t is a positive semi-definite matrix. Indeed, we will show that the Hessian of the KL constraint is exactly equal to the fisher information matrix, and the above quadratic programming exactly reveals a Natural Policy Gradient update. Hence Natural policy gradient can also be interpreted as performing sequential quadratic programming with KL constraint over policy's trajectory distributions.

To match to the practical algorithms in the literature (e.g., TRPO), below we focus on episode finite horizon setting again (i.e., an MDP with horizon H ).

Claim 14.5. Consider a finite horizon MDP with horizon H . Consider any fixed θ t . We have:

<!-- formula-not-decoded -->

Proof: We first recall the trajectory distribution in finite horizon setting.

<!-- formula-not-decoded -->

We first prove that the gradient of KL is zero. First note that:

<!-- formula-not-decoded -->

Thus, for ∇ θ KL ( Pr π θ t µ || Pr π θ µ ) , we have:

<!-- formula-not-decoded -->

where we have seen the last step when we argue the unbiased nature of policy gradient with an action independent baseline.

Now we move to the Hessian.

<!-- formula-not-decoded -->

where in the last equation we use the fact that E s h ,a h ∼ P π θ t h ∇ 2 π θ ( a h | s h ) π θ ( a h | s h ) = 0 .

The above claim shows that a second order taylor expansion of the KL constraint over trajectory distribution gives a local distance metric at θ t :

<!-- formula-not-decoded -->

where again F θ t := H E s,a ∼ d π θ t ∇ ln π θ t ( a | s ) ( ∇ ln π θ t ( a | s )) glyph[latticetop] is proportional to the fisher information matrix. Note that F θ t is a PSD matrix and thus d ( θ, θ t ) := ( θ -θ t ) F θ t ( θ -θ t ) is a valid distance metric. By sequential quadratic programming, we are using local geometry information of the trajectory distribution manifold induced by the parameterization θ , rather the naive Euclidean distance in the parameter space. Such a method is sometimes referred to as Covariant Policy Search , as the policy update procedure will be invariant to linear transformation of parameterization (See Section 14.3 for further discussion).

Now using the results from Claim 14.5, we can verify that the local policy optimization procedure in Eq. 0.2 exactly recovers the NPG update, where the step size is based on the trust region parameter δ . Denote ∆ = θ -θ t , we have:

<!-- formula-not-decoded -->

which gives the following update procedure:

<!-- formula-not-decoded -->

where note that we use the self-normalized learning rate computed using the trust region parameter δ .

## 14.2.1 Proximal Policy Optimization

Here we consider an glyph[lscript] ∞ style trust region constraint:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Namely, we restrict the new policy such that it is close to π θ t at every state s under the total variation distance. Recall the CPI's update, CPI indeed makes sure that the new policy will be close the old policy at every state. In other words, the new policy computed by CPI is a feasible solution of the constraint Eq. 0.4, but is not the optimal solution of the above constrained optimization program. Also one downside of the CPI algorithm is that one needs to keep all previous learned policies around, which requires large storage space when policies are parameterized by large deep neural networks.

Proximal Policy Optimization (PPO) aims to directly optimize the objective Eq. 0.3 using multiple steps of gradient updates, and approximating the constraints Eq. 0.4 via a clipping trick. We first rewrite the objective function using importance weighting:

<!-- formula-not-decoded -->

where we can easily approximate the expectation E s ∼ d π θ t µ E a ∼ π θ t ( ·| s ) via finite samples s ∼ d π θ t , a ∼ π θ t ( ·| s ) .

To make sure π θ ( a | s ) and π θ t ( a | s ) are not that different, PPO modifies the objective function by clipping the density ratio π θ ( a | s ) and π θ t ( a | s ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The outer min makes sure the objective function L ( θ ) is a lower bound of the original objective. PPO then proposes to collect a dataset ( s, a ) with s ∼ d π θ t µ and a ∼ π θ t ( ·| s ) , and then perform multiple steps of mini-batch stochastic gradient ascent on L ( θ ) .

One of the key difference between PPO and other algorithms such as NPG is that PPO targets to optimizes objective Eq. 0.5 via multiple steps of mini-batch stochastic gradient ascent with mini-batch data from d π θ t µ and π θ t , while algorithm such as NPG indeed optimizes the first order taylor expansion of Eq. 0.5 at θ t , i.e.,

<!-- formula-not-decoded -->

upper to some trust region constraints (e.g., ‖ θ -θ t ‖ 2 ≤ δ in policy gradient, and ‖ θ -θ t ‖ 2 F θ t ≤ δ for NPG ).

## 14.3 Bibliographic Remarks and Further Readings

The analysis of CPI is adapted from the original one in [Kakade and Langford, 2002]. There have been a few further interpretations of CPI. One interesting perspective is that CPI can be treated as a boosting algorithm [Scherrer and Geist, 2014].

More generally, CPI and NPG are part of family of incremental algorithms, including Policy Search by Dynamic Programming (PSDP) [Bagnell et al., 2004] and MD-MPI [Geist et al., 2019]. PSDP operates in a finite horizon setting and optimizes a sequence of time-dependent policies; from the last time step to the first time step, every iteration of, PSDP only updates the policy at the current time step while holding the future policies fixed - thus making incremental update on the policy. See [Scherrer, 2014] for more a detailed discussion and comparison of some of these approaches. Mirror Descent-Modified Policy Iteration (MD-MPI) algorithm [Geist et al., 2019] is a family of actor-critic style algorithms which is based on regularization and is incremental in nature; with negative entropy as the Bregman divergence (for the tabular case), MD-MPI recovers the NPG the tabular case (for the softmax parameterization).

Broadly speaking, these incremental algorithms can improve upon the stringent concentrability conditions for approximate value iteration methods, presented in Chapter 4. Scherrer [2014] provide a more detailed discussion of bounds which depend on these density ratios. As discussed in the last chapter, the density ratio for NPG can be interpreted as a factor due to transfer learning to a single, fixed distribution.

The interpretation of NPG as Covariant Policy Search is due to [Bagnell and Schneider, 2003], as the policy update procedure will be invariant to linear transformations of the parameterization; see [Bagnell and Schneider, 2003] for a more detailed discussion on this.

The TRPO algorithm is due to [Schulman et al., 2015]. The original TRPO analysis provides performance guarantees, largely relying on a reduction to the CPI guarantees. In this Chapter, we make a sharper connection of TRPO to NPG, which was subsequently observed by a number of researchers; this connection provides a sharper analysis for the generalization and approximation behavior of TRPO (e.g. via the results presented in Chapter 13). In practice, a popular variant is the Proximal Policy Optimization (PPO) algorithm [Schulman et al., 2017].

## Part 4

## Further Topics

## Chapter 15

## Imitation Learning

In this chapter, we study imitation learning. Unlike the Reinforcement Learning setting, in Imitation Learning, we do not have access to the ground truth reward function (or cost function), but instead, we have expert demonstrations. We often assume that the expert is a policy that approximately optimizes the underlying reward (or cost) functions. The goal is to leverage the expert demonstrations to learn a policy that performs as good as the expert.

We consider three settings of imitation learning: (1) pure offline where we only have expert demonstrations and no more real world interaction is allowed, (2) hybrid where we have expert demonstrations, and also is able to interact with the real world (e.g., have access to the ground truth transition dynamics); (3) interactive setting where we have an interactive expert and also have access to the underlying reward (cost) function.

## 15.1 Setting

We will focus on finite horizon MDPs M = {S , A , r, µ, P, H } where r is the reward function but is unknown to the learner. We represent the expert as a closed-loop policy π glyph[star] : S ↦→ ∆( A ) . For analysis simplicity, we assume the expert π glyph[star] is indeed the optimal policy of the original MDP M with respect to the ground truth reward r . Again, our goal is to learn a policy ̂ π : S ↦→ ∆( A ) that performs as well as the expert, i.e., V ̂ π needs to be close to V glyph[star] , where V π denotes the expected total reward of policy π under the MDP M . We denote d π as the state-action visitation of policy π under M .

We assume we have a pre-collected expert dataset in the format of { s glyph[star] i , a glyph[star] i } i M =1 where s glyph[star] i , a glyph[star] i ∼ d π glyph[star] .

## 15.2 Offline IL: Behavior Cloning

We study offline IL here. Specifically, we study the classic Behavior Cloning algorithm.

We consider a policy class Π = { π : S ↦→ ∆( A ) } . We consider the following realizable assumption.

Assumption 15.1. We assume Π is rich enough such that π glyph[star] ∈ Π .

For analysis simplicity, we assume Π is discrete. But our sample complexity will only scale with respect to ln( | Π | ) .

Behavior cloning is one of the simplest Imitation Learning algorithm which only uses the expert data D glyph[star] and does not require any further interaction with the MDP. It computes a policy via a reduction to supervised learning. Specifically,

we consider a reduction to Maximum Likelihood Estimation (MLE):

<!-- formula-not-decoded -->

Namely we try to find a policy from Π that has the maximum likelihood of fitting the training data. As this is a reduction to MLE, we can leverage the existing classic analysis of MLE (e.g., Chapter 7 in [Geer and van de Geer, 2000]) to analyze the performance of the learned policy.

Theorem 15.2 (MLE Guarantee) . Consider the MLE procedure (Eq. 0.1). With probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

We refer readers to Section E, Theorem 21 in [Agarwal et al., 2020b] for detailed proof of the above MLE guarantee.

Now we can transfer the average divergence between ̂ π and π glyph[star] to their performance difference V ̂ π and V π glyph[star] .

One thing to note is that BC only ensures that the learned policy ̂ π is close to π glyph[star] under d π glyph[star] . Outside of d π glyph[star] 's support, we have no guarantee that ̂ π and π glyph[star] will be close to each other. The following theorem shows that a compounding error occurs when we study the performance of the learned policy V ̂ pi .

Theorem 15.3 (Sample Complexity of BC) . With probability at least 1 -δ , BC returns a policy ̂ π such that:

<!-- formula-not-decoded -->

Proof: We start with the performance difference lemma and the fact that E a ∼ π ( ·| s ) A π ( s, a ) = 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the fact that sup s,a,π | A π ( s, a ) | ≤ 1 1 -γ , and the fact that ( E [ x ]) 2 ≤ E [ x 2 ] .

Using Theorem 15.2 and rearranging terms conclude the proof.

For Behavior cloning, from the supervised learning error the quadratic polynomial dependency on the effect horizon 1 / (1 -γ ) is not avoidable in worst case [Ross and Bagnell, 2010]. This is often referred as the distribution shift issue in the literature. Note that ̂ π is trained under d π glyph[star] , but during execution, ̂ π makes prediction on states that are generated by itself, i.e., d ̂ π , instead of the training distribution d π glyph[star] .

## 15.3 The Hybrid Setting: Statistical Benefit and Algorithm

The question we want to answer here is that if we know the underlying MDP's transition P (but the reward is still unknown), can we improve Behavior Cloning? In other words:

what is the benefit of the known transition in addition to the expert demonstrations?

Instead of a quadratic dependency on the effective horizon, we should expect a linear dependency on the effective horizon. The key benefit of the known transition is that we can test our policy using the known transition to see how far away we are from the expert's demonstrations, and then use the known transition to plan to move closer to the expert demonstrations.

In this section, we consider a statistical efficient, but computationally intractable algorithm, which we use to demonstrate that informationally theoretically, by interacting with the underlying known transition, we can do better than Behavior Cloning. In the next section, we will introduce a popular and computationally efficient algorithm Maximum Entropy Inverse Reinforcement Learning (MaxEnt-IRL) which operates under this setting (i.e., expert demonstrations with a known transition).

We start from the same policy class Π as we have in the BC algorithm, and again we assume realizability (Assumption 15.1) and Π is discrete.

Since we know the transition P , information theoretically, for any policy π , we have d π available (though it is computationally intractable to compute d π for large scale MDPs). We have ( s glyph[star] i , a glyph[star] i ) i M =1 ∼ d π glyph[star] .

Below we present an algorithm which we called Distribution Matching with Scheff´ e Tournament (DM-ST).

For any two policies π and π ′ , we denote f π,π ′ as the following witness function:

<!-- formula-not-decoded -->

We denote the set of witness functions as:

<!-- formula-not-decoded -->

Note that |F| ≤ | Π | 2 .

DM-ST selects ̂ π using the following procedure:

<!-- formula-not-decoded -->

The following theorem captures the sample complexity of DM-ST.

Theorem 15.4 (Sample Complexity of DM-ST) . With probability at least 1 -δ , DM-ST finds a policy ̂ π such that:

<!-- formula-not-decoded -->

Proof: The proof basically relies on a uniform convergence argument over F of which the size is | Π | 2 . First we note that for all policy π ∈ Π :

<!-- formula-not-decoded -->

where the first equality comes from the fact that F includes arg max f : ‖ f ‖ ∞ ≤ 1 [ E s,a ∼ d π f ( s, a ) -E s,a ∼ d glyph[star] f ( s, a )] . Via Hoeffding's inequality and a union bound over , we get that with probability at least 1 δ , for all f :

F -∈ F

<!-- formula-not-decoded -->

glyph[negationslash]

Denote ̂ f := arg max f ∈F [ E s,a ∼ d ̂ π f ( s, a ) -E s,a ∼ d glyph[star] f ( s, a ) ] , and ˜ f := arg max f ∈F E s,a ∼ d ̂ π f ( s, a ) -1 M ∑ M i =1 f ( s i , a i ) . Hence, for ̂ π , we have:

<!-- formula-not-decoded -->

where in the third inequality we use the optimality of ̂ π .

Recall that V π = E s,a ∼ d π r ( s, a ) / (1 -γ ) , we have:

<!-- formula-not-decoded -->

This concludes the proof.

Note that above theorem confirms the statistical benefit of having the access to a known transition: comparing to the classic Behavior Cloning algorithm, the approximation error of DM-ST only scales linearly with respect to horizon 1 / (1 -γ ) instead of quadratically.

## 15.3.1 Extension to Agnostic Setting

So far we focused on agnostic learning setting. What would happen if π glyph[star] glyph[negationslash]∈ Π ? We can still run our DM-ST algorithm as is. We state the sample complexity of DM-ST in agnostic setting below.

Theorem 15.5 (Agnostic Guarantee of DM-ST) . Assume Π is finite, but π glyph[star] glyph[negationslash]∈ Π . With probability at least 1 -δ , DM-ST learns a policy ̂ π such that:

<!-- formula-not-decoded -->

Proof: We first define some terms below. Denote ˜ π := argmin π ∈ Π ‖ d π -d glyph[star] ‖ 1 . Let us denote:

<!-- formula-not-decoded -->

Starting with a triangle inequality, we have:

<!-- formula-not-decoded -->

where the first inequality uses the definition of f , the second inequality uses the fact that ̂ π is the minimizer of max f ∈F E s,a ∼ d π f ( s, a ) -1 M ∑ M i =1 f ( s glyph[star] i , a glyph[star] i ) , along the way we also use Heoffding's inequality where ∀ f ∈ F , | E s,a ∼ d glyph[star] f ( s, a ) -∑ M i =1 f ( s glyph[star] i , a glyph[star] i ) | ≤ 2 √ ln( |F| /δ ) /M , with probability at least 1 -δ .

As we can see, comparing to the realizable setting, here we have an extra term that is related to min π ∈ Π ‖ d π -d glyph[star] ‖ 1 . Note that the dependency on horizon also scales linearly in this case. In general, the constant 3 in front of min π ∈ Π ‖ d π -d glyph[star] ‖ 1 is not avoidable in Scheff´ e estimator [Devroye and Lugosi, 2012].

## 15.4 Maximum Entropy Inverse Reinforcement Learning

Similar to Behavior cloning, we assume we have a dataset of state-action pairs from expert D glyph[star] = { s glyph[star] i , a glyph[star] i } N i =1 where s glyph[star] i , a glyph[star] i ∼ d π glyph[star] . Different from Behavior cloning, here we assume that we have access to the underlying MDP's transition, i.e., we assume transition is known and we can do planning if we were given a cost function.

Weassume that we are given a state-action feature mapping φ : S×A ↦→ R d (this can be extended infinite dimensional feature space in RKHS, but we present finite dimensional setting for simplicity). We assume the true ground truth cost function as c ( s, a ) := θ glyph[star] · φ ( s, a ) with ‖ θ glyph[star] ‖ 2 ≤ 1 and θ glyph[star] being unknown.

The goal of the learner is to compute a policy π : S ↦→ ∆( A ) such that when measured under the true cost function, it performs as good as the expert, i.e., E s,a ∼ d π θ glyph[star] · φ ( s, a ) ≈ E s,a ∼ d π glyph[star] θ glyph[star] · φ ( s, a ) .

We will focus on finite horizon MDP setting in this section. We denote ρ π ( τ ) as the trajectory distribution induced by π and d π as the average state-action distribution induced by π .

## 15.4.1 MaxEnt IRL: Formulation and The Principle of Maximum Entropy

MaxEnt IRL uses the principle of Maximum Entropy and poses the following policy optimization optimization program:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that ∑ N i =1 φ ( s glyph[star] i , a glyph[star] i ) /N is an unbiased estimate of E s,a ∼ d π glyph[star] φ ( s, a ) . MaxEnt-IRL searches for a policy that maximizes the entropy of its trajectory distribution subject to a moment matching constraint.

Note that there could be many policies that satisfy the moment matching constraint, i.e., E s,a ∼ d π φ ( s, a ) = E s,a ∼ d π glyph[star] φ ( s, a ) , and any feasible solution is guaranteed to achieve the same performance of the expert under the ground truth cost function θ glyph[star] · φ ( s, a ) . The maximum entropy objective ensures that the solution is unique.

Using the Markov property, we notice that:

<!-- formula-not-decoded -->

This, we can rewrite the MaxEnt-IRL as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 15.4.2 Algorithm

To better interpret the objective function, below we replace ∑ i φ ( s glyph[star] i , a glyph[star] i ) /N by its expectation E s,a ∼ d π glyph[star] φ ( s, a ) . Note that we can use standard concentration inequalities to bound the difference between ∑ i φ ( s glyph[star] i , a glyph[star] i ) /N and E s,a ∼ d π glyph[star] φ ( s, a ) .

Using Lagrange multipliers, we can rewrite the constrained optimization program in Eq. 0.2 as follows:

<!-- formula-not-decoded -->

The above objective conveys a clear goal of our imitation learning problem: we are searching for π that minimizes the MMDbetween d π and d π glyph[star] with a (negative) entropy regularization on the trajectory distribution of policy π .

To derive an algorithm that optimizes the above objective, we first again swap the min max order via the minimax theorem:

<!-- formula-not-decoded -->

The above objective proposes a natural algorithm where we perform projected gradient ascent on θ , while perform best response update on π , i.e., given θ , we solve the following planning problem:

<!-- formula-not-decoded -->

Note that the above objective can be understood as planning with cost function θ glyph[latticetop] φ ( s, a ) with an additional negative entropy regularization on the trajectory distribution. On the other hand, given π , we can compute the gradient of θ , which is simply the difference between the expected features:

<!-- formula-not-decoded -->

which gives the following gradient ascent update on θ :

<!-- formula-not-decoded -->

Algorithm of MaxEnt-IRL MaxEnt-IRL updates π and θ alternatively using Eq. 0.4 and Eq. 0.5, respectively. We summarize the algorithm in Alg. 9. Note that for the stochastic gradient of θ t , we see that it is the empirical average feature difference between the current policy π t and the expert policy π glyph[star] .

## Algorithm 9 MaxEnt-IRL

Input: Expert data D glyph[star] = { s glyph[star] i , a glyph[star] i } i M =1 , MDP M , parameters β, η, N .

- 2: for t = 1 , 2 , . . . , do
- 1: Initialize θ 0 with ‖ θ 0 ‖ 2 ≤ 1 .
- 3: Entropy-regularized Planning with cost θ glyph[latticetop] t φ ( s, a ) : π t ∈ argmin π E s,a ∼ d π [ θ glyph[latticetop] t φ ( s, a ) + β ln π ( a | s ) ] .
- 5: Stochastic Gradient Update: θ ′ = θ t + η [ 1 N ∑ N i =1 φ ( s i , a i ) -1 M ∑ M i =1 φ ( s glyph[star] i , a glyph[star] i ) ] .
- 4: Draw samples { s i , a i } N i =1 ∼ d π t by executing π t in M .
- 6: end for

Note that Alg. 9 uses an entropy-regularized planning oracle. Below we discuss how to implement such entropyregularized planning oracle via dynamic programming.

## 15.4.3 Maximum Entropy RL: Implementing the Planning Oracle in Eq. 0.4

The planning oracle in Eq. 0.4 can be implemented in a value iteration fashion using Dynamic Programming. We denote c ( s, a ) := θ · φ ( s, a ) .

We are interested in implementing the following planning objective:

<!-- formula-not-decoded -->

The subsection has its own independent interests beyond the framework of imitation learning. This maximum entropy regularized planning formulation is widely used in RL as well and it is well connected to the framework of RL as Inference.

As usually, we start from the last time step H -1 . For any policy π and any state-action ( s, a ) , we have the cost-to-go Q π H -1 ( s, a ) :

<!-- formula-not-decoded -->

We have:

<!-- formula-not-decoded -->

Take gradient with respect to ρ , set it to zero and solve for ρ , we get:

<!-- formula-not-decoded -->

Substitute π glyph[star] H -1 back to the expression in Eq. 0.6, we get:

<!-- formula-not-decoded -->

i.e., we apply a softmin operator rather than the usual min operator (recall here we are minimizing cost).

With V glyph[star] h +1 , we can continue to h . Denote Q glyph[star] ( s, a ) as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again we can show that the minimizer of the above program has the following form:

<!-- formula-not-decoded -->

Substitute π glyph[star] h back to Q glyph[star] h , we can show that:

<!-- formula-not-decoded -->

where we see again that V glyph[star] h is based on a softmin operator on Q glyph[star] .

Thus the soft value iteration can be summarized below:

Soft Value Iteration:

<!-- formula-not-decoded -->

We can continue the above procedure to h = 0 , which gives us the optimal policies, all in the form of Eq. 0.7

## 15.5 Interactive Imitation Learning: AggreVaTe and Its Statistical Benefit over Offline IL Setting

We study the Interactive Imitation Learning setting where we have an expert policy that can be queried at any time during training, and we also have access to the ground truth reward signal.

We present AggreVaTe (Aggregate Values to Imitate) first and then analyze its sample complexity under the realizable setting.

Again we start with a realizable policy class Π that is discrete. We denote ∆(Π) as the convex hull of Π and each policy π ∈ ∆(Π) is a mixture policy represented by a distribution ρ ∈ R | Π | with ρ [ i ] ≥ 0 and ‖ ρ ‖ 1 = 1 . With this parameterization, our decision space simply becomes ∆( | Π | ) , i.e., any point in ∆( | Π | ) corresponds to a mixture policy. Notation wise, given ρ ∈ ∆( | Π | ) , we denote π ρ as the corresponding mixture policy. We denote π i as the i-th policy in Π . We denote ρ [ i ] as the i -th element of the vector ρ .

For V glyph[star] h , we have:

## Algorithm 10 AggreVaTe

Input: The interactive expert, regularization λ

- 1: Initialize ρ 0 to be a uniform distribution.
- 2: for t = 0 , 2 , . . . , do
- 3: Sample s t ∼ d π ρ t
- 5: Policy update: ρ t +1 = argmax ρ ∈ ∆( | Π | ) ∑ t j =0 ∑ | Π | i =1 ρ [ i ] ( E a ∼ π i ( ·| s t ) A glyph[star] ( s t , a ) ) -λ ∑ | Π | i =1 ρ [ i ] ln( ρ [ i ])
- 4: Query expert to get A glyph[star] ( s t , a ) for all a ∈ A
- 6: end for

AggreVaTe assumes an interactive expert from whom I can query for action feedback. Basically, given a state s , let us assume that expert returns us the advantages of all actions, i.e., one query of expert oracle at any state s returns A glyph[star] ( s, a ) , ∀ a ∈ A . 1

The algorithm in summarized in Alg. 10.

To analyze the algorithm, let us introduce some additional notations. Let us denote glyph[lscript] t ( ρ ) = ∑ | Π | i =1 ρ [ i ] E a ∼ π i ( ·| s ) A glyph[star] ( s t , a ) , which is dependent on state s t generated at iteration t , and is a linear function with respect to decision variable ρ . AggreVaTe is essentially running the specific online learning algorithm, Follow-the-Regularized Leader (FTRL) (e.g., see [Shalev-Shwartz, 2011]) with Entropy regularization:

<!-- formula-not-decoded -->

Denote c = max s,a A π ( s, a ) . This implies that sup ρ,t ‖ glyph[lscript] t ( ρ ) ‖ ≤ c . FTRL with linear loss functions and entropy regularization gives the following deterministic regret guarantees ([Shalev-Shwartz, 2011]):

<!-- formula-not-decoded -->

We will analyze AggreVaTe's sample complexity using the above result.

Theorem 15.6 (Sample Complexity of AggreVaTe) . Denote c = sup s,a | A glyph[star] ( s, a ) | . Let us denote glyph[epsilon1] Π and glyph[epsilon1] stat as follows:

<!-- formula-not-decoded -->

With probability at least 1 -δ , after M iterations (i.e., M calls of expert oracle), AggreVaTe finds a policy ̂ π such that:

<!-- formula-not-decoded -->

Proof: At each iteration t , let us define ˜ glyph[lscript] t ( ρ t ) as follows:

<!-- formula-not-decoded -->

1 Technically one cannot use one query to get A glyph[star] ( s, a ) for all a . But one can use importance weighting to get an unbiased estimate of A glyph[star] ( s, a ) for all a via just one expert roll-out. For analysis simplicity, we assume one expert query at s returns the whole vector A glyph[star] ( s, · ) ∈ R |A| .

Denote E t [ glyph[lscript] t ( ρ t )] as the conditional expectation of glyph[lscript] t ( ρ t ) , conditioned on all history up and including the end of iteration t -1 . Thus, we have E t [ glyph[lscript] t ]( ρ t ) = ˜ glyph[lscript] t ( ρ t ) as ρ t only depends on the history up to the end of iteration t -1 . Also note that | glyph[lscript] t ( ρ ) | ≤ c . Thus by Azuma-Hoeffding inequality (Theorem A.5), we get:

<!-- formula-not-decoded -->

with probability at least 1 -δ . Now use Eq. 0.8, and denote ρ glyph[star] = argmin ρ ∈ ∆(Π) 1 M ∑ M -1 t =0 glyph[lscript] t ( ρ ) , i.e., the best minimizer that minimizes the average loss ∑ M -1 t =0 glyph[lscript] t ( ρ ) /M . we get:

<!-- formula-not-decoded -->

which means that there must exist a ˆ t ∈ { 0 , . . . , M -1 } , such that:

<!-- formula-not-decoded -->

Now use Performance difference lemma, we have:

<!-- formula-not-decoded -->

Rearrange terms, we get:

<!-- formula-not-decoded -->

which concludes the proof.

Remark We analyze the glyph[epsilon1] Π by discussing realizable setting and non-realizable setting separately. When Π is realizable, i.e., π glyph[star] ∈ Π , by the definition of our loss function glyph[lscript] t , we immediately have ∑ M i =1 glyph[lscript] t ( ρ glyph[star] ) ≥ 0 since A glyph[star] ( s, π glyph[star] ( s )) = 0 for all s ∈ S . Moreover, when π glyph[star] is not the globally optimal policy, it is possible that glyph[epsilon1] Π := ∑ M i =1 glyph[lscript] t ( ρ glyph[star] ) /M &gt; 0 , which implies that when M → ∞ (i.e., glyph[epsilon1] stat → 0 ), AggreVaTe indeed can learn a policy that outperforms the expert policy π glyph[star] . In general when π glyph[star] glyph[negationslash]∈ Π , there might not exist a mixture policy ρ that achieves positive advantage against π glyph[star] under the M training samples { s 0 , . . . , s M -1 } . In this case, glyph[epsilon1] Π &lt; 0 .

Does AggreVaTe avoids distribution shift? Under realizable setting, note that the bound explicitly depends on sup s,a | A glyph[star] ( s, a ) | (i.e., it depends on | glyph[lscript] t ( ρ t ) | for all t ). In worst case, it is possible that sup s,a | A glyph[star] ( s, a ) | = Θ(1 / (1 -γ )) , which implies that AggreVaTe could suffer a quadratic horizon dependency, i.e., 1 / (1 -γ ) 2 . Note that DM-ST provably scales linearly with respect to 1 / (1 -γ ) , but DM-ST requires a stronger assumption that the transition P is known.

When sup s,a | A glyph[star] ( s, a ) | = o (1 / (1 -γ )) , then AggreVaTe performs strictly better than BC. This is possible when the MDP is mixing quickly under π glyph[star] , or Q glyph[star] is L- Lipschitz continuous, i.e., | Q glyph[star] ( s, a ) -Q glyph[star] ( s, a ′ ) | ≤ L ‖ a -a ′ ‖ , with bounded range in action space, e.g., sup a,a ′ ‖ a -a ′ ‖ ≤ β ∈ R + . In this case, if L and β are independent of 1 / (1 -γ ) , then sup s,a | A glyph[star] ( s, a ) | ≤ Lβ , which leads to a βL 1 -γ dependency.

When π glyph[star] glyph[negationslash]∈ Π , the agnostic result of AggreVaTe and the agnostic result of DM-ST is not directly comparable. Note that the model class error that DM-ST suffers is algorithmic-independent, i.e., it is min π ∈ Π ‖ d π -d π glyph[star] ‖ 1 and it only depends on π glyph[star] and Π , while the model class glyph[epsilon1] Π in AggreVaTe is algorithmic path-dependent, i.e., in additional to π glyph[star] and Π , it depends the policies π 1 , π 2 , . . . computed during the learning process. Another difference is that min π ∈ Π ‖ d π -d π glyph[star] ‖ 1 ∈ [0 , 2] , while glyph[epsilon1] Π indeed could scale linearly with respect to 1 / (1 -γ ) , i.e., glyph[epsilon1] π ∈ [ -1 1 -γ , 0] (assume π glyph[star] is the globally optimal policy for simplicity).

## 15.6 Bibliographic Remarks and Further Readings

Behavior cloning was used in autonomous driving back in 1989 [Pomerleau, 1989], and the distribution shift and compounding error issue was studied by Ross and Bagnell [2010]. Ross and Bagnell [2010], Ross et al. [2011] proposed using interactive experts to alleviate the distribution shift issue.

MaxEnt-IRL was proposed by Ziebart et al. [2008]. The original MaxEnt-IRL focuses on deterministic MDPs and derived distribution over trajectories. Later Ziebart et al. [2010] proposed the Principle of Maximum Causal Entropy framework which captures MDPs with stochastic transition. MaxEnt-IRL has been widely used in real applications (e.g., [Kitani et al., 2012, Ziebart et al., 2009]).

To the best of our knowledge, the algorithm Distribution Matching with Scheff´ e Tournament introduced in this chapter is new here and is the first to demonstrate the statistical benefit (in terms of sample complexity of the expert policy) of the hybrid setting over the pure offline setting with general function approximation.

Recently, there are approaches that extend the linear cost functions used in MaxEnt-IRL to deep neural networks which are treated as discriminators to distinguish between experts datasets and the datasets generated by the policies. Different distribution divergences have been proposed, for instance, JS-divergence [Ho and Ermon, 2016], general f-divergence which generalizes JS-divergence [Ke et al., 2019], and Integral Probability Metric (IPM) which includes Wasserstein distance [Sun et al., 2019b]. While these adversarial IL approaches are promising as they fall into the hybrid setting, it has been observed that empirically, adversarial IL sometimes cannot outperform simple algorithms such as Behavior cloning which operates in pure offline setting (e.g., see experiments results on common RL benchmarks from Brantley et al. [2019]).

Another important direction in IL is to combine expert demonstrations and reward functions together (i.e., perform Imitation together with RL). There are many works in this direction, including learning with pre-collected expert data [Hester et al., 2017, Rajeswaran et al., 2017, Cheng et al., 2018, Le et al., 2018, Sun et al., 2018], learning with an interactive expert [Daum´ e et al., 2009, Ross and Bagnell, 2014, Chang et al., 2015, Sun et al., 2017, Cheng et al., 2020, Cheng and Boots, 2018]. The algorithm AggreVaTe was originally introduced in [Ross and Bagnell, 2014]. A policy gradient and natural policy gradient version of AggreVaTe are introduced in [Sun et al., 2017]. The precursor of AggreVaTe is the algorithm Data Aggregation (DAgger) [Ross et al., 2011], which leverages interactive experts and a reduction to no-regret online learning, but without assuming access to the ground truth reward signals.

The maximum entropy RL formulation has been widely used in RL literature as well. For instance, Guided Policy Search (GPS) (and its variants) use maximum entropy RL formulation as its planning subroutine [Levine and Koltun, 2013, Levine and Abbeel, 2014]. We refer readers to the excellent survey from Levine [2018] for more details of Maximum Entropy RL formulation and its connections to the framework of RL as Probabilistic Inference.

## Chapter 16

## Linear Quadratic Regulators

This chapter will introduce some of the fundamentals of optimal control for the linear quadratic regulator model. This model is an MDP, with continuous states and actions. While the model itself is often inadequate as a global model, it can be quite effective as a locally linear model (provided our system does not deviate away from the regime where our linear model is reasonable approximation).

The basics of optimal control theory can be found in any number of standards text Anderson and Moore [1990], Evans [2005], Bertsekas [2017]. The treatment of Gauss-Newton and the NPG algorithm are due to Fazel et al. [2018].

## 16.1 The LQR Model

In the standard optimal control problem, a dynamical system is described as

<!-- formula-not-decoded -->

where f t maps a state x t ∈ R d , a control (the action) u t ∈ R k , and a disturbance w t , to the next state x t +1 ∈ R d , starting from an initial state x 0 . The objective is to find the control policy π which minimizes the long term cost,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where H is the time horizon (which can be finite or infinite).

In practice, this is often solved by considering the linearized control (sub-)problem where the dynamics are approximated by

<!-- formula-not-decoded -->

with the matrices A t and B t are derivatives of the dynamics f and where the costs are approximated by a quadratic function in x t and u t .

This chapter focuses on an important special case: finite and infinite horizon problem referred to as the linear quadratic regulator (LQR) problem. We can view this model as being an local approximation to non-linear model. However, we will analyze these models under the assumption that they are globally valid.

Finite Horizon LQRs. The finite horizon LQR problem is given by

<!-- formula-not-decoded -->

where initial state x 0 ∼ D is assumed to be randomly distributed according to distribution D ; the disturbance w t ∈ R d follows the law of a multi-variate normal with covariance σ 2 I ; the matrices A t ∈ R d × d and B t ∈ R d × k are referred to as system (or transition) matrices; Q ∈ R d × d and R ∈ R k × k are both positive definite matrices that parameterize the quadratic costs. Note that this model is a finite horizon MDP, where the S = R d and A = R k .

Infinite Horizon LQRs. We also consider the infinite horizon LQR problem:

<!-- formula-not-decoded -->

Note that here we are assuming the dynamics are time homogenous. We will assume that the optimal objective function (i.e. the optimal average cost) is finite; this is referred to as the system being controllable . This is a special case of an MDP with an average reward objective.

Throughout this chapter, we assume A and B are such that the optimal cost is finite. Due to the geometric nature of the system dynamics (say for a controller which takes controls u t that are linear in the state x t ), there may exists linear controllers with infinite costs. This instability of LQRs (at least for some A and B and some controllers) leads to that the theoretical analysis often makes various assumptions on A and B in order to guarantee some notion of stability. In practice, the finite horizon setting is more commonly used in practice, particularly due to that the LQR model is only a good local approximation of the system dynamics, where the infinite horizon model tends to be largely of theoretical interest. See Section 16.6.

The infinite horizon discounted case? The infinite horizon discounted case tends not to be studied for LQRs. This is largely due to that, for the undiscounted case (with the average cost objective), we have may infinte costs (due to the aforementioned geometric nature of the system dynamics); in such cases, discounting will not necessarily make the average cost finite.

## 16.2 Bellman Optimality:

## Value Iteration &amp; The Algebraic Riccati Equations

A standard result in optimal control theory shows that the optimal control input can be written as a linear function in the state. As we shall see, this is a consequence of the Bellman equations.

## 16.2.1 Planning and Finite Horizon LQRs

Slightly abusing notation, it is convenient to define the value function and state-action value function with respect to the costs as follows: For a policy π , a state x , and h ∈ { 0 , . . . H -1 } , we define the value function V π h : R d → R as

<!-- formula-not-decoded -->

where again expectation is with respect to the randomness of the trajectory, that is, the randomness in state transitions. Similarly, the state-action value (or Q-value) function Q π h : R d × R k → R is defined as

<!-- formula-not-decoded -->

We define V glyph[star] and Q glyph[star] analogously.

The following theorem provides a characterization of the optimal policy, via the algebraic Riccati equations. These equations are simply the value iteration algorithm for the special case of LQRs.

Theorem 16.1. (Value Iteration and the Riccati Equations). Suppose R is positive definite. The optimal policy is a linear controller specified by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, P t can be computed iteratively, in a backwards manner, using the following algebraic Riccati equations, where for t ∈ [ H ] ,

<!-- formula-not-decoded -->

and where P H = Q . (The above equation is simply the value iteration algorithm).

Furthermore, for t ∈ [ H ] , we have that:

<!-- formula-not-decoded -->

We often refer to K glyph[star] t as the optimal control gain matrices. It is straightforward to generalize the above when σ ≥ 0 .

We have assumed that R is strictly positive definite, to avoid have to working with the pseudo-inverse; the theorem is still true when R = 0 , provided we use a pseudo-inverse.

Proof: By the Bellman optimality conditions (Theorem ?? for episodic MDPs), we know the optimal policy (among all possibly history dependent, non-stationary, and randomized policies), is given by a deterministic stationary policy which is only a function of x t and t . We have that:

<!-- formula-not-decoded -->

due to that x H = A H -1 x + B H -1 u + w H -1 , and E [ w glyph[latticetop] H -1 Qw H -1 ] = Trace ( σ 2 Q ) . Due to that this is a quadratic function of u , we can immediately derive that the optimal control is given by:

<!-- formula-not-decoded -->

where the last step uses that P H := Q .

For notational convenience, let K = K glyph[star] H -1 , A = A H -1 , and B = B H -1 . Using the optimal control at x , i.e.

where

u = -K glyph[star] H -1 x , we have:

<!-- formula-not-decoded -->

where the fourth step uses our expression for K = K glyph[star] H -1 . This proves our claim for t = H -1 .

This implies that:

<!-- formula-not-decoded -->

The remainder of the proof follows from a recursive argument, which can be verified along identical lines to the t = H -1 case.

## 16.2.2 Planning and Infinite Horizon LQRs

Theorem 16.2. Suppose that the optimal cost is finite and that R is positive definite. Let P be a solution to the following algebraic Riccati equation:

<!-- formula-not-decoded -->

(Note that P is a positive definite matrix). We have that the optimal policy is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the optimal control gain is:

We have that P is unique and that the optimal average cost is σ 2 Trace( P ) .

As before, P parameterizes the optimal value function. We do not prove this theorem here though it follows along similar lines to the previous proof, via a limiting argument.

To find P , we can again run the recursion:

<!-- formula-not-decoded -->

starting with P = Q , which can be shown to converge to the unique positive semidefinite solution of the Riccati equation (since one can show the fixed-point iteration is contractive). Again, this approach is simply value iteration.

## 16.3 Convex Programs to find P and K glyph[star]

For the infinite horizon LQR problem, the optimization may be formulated as a convex program. In particular, the LQR problem can also be expressed as a semidefinite program (SDP) with variable P , for the infinite horizon case. We now present this primal program along with the dual program.

Note that these programs are the analogues of the linear programs from Section ?? for MDPs. While specifying these linear programs for an LQR (as an LQR is an MDP) would result in infinite dimensional linear programs, the special structure of the LQR implies these primal and dual programs have a more compact formulation when specified as an SDP.

## 16.3.1 The Primal for Infinite Horizon LQR

The primal optimization problem is given as:

<!-- formula-not-decoded -->

where the optimization variable is P . This SDP has a unique solution, P glyph[star] , which satisfies the algebraic Riccati equations (Equation 0.1 ); the optimal average cost of the infinite horizon LQR is σ 2 Trace ( P glyph[star] ) ; and the optimal policy is given by Equation 0.2.

The SDP can be derived by relaxing the equality in the Riccati equation to an inequality, then using the Schur complement lemma to rewrite the resulting Riccati inequality as linear matrix inequality. In particular, we can consider the relaxation where P must satisfy:

<!-- formula-not-decoded -->

That the solution to this relaxed optimization problem leads to the optimal P glyph[star] is due to the Bellman optimality conditions.

Now the Schur complement lemma for positive semi-definiteness is as follows: define

<!-- formula-not-decoded -->

(for matrices D , E , and F of appropriate size, with D and F being square symmetric matrices). We have that X is PSD if and only if

<!-- formula-not-decoded -->

This shows that the constraint set is equivalent to the above relaxation.

## 16.3.2 The Dual

The dual optimization problem is given as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the optimization variable is a symmetric matrix Σ , which is a ( d + k ) × ( d + k ) matrix with the block structure:

<!-- formula-not-decoded -->

The interpretation of Σ is that it is the covariance matrix of the stationary distribution. This analogous to the statevisitation measure for an MDP.

This SDP has a unique solution, say Σ glyph[star] . The optimal gain matrix is then given by:

<!-- formula-not-decoded -->

## 16.4 Policy Iteration, Gauss Newton, and NPG

Note that the noise variance σ 2 does not impact the optimal policy, in either the discounted case or in the infinite horizon case.

Here, when we examine local search methods, it is more convenient to work with case where σ = 0 . In this case, we can work with cumulative cost rather the average cost. Precisely, when σ = 0 , the infinite horizon LQR problem takes the form:

<!-- formula-not-decoded -->

where they dynamics evolves as

Note that we have directly parameterized our policy as a linear policy in terms of the gain matrix K , due to that we know the optimal policy is linear in the state. Again, we assume that C ( K glyph[star] ) is finite; this assumption is referred to as the system being controllable .

<!-- formula-not-decoded -->

We now examine local search based approaches, where we will see a close connection to policy iteration. Again, we have a non-convex optimization problem:

Lemma 16.3. (Non-convexity) If d ≥ 3 , there exists an LQR optimization problem, min K C ( K ) , which is not convex or quasi-convex.

Regardless, we will see that gradient based approaches are effective. For local search based approaches, the importance of (some) randomization, either in x 0 or noise through having a disturbance, is analogous to our use of having a widecoverage distribution µ (for MDPs).

## 16.4.1 Gradient Expressions

Gradient descent on C ( K ) , with a fixed stepsize η , follows the update rule:

<!-- formula-not-decoded -->

It is helpful to explicitly write out the functional form of the gradient. Define P K as the solution to:

<!-- formula-not-decoded -->

and, under this definition, it follows that C ( K ) can be written as:

<!-- formula-not-decoded -->

Also, define Σ K as the (un-normalized) state correlation matrix, i.e.

<!-- formula-not-decoded -->

Lemma 16.4. (Policy Gradient Expression) The policy gradient is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as a result the gradient can be written as ∇ C ( K ) = 2 E K Σ K .

For convenience, define E K to be

Proof: Observe:

<!-- formula-not-decoded -->

Let ∇ denote the gradient with respect to K ; note that ∇ C K (( A -BK ) x 0 ) has two terms as function of K , one with respect to K in the subscript and one with respect to the input ( A -BK ) x 0 . This implies

<!-- formula-not-decoded -->

where we have used recursion and that x 1 = ( A -BK ) x 0 . Taking expectations completes the proof.

The natural policy gradient. Let us now motivate a version of the natural gradient. The natural policy gradient follows the update:

<!-- formula-not-decoded -->

where F ( θ ) is the Fisher information matrix. A natural special case is using a linear policy with additive Gaussian noise, i.e.

<!-- formula-not-decoded -->

where K ∈ R k × d and σ 2 is the noise variance. In this case, the natural policy gradient of K (when σ is considered fixed) takes the form:

<!-- formula-not-decoded -->

Note a subtlety here is that C ( π K ) is the randomized policy.

To see this, one can verify that the Fisher matrix of size kd × kd , which is indexed as [ G K ] ( i,j ) , ( i ′ ,j ′ ) where i, i ′ ∈ { 1 , . . . k } and j, j ′ ∈ { 1 , . . . d } , has a block diagonal form where the only non-zeros blocks are [ G K ] ( i, · ) , ( i, · ) = Σ K (this is the block corresponding to the i -th coordinate of the action, as i ranges from 1 to k ). This form holds more generally, for any diagonal noise.

## 16.4.2 Convergence Rates

We consider three exact rules, where we assume access to having exact gradients. As before, we can also estimate these gradients through simulation. For gradient descent, the update is

<!-- formula-not-decoded -->

For natural policy gradient descent, the direction is defined so that it is consistent with the stochastic case, as per Equation 0.4, in the exact case the update is:

<!-- formula-not-decoded -->

One show that the Gauss-Newton update is:

<!-- formula-not-decoded -->

(Gauss-Newton is non-linear optimization approach which uses a certain Hessian approximation. It can be show that this leads to the above update rule.) Interestingly, for the case when η = 1 , the Gauss-Newton method is equivalent to the policy iteration algorithm, which optimizes a one-step deviation from the current policy.

The Gauss-Newton method requires the most complex oracle to implement: it requires access to ∇ C ( K ) , Σ K , and R + B glyph[latticetop] P K B ; as we shall see, it also enjoys the strongest convergence rate guarantee. At the other extreme, gradient descent requires oracle access to only ∇ C ( K ) and has the slowest convergence rate. The natural policy gradient sits in between, requiring oracle access to ∇ C ( K ) and Σ K , and having a convergence rate between the other two methods.

In this theorem, ‖ M ‖ 2 denotes the spectral norm of a matrix M .

Theorem 16.5. (Global Convergence of Gradient Methods) Suppose C ( K 0 ) is finite and, for µ defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

suppose µ &gt; 0 .

- Gauss-Newton case: Suppose η = 1 , the Gauss-Newton algorithm (Equation 0.7) enjoys the following performance bound:
- Natural policy gradient case: For a stepsize η = 1 / ( ‖ R ‖ 2 + ‖ B ‖ 2 2 C ( K 0 ) µ ) , natural policy gradient descent (Equation 0.6) enjoys the following performance bound:

<!-- formula-not-decoded -->

- Gradient descent case: For any starting policy K 0 , there exists a (constant) stepsize η (which could be a function of K 0 ), such that:

<!-- formula-not-decoded -->

## 16.4.3 Gauss-Newton Analysis

We only provide a proof for the Gauss-Newton case (see 16.6 for further readings).

We overload notation and let K denote the policy π ( x ) = Kx . For the infinite horizon cost function, define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next lemma is identical to the performance difference lemma.

Lemma 16.6. (Cost difference lemma) Suppose K and K ′ have finite costs. Let { x ′ t } and { u ′ t } be state and action sequences generated by K ′ , i.e. starting with x ′ 0 = x and using u ′ t = -K ′ x ′ t . It holds that:

<!-- formula-not-decoded -->

and and

Also, for any x , the advantage is:

<!-- formula-not-decoded -->

Proof: Let c ′ t be the cost sequence generated by K ′ . Telescoping the sum appropriately:

<!-- formula-not-decoded -->

which completes the first claim (the third equality uses the fact that x = x 0 = x ′ 0 ). For the second claim, observe that:

<!-- formula-not-decoded -->

And, for u = K ′ x ,

<!-- formula-not-decoded -->

which completes the proof.

We have the following corollary, which can be viewed analogously to a smoothness lemma.

Corollary 16.7. ('Almost' smoothness) C ( K ) satisfies:

<!-- formula-not-decoded -->

To see why this is related to smoothness (recall the definition of a smooth function in Equation 0.6), suppose K ′ is sufficiently close to K so that:

<!-- formula-not-decoded -->

and the leading order term 2Trace(Σ K ′ ( K ′ -K ) glyph[latticetop] E K ) would then behave as Trace(( K ′ -K ) glyph[latticetop] ∇ C ( K )) .

Proof: The claim immediately results from Lemma 16.6, by using Equation 0.8 and taking an expectation.

We now use this cost difference lemma to show that C ( K ) is gradient dominated.

Lemma 16.8. (Gradient domination) Let K ∗ be an optimal policy. Suppose K has finite cost and µ &gt; 0 . It holds that:

<!-- formula-not-decoded -->

Proof: From Equation 0.8 and by completing the square,

<!-- formula-not-decoded -->

with equality when K ′ = K -( R + B glyph[latticetop] P K B ) -1 E K .

Let x ∗ t and u ∗ t be the sequence generated under K ∗ . Using this and Lemma 16.6,

<!-- formula-not-decoded -->

which completes the proof of first inequality. For the second inequality, observe:

<!-- formula-not-decoded -->

which completes the proof of the upper bound. Here the last step is because Σ K glyph[followsequal] E [ x 0 x glyph[latticetop] 0 ] .

The next lemma bounds the one step progress of Gauss-Newton.

Lemma 16.9. (Gauss-Newton Contraction) Suppose that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If η ≤ 1 , then

Proof: Observe that for PSD matrices A and B , we have that Trace( AB ) ≥ σ min ( A )Trace( B ) . Also, observe K ′ = K -η ( R + B glyph[latticetop] P K B ) -1 E K . Using Lemma 16.7 and the condition on η ,

<!-- formula-not-decoded -->

where the last step uses Lemma 16.8.

With this lemma, the proof of the convergence rate of the Gauss Newton algorithm is immediate.

Proof: (of Theorem 16.5, Gauss-Newton case) The theorem is due to that η = 1 leads to a contraction of 1 -µ ‖ Σ K ∗ ‖ at every step.

## 16.5 System Level Synthesis for Linear Dynamical Systems

We demonstrate another parameterization of controllers which admits convexity. Specifically in this section, we consider finite horizon setting, i.e.,

<!-- formula-not-decoded -->

Instead of focusing on quadratic cost function, we consider general convex cost function c ( x t , u t ) , and the goal is to optimize:

<!-- formula-not-decoded -->

where π is a time-dependent policy.

While we relax the assumption on the cost function from quadratic cost to general convex cost, we still focus on linearly parameterized policy, i.e., we are still interested in searching for a sequence of time-dependent linear controllers {-K glyph[star] t } H -1 t =0 with u t = -K glyph[star] t x t , such that it minimizes the above objective function. We again assume the system is controllable, i.e., the expected total cost under {-K glyph[star] t } H -1 t =0 is finite.

Note that due to the fact that now the cost function is not quadratic anymore, the Riccati equation we studied before does not hold , and it is not necessarily true that the value function of any linear controllers will be a quadratic function.

We present another parameterization of linear controllers which admit convexity in the objective function with respect to parameterization (recall that the objective function is non-convex with respect to linear controllers -K t ). With the new parameterization, we will see that due to the convexity, we can directly apply gradient descent to find the globally optimal solution.

Consider any fixed time-dependent linear controllers {-K t } H -1 t =0 . We start by rolling out the linear system under the controllers. Note that during the rollout, once we observe x t +1 , we can compute w t as w t = x t +1 -Ax t -Bu t . With x 0 ∼ µ , we have that at time step t :

<!-- formula-not-decoded -->

Note that we can equivalently write u t using x 0 and the noises w 0 , . . . , w t -1 . Hence for time step t , let us do the following re-parameterization. Denote

<!-- formula-not-decoded -->

Now we can express the control u t using M τ ; t for τ ∈ [0 , . . . , t -1] and M t as follows:

<!-- formula-not-decoded -->

which is equal to u t = -K t x t . Note that above we only reasoned time step t . We can repeat the same calculation for all t = 0 → H -1 .

The above calculation basically proves the following claim.

Claim 16.10. For any linear controllers π := {-K 0 , . . . , -K H -1 } , there exists a parameterization, ˜ π := {{ M t , M 0; t , . . . M t -1; t }} H -1 t =0 , such that when execute π and ˜ π under any initialization x 0 , and any sequence of noises { w t } H -1 t =0 , they generate exactly the same state-control trajectory.

We can execute ˜ π := {{ M t , M 0; t , . . . M t -1; t }} H -1 t =0 in the following way. Given any x 0 , we execute u 0 = M 0 x 0 and observe x 1 ; at time step t with the observed x t , we calculate w t -1 = x t -Ax t -1 -Bu t -1 , and execute the control u t = M t x 0 + ∑ t -1 τ =0 M τ ; t w τ . We repeat until we execute the last control u H -1 and reach x H .

What is the benefit of the above parameterization? Note that for any t , this is clearly over-parameterized: the simple linear controllers -K t has d × k many parameters, while the new controller { M t ; { M τ ; t } t -1 τ =0 } has t × d × k many parameters. The benefit of the above parameterization is that the objective function now is convex! The following claim formally shows the convexity.

Claim 16.11. Given ˜ π := {{ M t , M 0; t , . . . M t -1; t }} H -1 t =0 and denote its expected total cost as J ( ˜ π ) := E [ ∑ H -1 t =0 c ( x t , u t ) ] , where the expectation is with respect to the noise w t ∼ N (0 , σ 2 I ) and the initial state x 0 ∼ µ , and c is any convex function with respect to x, u . We have that J ( ˜ π ) is convex with respect to parameters {{ M t , M 0; t , . . . M t -1; t }} H -1 t =0 .

Proof: We consider a fixed x 0 and a fixed sequence of noises { w t } H -1 t =0 . Recall that for u t , we can write it as:

<!-- formula-not-decoded -->

which is clearly linear with respect to the parameterization { M t ; M 0; t , . . . M t -1; t } .

For x t , we can show by induction that it is linear with respect to { M τ ; M 0; τ , . . . M τ -1; τ } t -1 τ =0 . Note that it is clearly true for x 0 which is independent of any parameterizations. Assume this claim holds for x t with t ≥ 0 , we now check x t +1 . We have:

<!-- formula-not-decoded -->

Note that by inductive hypothesis x t is linear with respect to { M τ ; M 0; τ , . . . M τ -1; τ } t -1 τ =0 , and the part BM t x 0 + ∑ t -1 τ =0 BM τ ; t w τ is clearly linear with respect to { M t ; M 0; t , . . . M t -1; t } . Together, this implies that x t +1 is linear with respect to { M τ ; M 0; τ , . . . M τ -1; τ } t τ =0 , which concludes that for any x t with t = 0 , . . . , H -1 , we have x t is linear with respect to { M τ ; M 0; τ , . . . M τ -1; τ } H -1 τ =0 .

Note that the trajectory total cost is ∑ H -1 t =0 c ( x t , u t ) . Since c t is convex with respect to x t and u t , and x t and u t are linear with respect to { M τ ; M 0; τ , . . . M τ -1; τ } H -1 τ =0 , we have that ∑ H -1 t =0 c t ( x t , u t ) is convex with respect to { M τ ; M 0; τ , . . . M τ -1; τ } H -1 τ =0 .

In last step we can simply take expectation with respect to x 0 and w 1 , . . . , w H -1 . By linearity of expectation, we conclude the proof.

The above immediately suggests that (sub) gradient-based algorithms such as projected gradient descent on parameters { M t , M 0; t , . . . M t -1; t } H -1 t =0 can converge to the globally optimal solutions for any convex function c t . Recall that the best linear controllers {-K glyph[star] t } H -1 t =0 has its own corresponding parameterization { M glyph[star] t , M glyph[star] 0; t , . . . , M glyph[star] t -1; t } H -1 t =0 . Thus gradient-based optimization (with some care of the boundness of the parameters { M glyph[star] t , M glyph[star] 0; t , . . . , M glyph[star] t -1; t } H -1 t =0 ) can find a solution that at least as good as the best linear controllers {-K glyph[star] t } H -1 t =0 .

Remark The above claims easily extend to time-dependent transition and cost function, i.e., A t , B t , c t are timedependent. One can also extend it to episodic online control setting with adversarial noises w t and adversarial cost functions using no-regret online convex programming algorithms [Shalev-Shwartz, 2011]. In episodic online control, every episode k , an adversary determines a sequence of bounded noises w k 0 , . . . , w k H -1 , and cost function c k ( x, u ) ; the learner proposes a sequence of controllers ˜ π k = { M k t , M k 0; t , . . . , M k t -1; t } H -1 t =0 and executes them (learner does

not know the cost function c k until the end of the episode, and the noises are revealed in a way when she observes x k t and calculates w k t -1 as x k t -Ax k t -1 -Bu k t -1 ); at the end of the episode, learner observes c k and suffers total cost ∑ H -1 t =0 c k ( x k t , u k t ) . The goal of the episodic online control is to be no-regret with respect to the best linear controllers in hindsight:

<!-- formula-not-decoded -->

where we denote J k ( {-K glyph[star] t } H -1 t =0 ) as the total cost of executing {-K glyph[star] t } H -1 t =0 in episodic k under c k and noises { w k t } , i.e., ∑ H -1 t =0 c ( x t , u t ) with u t = -K glyph[star] t x t and x t +1 = Ax t + Bu t + w k t , for all t .

## 16.6 Bibliographic Remarks and Further Readings

The basics of optimal control theory can be found in any number of standards text [Anderson and Moore, 1990, Evans, 2005, Bertsekas, 2017]. The primal and dual formulations for the continuous time LQR are derived in Balakrishnan and Vandenberghe [2003], while the dual formulation for the discrete time LQR, that we us here, is derived in Cohen et al. [2019].

The treatment of Gauss-Newton, NPG, and PG algorithm are due to Fazel et al. [2018]. We have only provided the proof for the Gauss-Newton case based; the proofs of convergence rates for NPG and PG can be found in Fazel et al. [2018].

For many applications, the finite horizon LQR model is widely used as a model of locally linear dynamics, e.g. [Ahn et al., 2007, Todorov and Li, 2005, Tedrake, 2009, Perez et al., 2012]. The issue of instabilities are largely due to model misspecification and in the accuracy of the Taylor's expansion; it is less evident how the infinite horizon LQR model captures these issues. In contrast, for MDPs, practice tends to deal with stationary (and discounted) MDPs, due that stationary policies are more convenient to represent and learn; here, the non-stationary, finite horizon MDP model tends to be more of theoretical interest, due to that it is straightforward (and often more effective) to simply incorporate temporal information into the state. Roughly, if our policy is parametric, then practical representational constraints leads us to use stationary policies (incorporating temporal information into the state), while if we tend to use non-parametric policies (e.g. through some rollout based procedure, say with 'on the fly' computations like in model predictive control, e.g. [Williams et al., 2017]), then it is often more effective to work with finite horizon, non-stationary models.

Sample Complexity and Regret for LQRs. We have not treated the sample complexity of learning in an LQR (see [Dean et al., 2017, Simchowitz et al., 2018, Mania et al., 2019] for rates). Here, the basic analysis follows from the online regression approach, which was developed in the study of linear bandits [Dani et al., 2008, Abbasi-Yadkori et al., 2011]; in particular, the self-normalized bound for vector-valued martingales [Abbasi-Yadkori et al., 2011] (see Theorem A.9) provides a direct means to obtain sharp confidence intervals for estimating the system matrices A and B from data from a single trajectory (e.g. see [Simchowitz and Foster, 2020]).

Another family of work provides regret analyses of online LQR problems [Abbasi-Yadkori and Szepesv´ ari, 2011, Dean et al., 2018, Mania et al., 2019, Cohen et al., 2019, Simchowitz and Foster, 2020]. Here, naive random search suffices for sample efficient learning of LQRs [Simchowitz and Foster, 2020]. For the learning and control of more complex nonlinear dynamical systems, one would expect this is insufficient, where strategic exploration is required for sample efficient learning; just as in the case for MDPs (e.g. the UCB-VI algorithm).

Convex Parameterization of Linear Controllers The convex parameterization in Section 16.5 is based on [Agarwal et al., 2019] which is equivalent to the System Level Synthesis (SLS) parameterization [Wang et al., 2019]. Agar-

wal et al. [2019] uses SLS parameterization in the infinite horizon online control setting and leverages a reduction to online learning with memory. Note that in episodic online control setting, we can just use classic no-regret online learner such as projected gradient descent [Zinkevich, 2003]. For partial observable linear systems, the classic Youla parameterization[Youla et al., 1976] introduces a convex parameterization. We also refer readers to [Simchowitz et al., 2020] for more detailed discussion about Youla parameterization and the generalization of Youla parameterization. Moreover, using the performance difference lemma, the exact optimal control policy for adversarial noise with full observations can be exactly characterized Foster and Simchowitz [2020], Goel and Hassibi [2020] and yields a form reminiscent of the SLS parametrization Wang et al. [2019].

## Chapter 17

## Partially Observable Markov Decision Processes

To be added...

## Bibliography

- Yasin Abbasi-Yadkori and Csaba Szepesv´ ari. Regret bounds for the adaptive control of linear quadratic systems. In Conference on Learning Theory , pages 1-26, 2011.
- Yasin Abbasi-Yadkori, D´ avid P´ al, and Csaba Szepesv´ ari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , pages 2312-2320, 2011.
- Yasin Abbasi-Yadkori, Peter Bartlett, Kush Bhatia, Nevena Lazic, Csaba Szepesvari, and Gell´ ert Weisz. POLITEX: Regret bounds for policy iteration using expert prediction. In International Conference on Machine Learning , pages 3692-3702, 2019.
- Naoki Abe and Philip M. Long. Associative reinforcement learning using linear probabilistic concepts. In Proc. 16th International Conf. on Machine Learning , pages 3-11. Morgan Kaufmann, San Francisco, CA, 1999.
- Alekh Agarwal, Mikael Henaff, Sham Kakade, and Wen Sun. Pc-pg: Policy cover directed exploration for provable policy gradient learning. NeurIPS , 2020a.
- Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank mdps. NeurIPS , 2020b.
- Alekh Agarwal, Sham Kakade, and Lin F. Yang. Model-based reinforcement learning with a generative model is minimax optimal. In COLT , volume 125, pages 67-83, 2020c.
- Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift, 2020d.
- Naman Agarwal, Brian Bullins, Elad Hazan, Sham M Kakade, and Karan Singh. Online control with adversarial disturbances. arXiv preprint arXiv:1902.08721 , 2019.
- Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans, editors. Understanding the impact of entropy on policy optimization , 2019. URL https://arxiv.org/abs/1811.11214 .
- Hyo-Sung Ahn, YangQuan Chen, and Kevin L. Moore. Iterative learning control: Brief survey and categorization. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) , 37(6):1099-1121, 2007.
- Brian D. O. Anderson and John B. Moore. Optimal Control: Linear Quadratic Methods . Prentice-Hall, Inc., Upper Saddle River, NJ, USA, 1990. ISBN 0-13-638560-5.
- Andr´ as Antos, Csaba Szepesv´ ari, and R´ emi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning , 71(1):89-129, 2008.
- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Mach. Learn. , 47 (2-3):235-256, 2002. ISSN 0885-6125.

- Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pages 463-474. PMLR, 2020a.
- Alex Ayoub, Zeyu Jia, Csaba Szepesv´ ari, Mengdi Wang, and Lin F. Yang. Model-based reinforcement learning with value-targeted regression. arXiv preprint arXiv:2006.01107 , 2020b. URL https://arxiv.org/abs/2006. 01107 .
- Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J Kappen. Minimax pac bounds on the sample complexity of reinforcement lear ning with a generative model. Machine learning , 91(3):325-349, 2013.
- Mohammad Gheshlaghi Azar, Ian Osband, and R´ emi Munos. Minimax regret bounds for reinforcement learning. In Doina Precup and Yee Whye Teh, editors, Proceedings of Machine Learning Research , volume 70, pages 263-272, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR.
- J. Andrew Bagnell and Jeff Schneider. Covariant policy search. Proceedings of the 18th International Joint Conference on Artificial Intelligence , pages 1019-1024, 2003. URL http://dl.acm.org/citation.cfm?id= 1630659.1630805 .
- J Andrew Bagnell, Sham M Kakade, Jeff G Schneider, and Andrew Y Ng. Policy search by dynamic programming. In Advances in neural information processing systems , pages 831-838, 2004.
- V. Balakrishnan and L. Vandenberghe. Semidefinite programming duality and linear time-invariant systems. IEEE Transactions on Automatic Control , 48(1):30-41, 2003.
- Keith Ball. An elementary introduction to modern convex geometry. Flavors of geometry , 31:1-58, 1997.
- A. Beck. First-Order Methods in Optimization . Society for Industrial and Applied Mathematics, Philadelphia, PA, 2017. doi: 10.1137/1.9781611974997.
- Richard Bellman. Dynamic programming and Lagrange multipliers. Proceedings of the National Academy of Sciences , 42(10):767-769, 1956.
- Richard Bellman and Stuart Dreyfus. Functional approximations and dynamic programming. Mathematical Tables and Other Aids to Computation , 13(68):247-251, 1959.
- Dimitri P. Bertsekas. Dynamic Programming and Optimal Control . Athena Scientific, 2017.
- Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research , 3(Oct):213-231, 2002.
- Kiant´ e Brantley, Wen Sun, and Mikael Henaff. Disagreement-regularized imitation learning. In International Conference on Learning Representations , 2019.
- Kai-Wei Chang, Akshay Krishnamurthy, Alekh Agarwal, Hal Daume, and John Langford. Learning to search better than your teacher. In International Conference on Machine Learning , pages 2058-2066. PMLR, 2015.
- Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051, 2019.
- Ching-An Cheng and Byron Boots. Convergence of value aggregation for imitation learning. arXiv preprint arXiv:1801.07292 , 2018.
- Ching-An Cheng, Xinyan Yan, Nolan Wagener, and Byron Boots. Fast policy learning through imitation and reinforcement. arXiv preprint arXiv:1805.10413 , 2018.
- Ching-An Cheng, Andrey Kolobov, and Alekh Agarwal. Policy improvement from multiple experts. arXiv preprint arXiv:2007.00795 , 2020.

- Alon Cohen, Tomer Koren, and Yishay Mansour. Learning linear-quadratic regulators efficiently with only sqrtT regret. In International Conference on Machine Learning , pages 1300-1309, 2019.
- Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. In COLT , pages 355-366, 2008.
- Christoph Dann and Emma Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826, 2015.
- Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , pages 5713-5723, 2017.
- Hal Daum´ e, John Langford, and Daniel Marcu. Search-based structured prediction. Machine learning , 75(3):297-325, 2009.
- S. Dean, H. Mania, N. Matni, B. Recht, and S. Tu. On the sample complexity of the linear quadratic regulator. ArXiv e-prints , 2017.
- Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu. Regret bounds for robust adaptive control of the linear quadratic regulator. In Advances in Neural Information Processing Systems , pages 4188-4197, 2018.
- Luc Devroye and G´ abor Lugosi. Combinatorial methods in density estimation . Springer Science &amp; Business Media, 2012.
- Kefan Dong, Jian Peng, Yining Wang, and Yuan Zhou. Root-n-regret for learning in markov decision processes with function approximation and low bellman rank. In Conference on Learning Theory , pages 1554-1557. PMLR, 2020.
- Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations , 2019.
- Simon S. Du, Sham M. Kakade, Jason D. Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in RL. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , volume 139 of Proceedings of Machine Learning Research . PMLR, 2021.
- Lawrence C. Evans. An introduction to mathematical optimal control theory. University of California, Department of Mathematics , page 126, 2005. ISSN 14712334.
- Eyal Even-Dar, Sham M Kakade, and Yishay Mansour. Online Markov decision processes. Mathematics of Operations Research , 34(3):726-736, 2009.
- Maryam Fazel, Rong Ge 0001, Sham M. Kakade, and Mehran Mesbahi. Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator. In Proceedings of the 35th International Conference on Machine Learning , pages 1466-1475. PMLR, 2018.
- Dylan J Foster and Max Simchowitz. Logarithmic regret for adversarial online control. arXiv preprint arXiv:2003.00189 , 2020.
- Sara A Geer and Sara van de Geer. Empirical Processes in M-estimation , volume 6. Cambridge university press, 2000.
- Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. arXiv preprint arXiv:1901.11275 , 2019.
- Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization , 23(4):2341-2368, 2013.
- Peter W. Glynn. Likelihood ratio gradient estimation for stochastic systems. Commun. ACM , 33(10):75-84, 1990. ISSN 0001-0782.

- Gautam Goel and Babak Hassibi. The power of linear controllers in lqr control. arXiv preprint arXiv:2002.02574 , 2020.
- Botao Hao, Tor Lattimore, Csaba Szepesv´ ari, and Mengdi Wang. Online sparse reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 316-324. PMLR, 2021.
- Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Gabriel Dulac-Arnold, et al. Deep q-learning from demonstrations. arXiv preprint arXiv:1704.03732 , 2017.
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In Advances in neural information processing systems , pages 4565-4573, 2016.
- Daniel Hsu, Sham Kakade, and Tong Zhang. A spectral algorithm for learning hidden markov models. Journal of Computer and System Sciences , 78, 11 2008.
- Daniel J. Hsu, S. Kakade, and Tong Zhang. Random design analysis of ridge regression. Foundations of Computational Mathematics , 14:569-600, 2012.
- Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600, 2010.
- Zeyu Jia, Lin Yang, Csaba Szepesvari, and Mengdi Wang. Model-based reinforcement learning with value-targeted regression. Proceedings of Machine Learning Research , 120:666-686, 2020.
- Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E. Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In International Conference on Machine Learning , 2017.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 2137-2143, 2020.
- Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. arXiv preprint arXiv:2102.00815 , 2021.
- S. Kakade. A natural policy gradient. In NIPS , 2001.
- Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In Proceedings of the 19th International Conference on Machine Learning , volume 2, pages 267-274, 2002.
- Sham Machandranath Kakade. On the sample complexity of reinforcement learning . PhD thesis, University of College London, 2003.
- Liyiming Ke, Matt Barnes, Wen Sun, Gilwoo Lee, Sanjiban Choudhury, and Siddhartha Srinivasa. Imitation learning as f -divergence minimization. arXiv preprint arXiv:1905.12888 , 2019.
- Michael Kearns and Daphne Koller. Efficient reinforcement learning in factored mdps. In IJCAI , volume 16, pages 740-747, 1999.
- Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine Learning , 49 (2-3):209-232, 2002.
- Michael J Kearns and Satinder P Singh. Finite-sample convergence rates for q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002, 1999.
- Michael J. Kearns, Yishay Mansour, and Andrew Y. Ng. Approximate planning in large pomdps via reusable trajectories. In S. A. Solla, T. K. Leen, and K. M¨ uller, editors, Advances in Neural Information Processing Systems 12 , pages 1001-1007. MIT Press, 2000.

- Kris M Kitani, Brian D Ziebart, James Andrew Bagnell, and Martial Hebert. Activity forecasting. In European Conference on Computer Vision , pages 201-214. Springer, 2012.
- Tor Lattimore and Csaba Szepesv´ ari. Bandit algorithms . Cambridge University Press, 2020.
- Alessandro Lazaric, Mohammad Ghavamzadeh, and R´ emi Munos. Analysis of classification-based policy iteration algorithms. The Journal of Machine Learning Research , 17(1):583-612, 2016.
- Hoang M Le, Nan Jiang, Alekh Agarwal, Miroslav Dud´ ık, Yisong Yue, and Hal Daum´ e III. Hierarchical imitation and reinforcement learning. arXiv preprint arXiv:1803.00590 , 2018.
- Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909 , 2018.
- Sergey Levine and Pieter Abbeel. Learning neural network policies with guided policy search under unknown dynamics. In Advances in Neural Information Processing Systems , pages 1071-1079, 2014.
- Sergey Levine and Vladlen Koltun. Guided policy search. In International Conference on Machine Learning , pages 1-9, 2013.
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Breaking the sample size barrier in model-based reinforcement learning with a generative model. CoRR , abs/2005.12900, 2020.
- Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural proximal/trust region policy optimization attains globally optimal policy. CoRR , abs/1906.10306, 2019. URL http://arxiv.org/abs/1906.10306 .
- Thodoris Lykouris, Max Simchowitz, Aleksandrs Slivkins, and Wen Sun. Corruption robust exploration in episodic reinforcement learning. arXiv preprint arXiv:1911.08689 , 2019.
- Horia Mania, Stephen Tu, and Benjamin Recht. Certainty equivalent control of LQR is efficient. arXiv preprint arXiv:1902.07826 , 2019.
- Yishay Mansour and Satinder Singh. On the complexity of policy iteration. UAI , 01 1999.
- C. McDiarmid. On the method of bounded differences. In Surveys in Combinatorics , pages 148-188. Cambridge University Press, 1989.
- Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the global convergence rates of softmax policy gradient methods, 2020.
- Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937, 2016.
- Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. In International Conference on Artificial Intelligence and Statistics , pages 2010-2020. PMLR, 2020a.
- Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder P. Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. In The 23rd International Conference on Artificial Intelligence and Statistics, AISTATS , volume 108 of Proceedings of Machine Learning Research , 2020b.
- Aditya Modi, Jinglin Chen, Akshay Krishnamurthy, Nan Jiang, and Alekh Agarwal. Model-free representation learning and exploration in low-rank mdps. arXiv preprint arXiv:2102.07035 , 2021.
- R´ emi Munos. Error bounds for approximate policy iteration. In ICML , volume 3, pages 560-567, 2003.
- R´ emi Munos. Error bounds for approximate value iteration. In AAAI , 2005.

- Gergely Neu, Anders Jonsson, and Vicenc ¸ G´ omez. A unified view of entropy-regularized markov decision processes. CoRR , abs/1705.07798, 2017.
- Ian Osband and Benjamin Van Roy. Model-based reinforcement learning and the eluder dimension. In Proceedings of the 27th International Conference on Neural Information Processing Systems-Volume 1 , pages 1466-1474, 2014.
- Ian Osband and Benjamin Van Roy. On lower bounds for regret in reinforcement learning. ArXiv , abs/1608.02732, 2016.
- Alejandro Perez, Robert Platt, George Konidaris, Leslie Kaelbling, and Tomas Lozano-Perez. LQR-RRT*: Optimal sampling-based motion planning with automatically derived extension heuristics. In IEEE International Conference on Robotics and Automation , pages 2537-2542, 2012.
- Jan Peters and Stefan Schaal. Natural actor-critic. Neurocomput. , 71(7-9):1180-1190, 2008. ISSN 0925-2312.
- Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In Advances in neural information processing systems , pages 305-313, 1989.
- Martin Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley-Interscience, 1994.
- Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087 , 2017.
- H. Robbins. Some aspects of the sequential design of experiments. In Bulletin of the American Mathematical Society , volume 55, 1952.
- St´ ephane Ross and Drew Bagnell. Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 661-668, 2010.
- Stephane Ross and J Andrew Bagnell. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979 , 2014.
- St´ ephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics , pages 627-635, 2011.
- Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips. cc/paper/2013/file/41bfd20a38bb1b0bec75acf0845530a7-Paper.pdf .
- Bruno Scherrer. Approximate policy iteration schemes: a comparison. In International Conference on Machine Learning , pages 1314-1322, 2014.
- Bruno Scherrer and Matthieu Geist. Local policy search in a convex space and conservative policy iteration as boosted policy search. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 35-50. Springer, 2014.
- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning , pages 1889-1897, 2015.
- John Schulman, F. Wolski, Prafulla Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. ArXiv , abs/1707.06347, 2017.
- Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning , 4(2):107-194, 2011.

- Claude E. Shannon. Programming a computer playing chess. Philosophical Magazine , Ser.7, 41(312), 1959.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin F. Yang, and Yinyu Ye. Near-optimal time and sample complexities for for solving discounted markov decision process with a generative model. In Advances in Neural Information Processing Systems 31 , 2018.
- Max Simchowitz and Dylan J Foster. Naive exploration is optimal for online LQR. arXiv preprint arXiv:2001.09576 , 2020.
- Max Simchowitz, Horia Mania, Stephen Tu, Michael I Jordan, and Benjamin Recht. Learning without mixing: Towards a sharp analysis of linear system identification. In COLT , 2018.
- Max Simchowitz, Karan Singh, and Elad Hazan. Improper learning for non-stochastic control. arXiv preprint arXiv:2001.09254 , 2020.
- Satinder Singh and Richard Yee. An upper bound on the loss from approximate optimal-value functions. Machine Learning , 16(3):227-233, 1994.
- Wen Sun, Arun Venkatraman, Geoffrey J Gordon, Byron Boots, and J Andrew Bagnell. Deeply aggrevated: Differentiable imitation learning for sequential prediction. arXiv preprint arXiv:1703.01030 , 2017.
- Wen Sun, J Andrew Bagnell, and Byron Boots. Truncated horizon policy search: Combining reinforcement learning &amp;imitation learning. arXiv preprint arXiv:1805.11240 , 2018.
- Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In Conference on Learning Theory , pages 2898-2933, 2019a.
- WenSun, Anirudh Vemula, Byron Boots, and J Andrew Bagnell. Provably efficient imitation learning from observation alone. arXiv preprint arXiv:1905.10948 , 2019b.
- Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems , volume 99, pages 1057-1063, 1999.
- Csaba Szepesv´ ari and R´ emi Munos. Finite time bounds for sampling based fitted value iteration. In Proceedings of the 22nd international conference on Machine learning , pages 880-887. ACM, 2005.
- Russ Tedrake. LQR-trees: Feedback motion planning on sparse randomized trees. The International Journal of Robotics Research , 35, 2009.
- Michael J Todd. Minimum-Volume Ellipsoids: Theory and Algorithms , volume 23. SIAM, 2016.
- Emanuel Todorov and Weiwei Li. A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems. In American Control Conference , pages 300-306, 2005.
- Masatoshi Uehara, Xuezhou Zhang, and Wen Sun. Representation learning for online and offline rl in low-rank mdps, 2021.
- Ruosong Wang, Dean P. Foster, and Sham M. Kakade. What are the statistical limits of offline rl with linear function approximation?, 2020a.
- Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems , 33, 2020b.
- Yuanhao Wang, Ruosong Wang, and Sham M. Kakade. An exponential lower bound for linearly-realizable mdps with constant suboptimality gap. CoRR , abs/2103.12690, 2021. URL https://arxiv.org/abs/2103.12690 .

- Yuh-Shyang Wang, Nikolai Matni, and John C Doyle. A system-level approach to controller synthesis. IEEE Transactions on Automatic Control , 64(10):4079-4093, 2019.
- Gell´ ert Weisz, Philip Amortila, and Csaba Szepesv´ ari. Exponential lower bounds for planning in mdps with linearlyrealizable optimal action-value functions. In Algorithmic Learning Theory, 16-19 March 2021, Virtual Conference, Worldwide , volume 132 of Proceedings of Machine Learning Research , 2021a.
- Gell´ ert Weisz, Csaba Szepesv´ ari, and Andr´ as Gy¨ orgy. Tensorplan and the few actions lower bound for planning in mdps under linear realizability of optimal value functions, 2021b.
- Zheng Wen and Benjamin Van Roy. Efficient reinforcement learning in deterministic systems with value function generalization. Mathematics of Operations Research , 42(3):762-782, 2017.
- Grady Williams, Andrew Aldrich, and Evangelos A. Theodorou. Model predictive path integral control: From theory to parallel computation. Journal of Guidance, Control, and Dynamics , 40(2):344-357, 2017.
- Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256, 1992.
- Lin F Yang and Mengdi Wang. Reinforcement leaning in feature space: Matrix bandit, kernels, and regret bound. arXiv preprint arXiv:1905.10389 , 2019a.
- Lin F. Yang and Mengdi Wang. Sample-optimal parametric q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004, 2019b.
- Yinyu Ye. A new complexity result on solving the markov decision problem. Math. Oper. Res. , 30:733-749, 08 2005.
- Yinyu Ye. The simplex and policy-iteration methods are strongly polynomial for the markov decision problem with a fixed discount rate. Math. Oper. Res. , 36(4):593-603, 2011.
- Ming Yin, Yu Bai, and Yu-Xiang Wang. Near optimal provable uniform convergence in off-policy evaluation for reinforcement learning. ArXiv , abs/2007.03760, 2021.
- Dante Youla, Hamid Jabr, and Jr Bongiorno. Modern wiener-hopf design of optimal controllers-part ii: The multivariable case. IEEE Transactions on Automatic Control , 21(3):319-338, 1976.
- Andrea Zanette. Exponential lower bounds for batch reinforcement learning: Batch RL can be exponentially harder than online RL. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , Proceedings of Machine Learning Research. PMLR, 2021.
- Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning , pages 10978-10989. PMLR, 2020.
- Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted mdps with feature mapping. arXiv preprint arXiv:2006.13165 , 2020.
- Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory , pages 4532-4576. PMLR, 2021a.
- Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted mdps with feature mapping. In International Conference on Machine Learning , pages 12793-12802. PMLR, 2021b.
- Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In AAAI , pages 1433-1438, 2008.
- Brian D Ziebart, Nathan Ratliff, Garratt Gallagher, Christoph Mertz, Kevin Peterson, J Andrew Bagnell, Martial Hebert, Anind K Dey, and Siddhartha Srinivasa. Planning-based prediction for pedestrians. In 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems , pages 3931-3936. IEEE, 2009.

- Brian D Ziebart, J Andrew Bagnell, and Anind K Dey. Modeling interaction via the principle of maximum causal entropy. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) . Carnegie Mellon University, 2010.
- Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03) , pages 928-936, 2003.

## Appendix A

## Concentration

Lemma A.1. (Hoeffding's inequality) Suppose X 1 , X 2 , . . . X n are a sequence of independent, identically distributed (i.i.d.) random variables with mean µ . Let ¯ X n = n -1 ∑ n i =1 X i . Suppose that X i ∈ [ b -, b + ] with probability 1 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Chernoff bound implies that with probability 1 -δ :

<!-- formula-not-decoded -->

Now we introduce the definition of subGaussian random variables.

Definition A.2. [Sub-Gaussian Random Variable] A random variable X is σ -subGaussian if for all λ ∈ R , it holds that E [exp( λX )] ≤ exp ( λ 2 σ 2 / 2 ) .

One can show that a Gaussian random variable with zero mean and standard deviation σ is a σ -subGaussian random variable.

The following theorem shows that the tails of a σ -subGaussian random variable decay approximately as fast as that of a Gaussian variable with zero mean and standard deviation σ .

<!-- formula-not-decoded -->

The following lemma shows that the sum of independent sub-Gaussian variables is still sub-Gaussian.

Lemma A.4. Suppose that X 1 and X 2 are independent and σ 1 and σ 2 subGaussian, respectively. Then for any c ∈ R , we have cX being | c | σ -subGaussian. We also have X 1 + X 2 being √ σ 2 1 + σ 2 2 -subGaussian.

Theorem A.5 (Hoeffding-Azuma Inequality) . Suppose X 1 , . . . , X T is a martingale difference sequence where each X t is a σ t sub-Gaussian, Then, for all glyph[epsilon1] &gt; 0 and all positive integer N ,

<!-- formula-not-decoded -->

Similarly,

LemmaA.6. (Bernstein's inequality) Suppose X 1 , . . . , X n are independent random variables. Let ¯ X n = n -1 ∑ n i =1 X i , µ = E ¯ X n , and Var ( X i ) denote the variance of X i . If X i -EX i ≤ b for all i , then

<!-- formula-not-decoded -->

If all the variances are equal, the Bernstein inequality implies that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Lemma A.7 (Bernstein's Inequality for Martingales) . Suppose X 1 , X 2 . . . is a martingale difference sequence where | X i | ≤ M ∈ R + almost surely. Then for all positive t and n ∈ N + , we have:

<!-- formula-not-decoded -->

The following concentration bound is a simple application of the McDiarmid's inequality [McDiarmid, 1989] (e.g. see [Hsu et al., 2008] for proof).

Proposition A.8. (Concentration for Discrete Distributions) Let z be a discrete random variable that takes values in { 1 , . . . , d } , distributed according to q . We write q as a vector where glyph[vector] q = [Pr( z = j )] d j =1 . Assume we have N iid samples, and that our empirical estimate of glyph[vector] q is [ ̂ q ] j = ∑ N i =1 1 [ z i = j ] /N .

We have that ∀ glyph[epsilon1] &gt; 0 :

which implies that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.9 (Self-Normalized Bound for Vector-Valued Martingales; [Abbasi-Yadkori et al., 2011]) . Let { ε i } ∞ i =1 be a real-valued stochastic process with corresponding filtration {F i } ∞ i =1 such that ε i is F i measurable, E [ ε i |F i -1 ] = 0 , and ε i is conditionally σ -sub-Gaussian with σ ∈ R + . Let { X i } ∞ i =1 be a stochastic process with X i ∈ H (some Hilbert space) and X i being F t measurable. Assume that a linear operator Σ : H → H is positive definite, i.e., x glyph[latticetop] Σ x &gt; 0 for any x ∈ H . For any t , define the linear operator Σ t = Σ 0 + ∑ t i =1 X i X glyph[latticetop] i (here xx glyph[latticetop] denotes outer-product in H ). With probability at least 1 -δ , we have for all t ≥ 1 :

<!-- formula-not-decoded -->

Theorem A.10 (OLS with a fixed design) . Consider a fixed dataset D = { x i , y i } N i =1 where x i ∈ R d , and y i = ( θ glyph[star] ) glyph[latticetop] x i + glyph[epsilon1] i , where { glyph[epsilon1] i } N i =1 are independent σ -sub-Gaussian random variables, and E [ glyph[epsilon1] i ] = 0 . Denote Λ = ∑ N i =1 x i x glyph[latticetop] i , and we further assume that Λ is full rank such that Λ -1 exists. Denote ˆ θ as the least squares solution, i.e., ˆ θ = Λ -1 ∑ N i =1 x i y i . Pick δ ∈ (0 , 1) , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

The proof of the above theorem is standard (e.g. the proof can be easily adapted from Lemma 6 in [Hsu et al., 2012]).

Lemma A.11 (Least Squares Generalization Bound) . Given a dataset D = { x i , y i } n i =1 where x i ∈ X and x i , y i ∼ ν ,

y i = f glyph[star] ( x i ) + glyph[epsilon1] i , where | y i | ≤ Y, max x | f glyph[star] ( x ) | ≤ Y and | glyph[epsilon1] i | ≤ σ for all i , and { glyph[epsilon1] i } are independent from each other. Given a function class F : X ↦→ [0 , Y ] , we assume approximate realizable, i.e., min f ∈F E x ∼ ν | f glyph[star] ( x ) -f ( x ) | 2 ≤ glyph[epsilon1] approx . Denote ˆ f as the least square solution, i.e., ˆ f = argmin f ∈F ∑ n i =1 ( f ( x i ) -y i ) 2 . With probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

## Proof:

For notation simplicity, let us denote f glyph[star] ( x ) := E [ y | x ] . Let us consider a fixed function f ∈ F .

Denote random variables z f i := ( f ( x i ) -y i ) 2 -( f glyph[star] ( x i ) -y i ) 2 . We now show that E [ z f i ] = ‖ f -f glyph[star] ‖ 2 2 ,v , and E [( z f i ) 2 ] ≤ 4 Y 2 ‖ f -f glyph[star] ‖ 2 2 ,ν :

<!-- formula-not-decoded -->

where the last equality uses the fact that E [ y i | x i ] = f glyph[star] ( x i ) . For the second moment, we have:

<!-- formula-not-decoded -->

where the inequality uses the conditions that ‖ f glyph[star] ‖ ∞ ≤ Y, ‖ f ‖ ∞ ≤ Y, | y i | ≤ Y . Now we can bound the deviation from the empirical mean ∑ n i =1 z f i /n to its expectation ‖ f -f glyph[star] ‖ 2 2 ,ν using Bernstein's inequality. Together with a union bound over all f ∈ F , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Below we first bound ∑ n i =1 z ˆ f i /n where recall ˆ f is the least squares solution.

Denote ˜ f := argmin f ∈F ‖ f -f glyph[star] ‖ 2 2 ,ν . Note that ˜ f is independent of the dataset by its definition. Our previous calculation implies that E [ z ˜ f i ] = ‖ ˜ f -f glyph[star] ‖ 2 2 ,ν , and E [( z ˜ f i ) 2 ] ≤ 4 Y 2 ‖ ˜ f -f glyph[star] ‖ 2 2 ,ν . Directly applying Bernstein's inequality to bound ∑ n i =1 z ˜ f i /n -‖ ˜ f -f glyph[star] ‖ 2 2 ,ν , we have that with probability at least 1 -δ :

<!-- formula-not-decoded -->

Using this inequality, we discuss two cases. First when ∑ n i =1 z ˜ f i /n ≥ Y 2 ln(1 /δ ) /n . In this case, the above inequality implies that:

<!-- formula-not-decoded -->

Solve for ∑ n i =1 z ˜ f i /n , we get:

<!-- formula-not-decoded -->

For the other case where ∑ n i =1 z ˜ f i /n ≤ Y 2 ln(1 /δ ) /n , we immediately have:

<!-- formula-not-decoded -->

Thus ∑ n i =1 z ˜ f i /n ≤ 10 ‖ ˜ f -f glyph[star] ‖ 2 2 ,ν + 8 Y 2 ln(1 /δ ) 3 n regardless.

Since ‖ ˜ f -f glyph[star] ‖ 2 2 ,ν ≤ glyph[epsilon1] approx by the assumption, we can conclude that:

<!-- formula-not-decoded -->

Note that by definition of ˆ f , we have ∑ n i =1 z ˆ f i ≤ ∑ n i =1 z ˜ f i , which implies that:

<!-- formula-not-decoded -->

which concludes the upper bound of ∑ n i =1 z ˆ f i /n .

Going back to Eq. 0.1, by setting f = ˆ f there, and use the upper bound on ∑ n i =1 z ˆ f i /n , we have that:

<!-- formula-not-decoded -->

Solve for ‖ ˆ f -f glyph[star] ‖ 2 2 ,ν from the above inequality, we get:

<!-- formula-not-decoded -->

This concludes the proof.

Lemma A.12 (Uniform convergence for Bellman error under linear function hypothesis class ) . Given a feature φ : S × A ↦→ R d with ‖ φ ( s, a ) ‖ 2 ≤ 1 , ∀ s, a . Define H h = { w glyph[latticetop] φ ( s, a ) : ‖ w ‖ 2 ≤ W,w ∈ R d , ‖ w glyph[latticetop] φ ( · , · ) ‖ ∞ ≤ H } . Given g ∈ H , denote glyph[lscript] ( s, a, s ′ , g ) as follows:

<!-- formula-not-decoded -->

For any distribution ν ∈ ∆( S ) , with probability 1 -δ over the randomness of the m i.i.d triples { s i , a i , s ′ i } i m =1 with s i ∼ µ, a i ∼ Unif A , s ′ i ∼ P h ( ·| s i , a i ) , we have:

<!-- formula-not-decoded -->

Similarly, when glyph[lscript] ( s, a, s ′ , g ) is defined as:

<!-- formula-not-decoded -->

for any distribution ν ∈ ∆( S × A ) , with probability 1 -δ over randomness of the m i.i.d triples { s i , a i , s ′ i } i m =1 with s i , a i ∼ ν, s ′ i ∼ P h ( ·| s i , a i ) , we have:

<!-- formula-not-decoded -->

Proof: The proof uses the standard glyph[epsilon1] -net argument. Define N glyph[epsilon1] as the glyph[epsilon1] -net for W = { w : w ∈ R d , ‖ w ‖ 2 ≤ W,w glyph[latticetop] φ ( · , · ) ∈ H} . Standard glyph[epsilon1] -net argument shows that |N glyph[epsilon1] | ≤ (1 + W/glyph[epsilon1] ) d . Denote g ′ = { ( w ′ h ) glyph[latticetop] φ } H -1 h =0 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, if ‖ w h -w ′ h ‖ 2 ≤ glyph[epsilon1] , ‖ w h +1 -w ′ h +1 ‖ 2 ≤ glyph[epsilon1] , we have | glyph[lscript] ( s, a, s ′ , h, g ) -glyph[lscript] ( s, a, s ′ , h, g ′ ) | ≤ 2 Aglyph[epsilon1] .

Consider a w h ∈ N glyph[epsilon1] , w h +1 ∈ N glyph[epsilon1] , via Hoeffding's inequality, we have that with probability at least 1 -δ :

<!-- formula-not-decoded -->

where for notation simplicity, we denote ¯ E as the empirical average over the m data triples. Apply a union bound over all w h ∈ N glyph[epsilon1] and w h +1 ∈ N glyph[epsilon1] , we have that with probability at least 1 -δ :

<!-- formula-not-decoded -->

Thus for any w h ∈ W , w h +1 ∈ W , we have:

<!-- formula-not-decoded -->

Set glyph[epsilon1] = 1 / (4 Am ) , we have:

<!-- formula-not-decoded -->

This proves the first claim in the lemma.

The second claim in the lemma can be proved in a similar way, noting that without the importance weighting term on the loss glyph[lscript] , we have:

<!-- formula-not-decoded -->