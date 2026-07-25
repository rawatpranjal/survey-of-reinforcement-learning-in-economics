# Appendix A term ledger

Scope is the mathematical and reinforcement-learning theory vocabulary in the compiled survey. Application-specific vocabulary remains in its chapter. The appendix uses three treatment levels.

- `reference` means a compact definition in the opening table.
- `refresher` means equations plus the condition under which the object is valid.
- `result` means a stated theorem, proof, figure, and numerical check.

| Term family | Representative compiled use | Appendix treatment |
|---|---|---|
| maximum, argmax, supremum, infimum | `ch02_rl_algorithms/tex/rl_algorithms.tex:17` | reference |
| indicator | `ch02_rl_algorithms/tex/rl_algorithms.tex:33` | reference |
| return, value, action-value, advantage | `ch02_rl_algorithms/tex/rl_algorithms.tex:9` | reference |
| policy evaluation and control | `ch02_rl_algorithms/tex/rl_algorithms.tex:39` | reference |
| Bellman operator, fixed point, residual, value error | `ch02_rl_algorithms/tex/rl_algorithms.tex:39` | reference and refresher |
| temporal-difference error, bootstrapping, multi-step return | `ch02_rl_algorithms/tex/rl_algorithms.tex:26` | reference |
| on-policy and off-policy | `ch02_rl_algorithms/tex/rl_algorithms.tex:47` | reference |
| model-free and model-based | `ch03_theory/tex/planning_learning_v3.tex:241` | reference |
| function approximation and feature span | `ch03_theory/tex/planning_learning_v3.tex:428` | refresher |
| weighted inner product and projection | `ch03_theory/tex/planning_learning_v3.tex:430` | refresher and result |
| orthogonal and oblique projection | `ch03_theory/tex/planning_learning_v3.tex:493` | refresher and figure |
| normal equations and iterative linear solves | `ch02_rl_algorithms/tex/rl_algorithms.tex:132` | refresher |
| vector norm, operator norm, and nonexpansiveness | `ch03_theory/tex/planning_learning_v3.tex:430` | refresher and result |
| spectral radius and non-normal transient growth | `ch03_theory/tex/planning_learning_v3.tex:511` | result |
| Neumann series, resolvent, and effective horizon | `ch03_theory/tex/planning_learning_v3.tex:16` | result |
| stationary distribution, ergodicity, and mixing | `ch03_theory/tex/planning_learning_v3.tex:430` | result |
| Fisher matrix, natural gradient, and conditioning | `ch02_rl_algorithms/tex/rl_algorithms.tex:111` | refresher and figure |
| entropy and KL divergence | `ch02_rl_algorithms/tex/rl_algorithms.tex:268` | reference |
| trust region | `ch03_theory/tex/planning_learning_v3.tex:632` | reference and refresher |
| deadly triad | `ch02_rl_algorithms/tex/rl_algorithms.tex:89` | reference and refresher |
| asymptotic rate notation | `ch03_theory/tex/curse_of_dimensionality.tex:27` | reference |
| optimization, estimation, and approximation error | `ch03_theory/tex/planning_learning_v3.tex:305` | refresher |
| cumulative, average, and simple regret | `ch07_bandits/tex/dynamic_pricing.tex:3` | reference and refresher |
| sample complexity, PAC accuracy, and minimax rate | `ch03_theory/tex/planning_learning_v3.tex:241` | reference and refresher |
| occupancy measure | `ch03_theory/tex/planning_learning_v3.tex:550` | reference and refresher |
| coverage, support, and concentrability | `ch03_theory/tex/planning_learning_v3.tex:305` | reference, refresher, and figure |
| distribution shift | `ch08_offline_rl/tex/offline_rl.tex:39` | reference |
| importance ratios and change of measure | `ch10_causal/tex/causal_rl.tex:131` | reference and refresher |
| direct, importance-sampling, and doubly robust OPE | `ch13_field_deployments/tex/field_deployments.tex:409` | reference and refresher |
| positivity and sequential ignorability | `ch10b_rl_for_ci/tex/rl_for_ci.tex:21` | reference |
| nuisance functions, Neyman orthogonality, and cross-fitting | `ch10b_rl_for_ci/tex/rl_for_ci.tex:23` | reference and refresher |
| influence function and efficiency bound | `ch10b_rl_for_ci/tex/rl_for_ci.tex:23` | reference and refresher |
| Riesz representer | `ch10b_rl_for_ci/tex/rl_for_ci.tex:23` | reference and refresher |
| quantile and expectile | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:24` | reference and refresher |
| value at risk and conditional value at risk | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:132` | reference and refresher |
| coherent risk measure | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:104` | reference |
| ambiguity set and distributional robustness | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:311` | reference and refresher |
| rectangularity and robust Bellman recursion | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:460` | reference and refresher |
| constraint, Lagrange multiplier, and saddle point | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:588` | reference and result |
| world model, planning update, rollout, and model bias | `ch12_world_models/tex/s01_intro.tex:4` | reference |
| Jensen gap and maximization bias | `ch03_theory/tex/planning_learning_v3.tex:241` | result |
| law of large numbers and central limit theorem | `ch02_rl_algorithms/tex/rl_algorithms.tex:15` | result |
| martingale difference and almost-supermartingale | `ch03_theory/tex/planning_learning_v3.tex:196` | result |
| smoothness, strong convexity, and condition number | `ch03_theory/tex/planning_learning_v3.tex:592` | result |
| weak duality, strong duality, and Slater condition | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:588` | result |
| envelope theorem and Danskin derivative | `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:311` | result |
| Lipschitz map and contraction | `ch03_theory/tex/planning_learning_v3.tex:16` | result |
| Banach fixed-point theorem | `ch03_theory/tex/planning_learning_v3.tex:16` | result |
| Robbins-Monro conditions | `ch03_theory/tex/planning_learning_v3.tex:196` | result |

Excluded from the appendix are named economic models, market institutions, domain-specific engineering terms, and implementation vocabulary that is already defined at first use in its chapter.
