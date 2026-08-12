# Chapter 12 Humanization Ledger

Date: 2026-08-12

Scope: `ch12_world_models/tex/`

This pass targeted the most conspicuous AI-style prose in the world-model chapter. Drafting agents worked on one bounded subsection at a time. Each agent read the claim-bearing primary sources and returned exact replacements without editing files. A separate verifier then read the source material and ruled on every replacement before the main thread applied it manually.

## Coverage

The pass evaluated 186 replacement units across fifteen chunks and a final whole-chapter gate.

| Wave | Subsections | Replacement units |
|---|---|---:|
| 1 | Economic origins, Dyna origins, Dyna mechanics | 21 |
| 2 | Dyna simulation, Ha and Schmidhuber, RSSM and Dreamer | 45 |
| 3 | MBPO, value-aware objectives, MuZero | 36 |
| 4 | TD-MPC2, Cobweb design, Cobweb results | 39 |
| 5 | Fishery, supply chain, synthesis | 42 |
| Final | Residual Cobweb framing, glossary construction, and demand-learning scope | 3 |

## Main prose changes

- Replaced internal chapter narration, genealogical scaffolding, personified arguments, and ranking language with direct descriptions of models, estimators, planners, and results.
- Removed broad claims that model-based methods win under scarce interaction, that TD-MPC2 is a convergence point, and that individual simulations establish universal method rankings.
- Replaced causal explanations that were not isolated by the simulations with the observed quantities and the relevant design limitations.
- Rewrote figure and table captions to identify the displayed objects without interpreting or promoting the results.
- Removed internal implementation language from reader-facing prose, including cache, script, module, and pipeline references.

## Material source corrections discovered during the prose pass

- Esponda and Pouzo's Berk-Nash result is conditional on stabilized behavior and posterior concentration near weighted-divergence minimizers. The prior draft stated unconditional convergence to a wrong and suboptimal policy.
- The local `sutton1990_dyna.pdf` file is the 1991 SIGART article although the bibliography key refers to the 1990 ICML paper. Exact blocking-maze results were checked against Sutton and Barto (2018).
- Schmidhuber's 1990 world model is a deterministic recurrent predictor trained with squared error, not a stochastic negative-log-likelihood model.
- The VAML display was corrected to place the supremum inside the state-action integral, matching Farahmand's objective.
- The Asadi value bound uses a Markov reward process, a Lipschitz reward, and $\bar K=\min\{K_F,K_T\}$. Failure of $\gamma\bar K<1$ does not prove that planning diverges.
- TD-MPC2 trains its policy prior with a separate maximum-entropy objective, not behavioral cloning, and runs its planner for a fixed number of iterations.
- Arifovic's election operator uses the previous market price and the cost function. It does not require the true demand parameters.
- The supply-chain benchmark uses simulation-based coordinate search over a Clark-Scarf policy class. Its reported cost is a reference value, not a certified global optimum.
- Voelcker et al. correct a variance term in sampled stochastic model losses but identify a separate MuZero value-update bias that the correction does not remove.
- Madeka et al. retain controlled inventory dynamics in a differentiable simulator. Their supervised term concerns policy-value estimation and imputation of partially observed exogenous variables, not a one-shot replacement for sequential control.

## Simulation and source limitations retained in the record

- The Cobweb and Fishery audit markdown files are stale relative to their current scripts and artifacts. Verification used the scripts, matching caches where available, stdout, generated tables, figures, and primary papers.
- The current checkout does not contain the supply-chain cache files. Its checked-in script, PNG, table, stdout, and audit have matching timestamps and internally consistent numbers, but the numerical results could not be regenerated or hash-checked from a local cache. The stdout records an earlier checkout path.
- Several targeted tests could not collect in the active Python environment because optional packages such as `matplotlib` and `torch` are absent. These were treated as unexecuted tests, not passes.
- The Donti paper cited in the MuZero footnote was absent from the repository. A verifier read the corresponding arXiv paper for the claim check, but the source gap remains in the chapter's local paper collection.

## Final gate

The whole-chapter verifier found no accidental movement of equations, numbers, citations, labels, chronology, method identities, or comparison scope. It identified three residual corrections, comprising the Cobweb fixed-action comparator, the glossary construction, and the DQN demand-learning claim. All three were applied as exact replacements. The standalone chapter compiles to 39 pages. Visual inspection covered the TD-MPC2, Cobweb, Fishery, supply-chain, glossary, and synthesis pages.
