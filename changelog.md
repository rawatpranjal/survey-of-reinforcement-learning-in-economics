# Changelog

Reverse-chronological log of structural changes to the survey repo.

---

## 2026-01-28: Chapter 1 Restructure — Three-Part Historical Arc

**What:** Reorganized `ch01_history/tex/history.tex` into three main sections reflecting the field's development: Roots (pre-Sutton), The Classical Synthesis (Barto through Baird), and The Modern Era (Mnih onwards). Added new subsection on Barto, Sutton, and Anderson (1983). Elevated DQN to its own subsection with expanded content.

**New structure:**
- Section 1: Roots
  - 1.1 The Optimal Control Thread (Bellman 1957, Howard 1960)
  - 1.2 The Game Engines Thread (Minsky 1951, Samuel 1959)
  - 1.3 The Animal Psychology Thread (Thorndike 1898, Rescorla-Wagner 1972)
- Section 2: The Classical Synthesis
  - 2.1 Barto, Sutton, and Anderson (1983) — NEW: pole-balancing, ASE/ACE, first actor-critic
  - 2.2 Sutton (1988) — TD(λ) formalization
  - 2.3 Watkins (1989) — Q-learning
  - 2.4 Williams (1992) — REINFORCE
  - 2.5 Tesauro (1994) — TD-Gammon
  - 2.6 Baird (1995) — deadly triad
- Section 3: The Modern Era
  - 3.1 Deep Q-Networks (2015) — expanded with architecture details
  - 3.2 TRPO and PPO (2015, 2017)
  - 3.3 AlphaGo and AlphaGo Zero (2016, 2017)
  - 3.4 Toward a Unified Framework

**New content:** Barto, Sutton, Anderson (1983) subsection (~40 lines) covering the pole-balancing task, ASE (Associative Search Element) with stochastic action generation, ACE (Adaptive Critic Element) with TD-like internal reinforcement, eligibility traces for temporal credit assignment, and the actor-critic architecture that remains foundational in modern deep RL.

**Modified files:**
- `ch01_history/tex/history.tex` — full restructure with three `\section{}` divisions
- `changelog.md` — this entry

---

## 2026-01-28: Chapter 6 Restructure — Economic Bandits with 4 Simulations

**What:** Reorganized Chapter 6 from "Bandits and Online Learning" to "Economic Bandits" with a clear 8-subsection hierarchy and 4 new simulation scripts. The chapter thesis is that economic structure (unimodality, demand curves, budget constraints) buys exponential improvements in bandit regret, from O(sqrt(T)) agnostic to O(log T) structural.

**New structure:**
- 6.1 Introduction (bandits as stateless MDPs, regret, chapter thesis)
- 6.2 Classical Bandit Algorithms (epsilon-greedy, UCB1, Thompson Sampling, Lai-Robbins bound)
- 6.3 Dynamic Pricing with Demand Learning (unimodal revenue, Misra 2019, Mueller 2019, Xu 2021, Goyal 2022)
- 6.4 Bid Optimization in Auctions (second-price reserve, Myerson regularity, Cesa-Bianchi 2015, Akcay 2022)
- 6.5 Budget-Constrained Bandits (BwK, primal-dual, Flajolet 2017, Sankararaman 2018, Nuara 2018)
- 6.6 Strategic Interactions (multi-agent pricing, Nash convergence, Guo 2023)
- 6.7 Causal Inference and Confounding (IV, Proximal RL, Super RL)
- 6.8 Discussion (structural vs agnostic tradeoffs, open questions)

**New simulation scripts** (all in `ch06_bandits/sims/`):
- `bandit_fundamentals.py` — K=10 Gaussian bandit, 30 seeds, T=10,000. Compares epsilon-greedy, UCB1, Thompson Sampling with Lai-Robbins and minimax theoretical overlays.
- `dynamic_pricing_bandit.py` — K=20 prices, logistic demand, 30 seeds, T=50,000. Flagship simulation showing unimodal UCB achieves O(log T) vs UCB1 O(sqrt(T)).
- `auction_reserve_price.py` — 3 bidders LogNormal(0,0.5), K=25 reserves, 30 seeds, T=20,000. Compares UCB1 vs unimodal UCB against Myerson optimal.
- `bandits_with_knapsacks.py` — K=10 bid levels, budget B=0.3T, 30 seeds, T=10,000. BwK primal-dual vs unconstrained UCB vs proportional pacing, LP relaxation benchmark.

**Outputs:** 4 PNG figures (300 dpi) + 4 LaTeX results tables (.tex), all directly includable.

**New files:**
- `ch06_bandits/tex/economic_bandits.tex` — restructured chapter (201 lines)
- `ch06_bandits/sims/bandit_fundamentals.py`, `dynamic_pricing_bandit.py`, `auction_reserve_price.py`, `bandits_with_knapsacks.py`
- `ch06_bandits/tex/backups/2026-01-28_se_in_rl_full.tex` — archived original

**Modified files:**
- `docs/main.tex` — section title changed to "Economic Bandits", input path to `economic_bandits`
- `docs/refs.bib` — added `CombesProutiere2014` entry
- `claude.md` — chapter table and tasklist updated
- `changelog.md` — this entry

---

## 2026-01-28: Revise Chapter 2 — Draw More from Bertsekas

**What:** Rewrote `ch02_planning_learning/tex/planning_learning_theory.tex` (265 lines to ~310 lines) around Bertsekas' abstract DP framework. Removed the "Layer 1/2/3" organization; replaced with operator-property-driven structure organized into 7 subsections.

**New structure:**
- 2.1 The Abstract Dynamic Programming Framework (monotonicity + contraction as organizing principle)
- 2.2 The Four-Model Hierarchy (contractive, semicontractive, noncontractive, minimax)
- 2.3 From Operators to Algorithms (exact methods, stochastic approximation, GPI)
- 2.4 Approximation in Value Space (rollout, cost improvement, off-line/on-line, region of convergence)
- 2.5 Breaking the Curse of Dimensionality (function approximation, deadly triad)
- 2.6 Sample Complexity and the Model-Based Advantage (Q-learning vs model-based gap, policy gradient, actor-critic)
- 2.7 Stabilizing Approximate Dynamic Programming (error propagation, regularized operators, MCTS)

**New formal environments:**
- Definition: Monotone Operator (Sec 2.1)
- Definition: Proper Policy (Sec 2.2.2)
- Definition: S-Regular Policy (Sec 2.2.2)
- Definition: Rollout Algorithm (Sec 2.4)
- Definition: Generalized Policy Iteration (Sec 2.3)
- Theorem: Cost Improvement Property of Rollout (Sec 2.4)

**Content cut:** Value equivalence subsection and representation complexity removed; too application-specific for a theory chapter.

**Modified files:**
- `ch02_planning_learning/tex/planning_learning_theory.tex` — full rewrite
- `ch02_planning_learning/tex/backups/2026-01-28-planning_learning_theory.tex` — backup of previous version
- `changelog.md` — this entry

**Compilation:** Full document compiles cleanly (104 pages). No new undefined citations.

---

## 2026-01-28: Restructure to 9-Chapter Layout

**What:** Consolidated from 11 chapters (ch00-ch10) to 9 chapters (ch00-ch08). Removed the "RL as Economic Behaviour" chapter (IRL content belongs in ORE_main sister survey). Merged ch03_benchmarks and ch08_applications into new ch03_applications. Renumbered remaining chapters.

**New chapter structure:**

| Ch | Folder | Title |
|----|--------|-------|
| 0 | `ch00_introduction/` | Introduction |
| 1 | `ch01_history/` | Historical Developments |
| 2 | `ch02_planning_learning/` | Unified Theory of DP and RL |
| 3 | `ch03_applications/` | Applications of RL |
| 4 | `ch04_rl_econ_models/` | Solving Economic Models with RL |
| 5 | `ch05_rl_in_games/` | RL in Games |
| 6 | `ch06_bandits/` | Bandits and Online Learning |
| 7 | `ch07_rlhf/` | RLHF and Preference Learning |
| 8 | `ch08_conclusion/` | Conclusion |

**Folder operations:**
- Created `ch03_applications/` with merged content from ch03_benchmarks (benchmarks.tex + 9 sims) and ch08_applications
- Renamed `ch04_rl_structural_est/` -> `ch04_rl_econ_models/`
- Renamed `ch06_rl_in_games/` -> `ch05_rl_in_games/`
- Renamed `ch07_bandits/` -> `ch06_bandits/`
- Renamed `ch09_rlhf/` -> `ch07_rlhf/`
- Renamed `ch10_conclusion/` -> `ch08_conclusion/`
- Archived `ch05_rl_as_behaviour/` -> `archive/ch05_rl_as_behaviour/`
- Archived `ch03_benchmarks/` -> `archive/ch03_benchmarks/`
- Archived `ch08_applications/` -> `archive/ch08_applications/`

**Modified files:**
- `docs/main.tex` -- rewritten input block with new paths and section labels
- `ch03_applications/tex/applications.tex` -- updated internal paths from ch03_benchmarks to ch03_applications
- `claude.md` -- chapter table and tasklist updated
- `changelog.md` -- this entry

---

## 2026-01-28: Merge Ch08 into Ch03 as "Real World Applications of RL"

**What:** Merged ch08 (Applications prose) into ch03 (Benchmarks with simulations). Renamed combined chapter to "Real World Applications of RL." Each application subsection now pairs deployment literature with a formal MDP definition and benchmark simulation. Removed the original "Absence of Standardized Benchmarks" intro subsection and Definition 1 / algorithm battery table; replaced with a concise chapter-opening paragraph.

**New content:**
- Benchmark 7 (Financial Trading): JPMorgan LOXM deployment prose + new Order Execution MDP + `benchmark_execution.py` (TWAP, VWAP, Greedy heuristics; complexity dial T periods)
- Benchmark 8 (Operations/Energy): Google data center cooling prose + new Data Center Cooling MDP + `benchmark_datacenter.py` (Fixed, Load-proportional, PID heuristics; complexity dial N zones)
- Benchmarks 4, 5, 6, 9: prepended ch08 deployment prose (DiDi, Lyft, Gallego, Alibaba, Misra, Chen, Microsoft, Liu, RTB papers, Clark-Scarf, Gijsbrechts, vanHezewijk)
- Summary table expanded from 7 to 9 rows
- "Common Patterns and Open Challenges" section appended from ch08

**Modified files:**
- `ch03_benchmarks/tex/benchmarks.tex` — major rewrite (441 lines to ~640 lines)
- `ch03_benchmarks/sims/benchmark_execution.py` — new simulation script
- `ch03_benchmarks/sims/benchmark_datacenter.py` — new simulation script
- `docs/main.tex` — section title changed, ch08 \input removed, sections renumbered
- `ch08_applications/tex/applications.tex` — cleared (content merged into ch03)
- `claude.md` — chapter table updated
- `changelog.md` — this entry

---

## 2026-01-28: Expand Ch08 Applications with Ch03 Papers + Rename

**What:** Renamed section to "Reinforcement Learning for Real World Decision Making" (in main.tex) and expanded `ch08_applications/tex/applications.tex` from 42 lines to a full survey chapter by integrating application-domain papers originally cited in ch03 benchmarks.

**New content:**
- Transportation: added Qin2021dispatch (formal DiDi paper, replaces informal DiDi2019 as primary cite), Li2019ridesharing (mean-field MARL for fleet coordination), Han2022lyft (Lyft deployment)
- Dynamic Pricing: added Gallego1994pricing (classical intensity control baseline), Chen2023hotelrl (RL for hotel revenue management)
- Advertising: added RTB paragraph with Cai2017rtb (model-based RTB), Wu2018budgetbidding (budget-constrained bidding), Guo2024diffbid (diffusion-based bidding)
- NEW subsection "Supply Chain and Inventory Management": ClarkScarf1960inventory (classical multi-echelon theory), Gijsbrechts2022inventory (deep RL evaluation, mixed results), vanHezewijk2023inventory (multi-echelon PPO)

**Modified files:**
- `ch08_applications/tex/applications.tex` — expanded from 42 to ~80 lines with 11 new citations
- `docs/main.tex` — section title already renamed to "Reinforcement Learning for Real World Decision Making"
- `claude.md` — updated ch08 row in master plan table
- `changelog.md` — this entry

**No new bib entries needed:** all 11 papers were already in refs.bib from the ch03 benchmarks expansion.

---

## 2026-01-28: Rename Ch05 to "RL as Economic Behaviour"

**What:** Renamed `ch05_inverse_rl/` to `ch05_rl_as_behaviour/` with new focus: RL as a descriptive model of how economic agents learn and behave, not RL as an algorithm to solve models.

**Renames:**
- `ch05_inverse_rl/` -> `ch05_rl_as_behaviour/`

**New files:**
- `ch05_rl_as_behaviour/tex/rl_as_behaviour.tex` -- new chapter tex with seven subsections: From Animal Learning to Economic Agents, Reinforcement Learning and Replicator Dynamics, Models of Human Learning in Games (Roth-Erev, EWA, Fudenberg-Levine), Neuroscience Foundations (Schultz-Dayan-Montague 1997), Algorithmic Behaviour and No-Regret Learning (Calvano 2020, Lomys-Magnolfi 2024), Inferring Preferences from Observed Behaviour (reframed IRL via `\input` of existing irl.tex), Bounded Rationality and Satisficing (Simon 1957)

**Modified files:**
- `ch06_rl_in_games/tex/marl.tex` -- removed replicator dynamics section (lines 3-77), moved to ch05; added transition sentence
- `docs/main.tex` -- added `\input{../ch05_rl_as_behaviour/tex/rl_as_behaviour}` between ch04 and ch06
- `docs/refs.bib` -- added 8 new entries: RothErev1995, ErevRoth1998, CamererHo1999, FudenbergLevine1998, SchultzDayanMontague1997, Calvano2020, TaylorJonker1978, Simon1957
- `claude.md` -- updated ch05 row in master plan table
- `changelog.md` -- this entry

**Design decisions:**
- Existing `irl.tex` kept in place and included via `\input` from new chapter file; reframed as "Inferring Preferences from Observed Behaviour" subsection
- Replicator dynamics content rewritten (not copy-pasted) for new chapter context; ch06 now opens with competitive MARL algorithms
- New chapter follows the arc: historical roots -> formal dynamics -> experimental evidence -> neuroscience -> algorithmic markets -> IRL -> bounded rationality

---

## 2026-01-28: Chapter 4 Rewrite — RL for Structural Estimation (Fleshed Out)

**What:** Rewrote `ch04_rl_structural_est/tex/rl_in_se.tex` from ~4 pages to ~15 pages, focusing exclusively on papers that use genuine RL algorithms (Q-learning, TD, policy gradient, DDPG, PPO) to solve or estimate structural economic models. Added unified notation (MDP tuple, consistent $\gamma$ for discount factor, $r(s,a)$ for reward). Each paper now gets a full treatment: problem statement, RL method with equations, key result, and limitation.

**New structure:**
- Introduction (unified notation, scope)
- Dynamic Discrete Choice Estimation (Adusumilli & Eckardt 2022, Hu & Yang 2025)
- Dynamic Oligopoly and Strategic Interaction (Asker et al. 2020, Hollenbeck 2019, with Fershtman & Pakes 2012 as precursor footnote and Lomys & Magnolfi 2024 / Covarrubias 2022 as brief mentions)
- Auction Equilibria and Mechanism Design (Graf et al. 2025, Brero et al. 2021, Ravindranath et al. 2024)
- Macroeconomic Models (Atashbar & Shi 2023, Curry et al. 2022, Hinterlang & Tänzer 2024)
- Assessment (five cross-cutting themes)

**Demoted/removed:** Fershtman & Pakes 2012 (not RL proper; now footnote), Lomys & Magnolfi 2024 (estimation method is standard econometrics; brief mention), Covarrubias 2022 (unclear RL loop; brief mention).

**Modified files:**
- `ch04_rl_structural_est/tex/rl_in_se.tex` — full rewrite
- `docs/refs.bib` — added RustRawat2026, PyciaAndTroyan2019 bib entries

**Backup:** `ch04_rl_structural_est/tex/backups/2026-01-28-rl_in_se.tex`

---

## 2026-01-28: Chapter 3 Expansion — Four New Economic Benchmarks

**What:** Expanded the Chapter 3 benchmark suite from 3 to 7 environments, covering the major economic decision archetypes. Created a shared abstraction layer for all benchmarks.

**New files:**
- `ch03_benchmarks/sims/econ_benchmark.py` — shared base class (EconBenchmark ABC), QNetwork, ReplayBuffer, generic value iteration, DQN training, and evaluation utilities
- `ch03_benchmarks/sims/benchmark_dispatch.py` — zone-based ride dispatch (matching/allocation archetype, DiDi-inspired)
- `ch03_benchmarks/sims/benchmark_hotel_rm.py` — hotel revenue management (dynamic pricing with perishable inventory)
- `ch03_benchmarks/sims/benchmark_rtb_bidding.py` — real-time bidding with budget pacing (constrained sequential bidding)
- `ch03_benchmarks/sims/benchmark_inventory.py` — multi-echelon inventory management (quantity choice with supply chain coupling)

**Modified files:**
- `ch03_benchmarks/tex/benchmarks.tex` — added 4 new subsections (Benchmarks 4-7), summary table of all 7 benchmarks, updated intro paragraph count
- `docs/refs.bib` — added 12 new bibliography entries (Qin2021, Li2019, Han2022, Gallego1994, Chen2023, Cai2017, Wu2018, Guo2024, ClarkScarf1960, Gijsbrechts2022, vanHezewijk2023)
- `claude.md` — updated ch03 row in master plan table
- `changelog.md` — this entry

**Benchmark suite summary:**

| # | Benchmark | Archetype | Complexity Dial |
|---|-----------|-----------|----------------|
| 1 | Gridworld | Verification | Grid size N |
| 2 | Bus Engine | Curse of dimensionality | Fleet size N |
| 3 | Discount Targeting | Dynamic vs static | State components |
| 4 | Optimal Dispatch (NEW) | Matching/allocation | Zones K |
| 5 | Hotel Revenue Mgmt (NEW) | Pricing | Capacity C |
| 6 | RTB Budget Pacing (NEW) | Constrained bidding | Periods T |
| 7 | Multi-Echelon Inventory (NEW) | Quantity choice | Echelons K |

**Design decisions:**
- Existing scripts (gridworld, bus engine, discount targeting) left unchanged; new scripts import from shared econ_benchmark.py
- Each new benchmark follows the same structure: environment class implementing EconBenchmark ABC, exact DP for small instances, DQN with 3 seeds, 3 domain-specific heuristics, two-panel scaling figure, LaTeX results table

---

## 2026-01-27: Chapter 2 Paper Integration — Strengthen Theory with 67 Paper Summaries

**What:** Read all 67 paper summaries in `ch02_planning_learning/papers/`, transcribed the remaining PDF (`NDP.pdf` via docling), and integrated findings into `ch02_planning_learning/tex/planning_learning_theory.tex`.

**New content added to `planning_learning_theory.tex`:**
- Strengthened Layer 2 (Stochastic Approximation): expanded Borkar ODE method discussion
- Strengthened Layer 3 (The Continuum): added Bertsekas 2024 unified DP framework, MBPO rollout analysis
- Strengthened Convergence section: added He 2021 discounted MDP regret, Mitra 2024 TD bounds, Lee 2025 average-reward VI, Mao 2021 non-stationary MDPs
- Strengthened Function Approximation section: added Jin 2020 LSVI-UCB, Zhu 2024 representation complexity, Scherrer 2015 approximate MPI
- NEW subsection: Policy Gradient Theory (Sutton et al. 2000, Agarwal 2021, Xiao 2022, Cen 2022, Müller 2024, Wu 2020, Tian 2023)
- NEW subsection: Regularized Operators (Geist 2019, Vieillard 2020, Lim 2024, Grill 2020 MCTS-as-regularized-optimization)
- NEW subsection: Offline RL and Distribution Shift (Li 2024, Shi 2024, Agarwal et al. 2020)
- NEW subsection: Continuous-Time Formulations (Wang 2020, Tang 2023, Jia 2022, Kim 2020)
- Expanded Implications for Structural Econometrics: from 5 to 7 paragraphs (added policy gradient estimation, regularization as prior, offline RL for counterfactuals)

**Bibliography:** Added `wang2020continuous` entry to `docs/refs.bib`. All other cited papers already had entries.

**Transcription:** `NDP.pdf` (Bertsekas & Tsitsiklis, Neuro-Dynamic Programming) transcribed to `ch02_planning_learning/papers/NDP.md` (280s via docling).

**Compilation:** Full document compiles cleanly (86 pages). Zero undefined citations.

**Also fixed:** Updated `ch03_benchmarks/tex/benchmarks.tex` citation keys (`silver2016mastering` → `Silver2016`, `rust1987` → `Rust1987`). Created placeholder figures and tables in `ch03_benchmarks/sims/` for compilation.

---

## 2026-01-27: Renumber Chapter Directories to Match Logical Order

**What:** Renamed chapter directories so folder numbers match the logical chapter order established when ch03_benchmarks was inserted. Previously ch03_rl_structural_est through ch09_conclusion used old numbering; now they are ch04 through ch10.

**Renames (bottom-up to avoid conflicts):**
- `ch09_conclusion/` -> `ch10_conclusion/`
- `ch08_rlhf/` -> `ch09_rlhf/`
- `ch07_applications/` -> `ch08_applications/`
- `ch06_bandits/` -> `ch07_bandits/`
- `ch05_rl_in_games/` -> `ch06_rl_in_games/`
- `ch04_inverse_rl/` -> `ch05_inverse_rl/`
- `ch03_rl_structural_est/` -> `ch04_rl_structural_est/`

**Updated:**
- `docs/main.tex`: all 7 `\input` paths updated to new directory names
- `claude.md`: master plan table folder column and tasklist chapter references updated
- `changelog.md`: this entry

**Unchanged:** ch00_introduction, ch01_history, ch02_planning_learning, ch03_benchmarks (already correctly numbered).

---

## 2026-01-27: Chapter 2-3 Restructure (Planning-Learning Continuum)

**What:** Split old Chapter 2 into two chapters. Chapter 2 becomes pure operator theory. Chapter 3 becomes a benchmark suite of economic environments testing RL against classical methods. All subsequent chapters shift down by one.

**New Chapter 2 (The Operator Foundation):**
- Created `ch02_planning_learning/tex/planning_learning_theory.tex`
- Theory-only: Bellman operator layers (exact, stochastic approx, continuum), extensions (convergence, function approx, value equivalence), econometrics implications
- No examples or simulations; those moved to Chapter 3

**New Chapter 3 (Economic Benchmarks):**
- Created `ch03_benchmarks/` directory with `tex/`, `sims/`, `papers/`, `notes/`
- Created `ch03_benchmarks/tex/benchmarks.tex` with framing, algorithm battery, three benchmarks
- Created `ch03_benchmarks/sims/benchmark_gridworld.py` (adapted from ch02 gridworld)
- Created `ch03_benchmarks/sims/benchmark_bus_engine.py` (adapted from ch02 bus engine)
- Created `ch03_benchmarks/sims/benchmark_discount_targeting.py` (NEW environment)
- Saved heuristic decision rules notes to `ch03_benchmarks/notes/heuristic_decision_rules.md`

**Archived:**
- `ch02_planning_learning/tex/planning_learning.tex` -> `tex/backups/2026-01-27-120000_planning_learning.tex`
- `ch02_planning_learning/tex/planning_learning_alt.tex` -> `tex/backups/2026-01-27-120000_planning_learning_alt.tex`

**Updated:**
- `docs/main.tex`: new chapter includes (planning_learning_theory, benchmarks), renumbered sections
- `CLAUDE.md`: master plan table updated with new chapter structure
- `changelog.md`: this entry

**Chapter renumbering:** Old Ch2 -> Ch2 (theory) + Ch3 (benchmarks, NEW). Old Ch3-Ch9 -> Ch4-Ch10.

---

## 2026-01-27: Initial Restructure Plan

**What:** Designed chapter-based repo structure for "Survey of Reinforcement Learning in Economics" arXiv paper.

**Chapters defined:**
- ch00: Introduction
- ch01: Historical Developments
- ch02: Planning and Learning (DP vs RL)
- ch03: RL for Structural Estimation
- ch04: Inverse RL
- ch05: RL in Games (MARL)
- ch06: Bandits & Online Learning
- ch07: Applications
- ch08: RLHF & Preference Learning
- ch09: Conclusion

**Decisions made:**
- Each chapter gets `tex/`, `papers/`, `sims/` subdirs
- RLHF added as ch08 (was cut from ORE project)
- Bandits split out as separate chapter from old "Economic Models for RL" section
- "Economic Models for RL" chapter placement deferred
- Old repo dirs (`qlearning/`, `ppo/`, `md/`, etc.) archived
- `docs/` holds combined LaTeX main file
- Sister survey ORE_main referenced but not duplicated
- Created `claude.md` with writing style, notation, simulation standards

**Files created:**
- `claude.md` -- project context and style guide
- `changelog.md` -- this file
- `docs/plans/2026-01-27-repo-restructure.md` -- detailed implementation plan

**Pending:** Execute the 10-task restructure plan (directory skeleton, file distribution, archiving, main.tex rewrite, .gitignore, README).
