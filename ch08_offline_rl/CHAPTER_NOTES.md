# ch08 — Offline Reinforcement Learning

Project notes for the Offline RL chapter. The chapter covers the pessimism principle, four pessimism-based algorithms (FQI, CQL, IQL, BCQ), the supervised-conditioning family (DT, RvS), and an empirical comparison on a perishable inventory pricing MDP. This file is the working memory for the chapter and is intended to let a fresh session pick up without re-deriving context.

## Identity

- **Directory:** `ch08_offline_rl/`.
- **Master tex file:** `tex/offline_rl.tex` (monolithic — no per-section files).
- **Section title in `docs/main.tex`:** `\section{Offline Reinforcement Learning}` with `\label{section:offline_rl}`, included at line 193.
- **Standalone compile:** `cd docs && pdflatex -shell-escape -jobname=ch08_offline_rl "\def\chapterfile{../ch08_offline_rl/tex/offline_rl}\input{compile_chapter}"` (twice with `bibtex ch08_offline_rl` between). Output PDF: `docs/ch08_offline_rl.pdf`.
- **Monograph compile:** standard `cd docs && pdflatex -shell-escape main && bibtex main && pdflatex -shell-escape main && pdflatex -shell-escape main`.

## Spine

| § | Topic |
|---|---|
| Intro | Offline RL problem statement, distributional shift, behavior policy framing |
| 1 | The Pessimism Principle — PEVI, concentrability, impossibility results |
| 2 | Algorithms — FQI, CQL, IQL, BCQ as four instantiations of pessimism |
| 3 | Trajectory Models and Return-Conditioned Supervised Learning — DT (Chen 2021) and RvS (Emmons 2022) as the supervised-conditioning alternative to pessimism; Brandfonbrener 2022 caveat in footnote |
| 4 | Simulation Study: Offline RL for Dynamic Pricing — perishable inventory MDP, 7 trained methods + DP oracle, coverage-sensitivity figure |

The §3 subsection was added 2026-05-18 during the DT/RvS migration from `ch12_world_models/tex/v3_archived/05_forecaster_as_policy.tex`. The simulation in §4 was extended at the same time to include DT and RvS rows in the rank-ordered comparison table.

## Simulation

One sim, `sims/offline_rl_pricing.py`. Uses the standard `sims.sim_cache.compute_or_load` per-paradigm caching pattern (matches ch12 cobweb/fishery sims).

### Environment

- **MDP.** Perishable inventory pricing. State $(i, d, t)$ with inventory $i \in \{0, \ldots, 30\}$, demand regime $d \in \{0, 1, 2, 3\}$, time remaining $t \in \{0, \ldots, 20\}$. Action $p \in \{1, \ldots, 10\}$.
- **Demand.** $Q \sim \text{Poisson}(\lambda_0[d] \cdot e^{-0.15 p})$, $\lambda_0 = (1.5, 3.0, 5.0, 8.0)$.
- **Reward.** $r = p \cdot \min(Q, i)$ during the episode; terminal salvage cost $-2.00$ per unsold unit.
- **Behavioral policy.** Maximum price $p = 10$ with prob $0.85$, uniform random with prob $0.15$.
- **DP oracle.** Exact backward induction on the tabular MDP.

### Methods (7 trained + DP oracle)

The current rank order is computed at table-generation time from the per-seed means. Expected band by family:

- **DP Oracle** — exact upper bound, 100%.
- **CQL, IQL** — pessimism-based, expected above BC.
- **BCQ** — action-constrained, expected ≈ BC.
- **BC** — behavioral cloning baseline.
- **FQI** — unconstrained Bellman, expected below BC due to overestimation cascade.
- **DT, RvS** — return-conditioned supervised; expected middle band, possibly trailing pessimism methods on this small env (per Brandfonbrener 2022 caveat).

### Outputs

- `offline_rl_pricing_results.tex` — rank-ordered table (all 8 rows including DP Oracle).
- `offline_rl_pricing_coverage.png` — coverage-sensitivity figure with 7 method lines + DP-optimal reference.
- `offline_rl_pricing_stdout.txt` — run log.
- `offline_rl_pricing_audit.md` — 7-point audit per `feedback_one_sim_at_a_time.md`.

## Conventions

- **Per-paradigm `compute_or_load` caching.** Cache keys: `'shared'`, `'DP_Oracle'`, `'BC'`, `'FQI'`, `'CQL'`, `'IQL'`, `'BCQ'`, `'DT'`, `'RvS'`, plus `'coverage_<method>'` for each. Changing one method's hyperparameters invalidates that method only.
- **Same dataset across methods.** `compute_shared()` generates 20 seed-keyed offline datasets at the default behavioral noise (0.15). All 7 methods consume the same datasets.
- **Return-conditioning protocol for DT and RvS.** Target return $R^\star = \text{dp\_init\_val}$ (oracle return at start state). This is an extrapolation request and the strongest stress test of return-conditioning.
- **Coverage experiment.** Each method evaluated at $\epsilon_b \in \{0.05, 0.3, 0.9\}$ on fresh per-eps datasets. Datasets are seed-deterministic so methods see the same data at each eps level.
- **Rank-ordered table and figure.** Methods sorted by mean return descending, with DP Oracle pinned to row 1. Implemented in `_rank_ordered()` in the sim script.
- **No `\textbf{}` in body prose**, no em dashes, no colons in prose, no `\paragraph*{}` headers, no bullet points outside numbered lists. Same conventions as the rest of the monograph.

## Bibliography keys (verified resolved)

- Foundations: `Levine2020`, `Fujimoto2019`, `Ernst2005`, `Lange2012`.
- Pessimism theory: `JinYang2021`, `Munos2008`, `Rashidinejad2021`, `Zanette2021`.
- Algorithms: `Kumar2020` (CQL), `Kostrikov2022` (IQL), `Fujimoto2019` (BCQ).
- Trajectory models / supervised conditioning: `Chen2021DT`, `Janner2021TT`, `Janner2022diffuser`, `Ajay2023`, `Emmons2022`, `Brandfonbrener2022`.

## Hyperparameter reference (key values)

### Shared

- T = 20, N_OFFLINE_EPISODES = 500, N_EVAL_EPISODES = 1000, N_SEEDS = 20.
- BEHAVIORAL_NOISE = 0.15 (main); EPSILON_B_VALUES = [0.05, 0.3, 0.9] (coverage).

### Q-method family

- HIDDEN_DIM = 128, LEARNING_RATE = 1e-3, BATCH_SIZE = 256.
- N_FQI_ITERATIONS = 200, N_GRADIENT_STEPS = 300 (BC and BCQ behavior pre-training).
- CQL_ALPHA = 0.1, IQL_TAU = 0.7, BCQ_THRESHOLD = 0.3.

### Decision Transformer (DT)

- DT_HIDDEN_DIM = 64, DT_N_LAYERS = 2, DT_N_HEADS = 4.
- DT_CONTEXT_K = 10 (half the horizon).
- DT_N_GRADIENT_STEPS = 500, DT_BATCH_SIZE = 32, DT_LEARNING_RATE = 3e-4.
- DT_RETURN_NORM = 300.0 (divisor for normalizing return-to-go inputs).
- Target return at deployment: `dp_init_val`.

### RvS

- RVS_HIDDEN_DIM = 128, RVS_N_GRADIENT_STEPS = 500, RVS_LEARNING_RATE = 1e-3.
- RVS_RETURN_NORM = 300.0.
- Target return at deployment: `dp_init_val`.

## Open threads

- **Sim writeup prose update.** The §4 Simulation Study prose was written before DT and RvS were added; it discusses FQI/CQL/IQL/BCQ specifically. After the first full run of the refactored sim, the prose needs minor edits to mention DT and RvS in the results discussion and to refresh any numbers that shifted.
- **Operational instance (sepsis offline RL).** `ch12_world_models/PHASE5_ENRICHMENT_NOTES.md` (§8 DEFERRED) has Komorowski 2018, Raghu 2018, Gottesman 2019 prose drafts queued for an "Operational Instance" subsection in ch08. Not yet integrated.
- **MOReL / MOPO model-based offline RL.** Same `PHASE5_ENRICHMENT_NOTES.md` (§8b) has Kidambi 2020 and Yu 2020 prose drafts queued for a model-based-offline subsection. Not yet integrated.
- **Standalone PDF shows `§??` for ch02 forward references** (`section:rl_algorithms`). Won't-fix; the full monograph compile resolves these.

## File-level cheat sheet

```
ch08_offline_rl/
├── CHAPTER_NOTES.md                 ← this file
├── papers/                          ← reference PDFs
├── tex/
│   └── offline_rl.tex               ← chapter master (monolithic)
└── sims/
    ├── offline_rl_pricing.py        ← 7-method comparison sim (refactored 2026-05-18)
    ├── offline_rl_pricing_audit.md  ← 7-point audit
    ├── offline_rl_pricing_results.tex
    ├── offline_rl_pricing_coverage.png
    ├── offline_rl_pricing_stdout.txt
    └── cache/                       ← per-paradigm pkl cache (gitignored)
```

## Project memory references

- `feedback_table_rank_order.md` — rank-ordered tables and figure legends.
- `feedback_one_sim_at_a_time.md` — full 7-point audit before code; go slow.
- `feedback_update_stdout.md` — regenerate stdout + recompile after any sim change.
- `feedback_always_show_pdf.md` — recompile PDF after any tex edit.
- `feedback_no_para_headers.md` — no `\paragraph*{}` in ordinary prose.
- `feedback_hedge_parallels.md` — hedge econ↔ML parallels with "in the same family as" / "structurally parallel".
