# Sim Audit Index — 2026-05-19

35 in-paper simulation scripts audited via the 7-point Simulation Audit defined in `CLAUDE.md`. Each audit was run by an opus subagent in hostile-reviewer mode. Diagram-only sims are capped at 25%.

## Score Distribution

| Bucket | Count | Sims |
|---|---|---|
| ≥50% (halt code work) | **6** | durable_goods_monopoly (65), causal_bandit_parallel (55), td_lambda_corridor (50), nfxp_ccp_td (50), cournot_bertrand_marl (50), offline_rl_pricing (50) |
| 30–49% | 2 | lqc_fvi_fqi (30), fishery_paradigms (30) |
| 20–29% | 18 | (15 at 25%, 3 at 20%) |
| 10–19% | 8 | |
| 0–9% | 1 | identification_dags (5) |

## High-Risk Findings (≥50% — halt code work per CLAUDE.md)

| Score | Chapter / Sim | Headline finding |
|---|---|---|
| **65%** | ch06_games / durable_goods_monopoly | Section titled "The Coase Conjecture" but the sim is a 2-period game with hard-coded 2-element price set; Coase is asymptotic — a 2-period model with the screening price pre-supplied cannot exhibit it. Status column also mis-checkmarked via undisclosed "near threshold" exception. |
| **55%** | ch10b / causal_bandit_parallel | "TS_C" mislabelled — script implements context-conditional Thompson sampling, not Bareinboim 2015's TS_C (missing consistency-axiom seeding and RDC bias weighting). Non-monotone m grid contradicts tex's √(m*/T) claim. Reference-line caption inverted. |
| **50%** | ch03 / td_lambda_corridor | Closed-form `V*(s) = γ^(19−s)` in script and tex is off by one factor of γ vs the implemented Bellman recursion. Reported "final RMSVE = 0.0091" is the resulting bias floor, not convergence. One-character fix. |
| **50%** | ch05 / nfxp_ccp_td | Script doesn't execute as committed (NameError on `sim_cache` imports); cached pickle/table came from a prior working version. TD-CCP variants reformulated from Adusumilli-Eckardt 2022 omitting the locally robust PMLE correction (Theorem 5 — the paper's main contribution). Bib entry for `AdusumilliEckardt2022` has hallucinated co-author "Tate, G." and wrong title. |
| **50%** | ch06_games / cournot_bertrand_marl | Bertrand Nash formula wrong in script line 69 — agents converge to true Nash but the reported `|a−a*|=0.33` measures distance to a fictitious target. Cournot uniqueness claim false on integer grid (three pure NE: (2,4), (3,3), (4,2)). "Conv. iter = 1000" constant masquerading as a measurement. Nash-Q silently picks joint-payoff-max equilibrium, not Hu-Wellman. |
| **50%** | ch08 / offline_rl_pricing | IQL policy step uses `argmax_a Q` instead of advantage-weighted regression. BCQ implements discrete BCQ-D (Fujimoto 2019b benchmark) but chapter cites continuous BCQ (Fujimoto 2019, VAE + perturbation). DT uses fused-token form, not Chen 2021's three-tokens-per-timestep. **BC, BCQ, DT, RvS all report bit-identical `169.27 ± 0.60`** because they collapse to the same deterministic policy under heavily concentrated behavioral data. Empty `papers/` directory. |

## Full Index (sorted by score, descending)

| Score | Chapter | Sim | Audit | Diagram-only |
|---|---|---|---|---|
| 65% | ch06_games | durable_goods_monopoly | [link](ch06_games__durable_goods_monopoly_2026-05-19.md) | no |
| 55% | ch10b_rl_for_ci | causal_bandit_parallel | [link](ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md) | no |
| 50% | ch03_theory | td_lambda_corridor | [link](ch03_theory__td_lambda_corridor_2026-05-19.md) | no |
| 50% | ch05_econ_models | nfxp_ccp_td | [link](ch05_econ_models__nfxp_ccp_td_2026-05-19.md) | no |
| 50% | ch06_games | cournot_bertrand_marl | [link](ch06_games__cournot_bertrand_marl_2026-05-19.md) | no |
| 50% | ch08_offline_rl | offline_rl_pricing | [link](ch08_offline_rl__offline_rl_pricing_2026-05-19.md) | no |
| 30% | ch03_theory | lqc_fvi_fqi | [link](ch03_theory__lqc_fvi_fqi_2026-05-19.md) | no |
| 30% | ch12_world_models | fishery_paradigms | [link](ch12_world_models__fishery_paradigms_2026-05-19.md) | no |
| 25% | ch03_theory | deadly_triad_geometry | [link](ch03_theory__deadly_triad_geometry_2026-05-19.md) | yes (capped) |
| 25% | ch03_theory | info_geometry_npg | [link](ch03_theory__info_geometry_npg_2026-05-19.md) | yes (capped) |
| 25% | ch03_theory | mm_surrogate_trpo | [link](ch03_theory__mm_surrogate_trpo_2026-05-19.md) | yes (capped) |
| 25% | ch03_theory | trust_region_lqc | [link](ch03_theory__trust_region_lqc_2026-05-19.md) | yes (capped) |
| 25% | ch03b_deeprl_practice | brock_mirman_bellman | [link](ch03b_deeprl_practice__brock_mirman_bellman_2026-05-19.md) | no |
| 25% | ch04_control_problems | benchmark_bus_engine | [link](ch04_control_problems__benchmark_bus_engine_2026-05-19.md) | no |
| 25% | ch06_macro | lq_mfg | [link](ch06_macro__lq_mfg_2026-05-19.md) | no |
| 25% | ch07_bandits | curve_learning_pricing | [link](ch07_bandits__curve_learning_pricing_2026-05-19.md) | no |
| 25% | ch07_bandits | knowledge_ladder | [link](ch07_bandits__knowledge_ladder_2026-05-19.md) | no |
| 25% | ch09_rlhf | rlhf_dpo_pipeline | [link](ch09_rlhf__rlhf_dpo_pipeline_2026-05-19.md) | yes (capped) |
| 25% | ch10_causal | counterfactual_ope | [link](ch10_causal__counterfactual_ope_2026-05-19.md) | no |
| 25% | ch10b_rl_for_ci | dtr_qlearning_vs_murphy | [link](ch10b_rl_for_ci__dtr_qlearning_vs_murphy_2026-05-19.md) | no |
| 25% | ch11_dist_robust_constrained | carbon_constrained_production | [link](ch11_dist_robust_constrained__carbon_constrained_production_2026-05-19.md) | no |
| 25% | ch11_dist_robust_constrained | robust_consumption_savings | [link](ch11_dist_robust_constrained__robust_consumption_savings_2026-05-19.md) | no |
| 25% | ch12_world_models | cobweb_paradigms | [link](ch12_world_models__cobweb_paradigms_2026-05-19.md) | no |
| 20% | ch06_macro | rbc_dp_vs_drl | [link](ch06_macro__rbc_dp_vs_drl_2026-05-19.md) | no |
| 20% | ch07_bandits | uninformative_price | [link](ch07_bandits__uninformative_price_2026-05-19.md) | yes |
| 20% | ch10_causal | confounded_ope | [link](ch10_causal__confounded_ope_2026-05-19.md) | no |
| 15% | ch02_rl_algorithms | algorithm_architectures | [link](ch02_rl_algorithms__algorithm_architectures_2026-05-19.md) | yes |
| 15% | ch03b_deeprl_practice | overestimation_bias | [link](ch03b_deeprl_practice__overestimation_bias_2026-05-19.md) | yes |
| 15% | ch07_bandits | regret_rates | [link](ch07_bandits__regret_rates_2026-05-19.md) | yes |
| 15% | ch09_rlhf | job_search_preference_learning | [link](ch09_rlhf__job_search_preference_learning_2026-05-19.md) | no |
| 15% | ch12_world_models | dyna_maze | [link](ch12_world_models__dyna_maze_2026-05-19.md) | no |
| 10% | ch03_theory | brock_mirman_newton | [link](ch03_theory__brock_mirman_newton_2026-05-19.md) | no |
| 10% | ch05_econ_models | estimation_flowcharts | [link](ch05_econ_models__estimation_flowcharts_2026-05-19.md) | yes |
| 10% | ch10b_rl_for_ci | dynamic_dml_snmm | [link](ch10b_rl_for_ci__dynamic_dml_snmm_2026-05-19.md) | no |
| 5% | ch10_causal | identification_dags | [link](ch10_causal__identification_dags_2026-05-19.md) | yes |

## Triage Thresholds (from CLAUDE.md)

- **100%**: career-defining. Halt all sibling sim work. Re-run + hash check.
- **≥50%**: halt code work on this sim until user sees verdict.
- **25%**: reviewer 2 catches it but substance survives.
- **0%**: hostile reviewer reads twice, finds nothing.

## Cross-Cutting Patterns

1. **Algorithm-identity / paper-name mismatch is the #1 failure mode** (durable_goods_monopoly, causal_bandit_parallel, cournot_bertrand_marl, offline_rl_pricing, nfxp_ccp_td). Pattern: code implements the right family but the wrong specific algorithm, then the tex prose makes claims that match the named (not the implemented) algorithm.
2. **Sub-10-seed reporting is widespread** even outside ≥50%: carbon_constrained_production (N=1), robust_consumption_savings (N=1), benchmark_bus_engine (N=3), brock_mirman_bellman (N=3), trust_region_lqc (single-seed for unconstrained step). CLAUDE.md mandates N≥10 with mean+SE.
3. **Stale paths from the chapter renames** (ch11→ch10b, ch08_rlhf→ch09_rlhf, ch12_forecasting_rl→ch10_causal) leak into stdout files, script docstrings, and tex footnotes for at least 4 sims.
4. **Hallucinated / incorrect `refs.bib` entries**: confirmed for `AdusumilliEckardt2022` (wrong co-author "Tate, G.", wrong title). Possibly more — only spot-checked.
5. **Diagram-only sims are mostly clean** (median 20%, max 25% at the cap). The two ≥50% findings on diagrams are both about misleading captions vs the underlying math (deadly_triad notation swap, trust_region_lqc √2 inconsistency), but neither breaches the diagram cap.
