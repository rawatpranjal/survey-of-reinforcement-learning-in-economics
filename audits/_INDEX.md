# Sim Audit Index — 2026-05-20 (post-fix-batch)

35 in-paper simulation scripts audited via the 7-point Simulation Audit defined in `CLAUDE.md`. Each audit was run by an opus subagent in hostile-reviewer mode. Diagram-only sims are capped at 25%.

The original 2026-05-19 audit flagged six sims at ≥50% (halt code work). A 16-commit fix batch landed on `humanize-pass` between 2026-05-19 and 2026-05-20; all six are now below the halting threshold. Cross-cutting pattern #3 (stale chapter paths) closed 2026-05-20.

## Score Distribution (current)

| Bucket | Count | Sims |
|---|---|---|
| ≥50% (halt code work) | **0** | — |
| 30–49% | 2 | lqc_fvi_fqi (30), fishery_paradigms (30) |
| 25–29% | 0 | — |
| 20–24% | 5 | durable_goods_monopoly (20), causal_bandit_parallel (20), nfxp_ccp_td (20-25), rbc_dp_vs_drl (20), uninformative_price (20), confounded_ope (20) |
| 15–19% | 7 | offline_rl_pricing (15, Phase 2 recovery + 2026-05-20 verify), cournot_bertrand_marl (15), algorithm_architectures (15), regret_rates (15), job_search_preference_learning (15), dyna_maze (15), overestimation_bias (15) |
| 10–14% | 4 | td_lambda_corridor (10-15), brock_mirman_newton (10), estimation_flowcharts (10), dynamic_dml_snmm (10) |
| 0–9% | 1 | identification_dags (5) |
| Other | 15 at 25% (diagram-cap or substantive) — see Full Index below |

## Resolved High-Risk Findings (post 2026-05-19 fix batch)

| Old | New | Sim | Fix commit | Disposition |
|---|---|---|---|---|
| **65%** | 20% | ch06_games / durable_goods_monopoly | 99b779c | Section retitled "The Coase Conjecture in a Durable Goods Monopoly" with TWO subsections — new asymptotic Coase sim (backward induction in T, δ; uniform-buyer continuum; scalar Bellman recursion). Original 2-period sim reframed as "Screening vs Pooling". Removed hidden 0.45–0.60 tolerance; transparent \|Δ\| column; n=1 → n=10 with SE. |
| **55%** | 20% | ch10b / causal_bandit_parallel | 82fd598 | `causal_thompson_sampling` was implementing a context-conditional variant only (missing Bareinboim 2015 consistency-axiom seeding + RDC bias weighting). Renamed throughout to `context_conditional_thompson_sampling`. Reference-line caption corrected (asymptotic lower bound, not finite-T upper bound). |
| **50%** | 10-15% | ch03 / td_lambda_corridor | 79e8bbf | Off-by-one γ in closed-form V\*(s): was γ^(19−s), now γ^(18−s). MC RMSVE 0.0091 (bias floor) → 0.0000. |
| **50%** | 20-25% | ch05 / nfxp_ccp_td | af118bc | Fixed `sim_cache` import (script now runs end-to-end); 5→10 seeds with PyTorch seed; new SE columns. Explicit footnote disclosing omitted locally-robust PMLE correction (Theorem 5 of Adusumilli-Eckardt). Bib entry `AdusumilliEckardt2022` author corrected via 7b286c0 (removed hallucinated co-author "Tate, G."). |
| **50%** | 15% | ch06_games / cournot_bertrand_marl | 99b779c | Bertrand FOC had a stray "+e·c"; p* now correctly 4. Removed phantom "Conv. iter" column. Named three pure Nash on Cournot integer grid (was falsely claimed unique). Added Calvano 2020 cite + Nash-Q tie-break footnote. |
| **50%** | 15% | ch08 / offline_rl_pricing | 99fc581 + 719243f | Phase 1 (99fc581): three algorithm-identity drifts owned in prose (IQL→IQL-argmax, BCQ→BCQ-D, DT fused-token); paragraph explains BC/BCQ-D/DT/RvS = 169.27 four-way collapse; added `Fujimoto2019b` for BCQ-D citation. Phase 2 (719243f): rebalanced behavioral [10,10,10,10] → [5,7,8,9] with triangular kernel; four-way collapse gone; new rank order RvS 97.0 > BC 96.8 > DT 96.3 > CQL 92.6 > BCQ-D 92.0 > IQL-argmax 91.8 > FQI 24.7. **Verified 2026-05-20** ([polish report](ch08_offline_rl__offline_rl_pricing_polish_2026-05-20.md)) — script, table, stdout, prose, and chapter PDF all on Phase 2; the single remaining `169.27` mention is in the rebalance footnote acknowledging the prior collapse. Phase 1/2 mismatch closed. |

## Full Index (sorted by current score, descending)

| Score | Chapter | Sim | Audit | Diagram-only |
|---|---|---|---|---|
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
| 20-25% | ch05_econ_models | nfxp_ccp_td (fixed → re-scored) | [link](ch05_econ_models__nfxp_ccp_td_2026-05-19.md) | no |
| 20% | ch06_games | durable_goods_monopoly (fixed → re-scored) | [link](ch06_games__durable_goods_monopoly_2026-05-19.md) | no |
| 20% | ch10b_rl_for_ci | causal_bandit_parallel (fixed → re-scored) | [link](ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md) | no |
| 20% | ch06_macro | rbc_dp_vs_drl | [link](ch06_macro__rbc_dp_vs_drl_2026-05-19.md) | no |
| 20% | ch07_bandits | uninformative_price | [link](ch07_bandits__uninformative_price_2026-05-19.md) | yes |
| 20% | ch10_causal | confounded_ope | [link](ch10_causal__confounded_ope_2026-05-19.md) | no |
| 15% | ch02_rl_algorithms | algorithm_architectures | [link](ch02_rl_algorithms__algorithm_architectures_2026-05-19.md) | yes |
| 15% | ch03b_deeprl_practice | overestimation_bias | [link](ch03b_deeprl_practice__overestimation_bias_2026-05-19.md) | yes |
| 15% | ch06_games | cournot_bertrand_marl (fixed → re-scored) | [link](ch06_games__cournot_bertrand_marl_2026-05-19.md) | no |
| 15% | ch07_bandits | regret_rates | [link](ch07_bandits__regret_rates_2026-05-19.md) | yes |
| 15% | ch09_rlhf | job_search_preference_learning | [link](ch09_rlhf__job_search_preference_learning_2026-05-19.md) | no |
| 15% | ch08_offline_rl | offline_rl_pricing (Phase 2 + polish-verified) | [link](ch08_offline_rl__offline_rl_pricing_2026-05-19.md) | no |
| 15% | ch12_world_models | dyna_maze | [link](ch12_world_models__dyna_maze_2026-05-19.md) | no |
| 10-15% | ch03_theory | td_lambda_corridor (fixed → re-scored) | [link](ch03_theory__td_lambda_corridor_2026-05-19.md) | no |
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

1. **Algorithm-identity / paper-name mismatch is the #1 failure mode** (durable_goods_monopoly, causal_bandit_parallel, cournot_bertrand_marl, offline_rl_pricing, nfxp_ccp_td). Pattern: code implements the right family but the wrong specific algorithm, then the tex prose makes claims that match the named (not the implemented) algorithm. **All five owned in prose 2026-05-20.**
2. **Sub-10-seed reporting is widespread** even outside ≥50%: carbon_constrained_production (N=1), robust_consumption_savings (N=1), benchmark_bus_engine (N=3), brock_mirman_bellman (N=3), trust_region_lqc (single-seed for unconstrained step). CLAUDE.md mandates N≥10 with mean+SE. **OPEN.**
3. **Stale paths from the chapter renames** (ch11→ch10b, ch08_rlhf→ch09_rlhf, ch12_forecasting_rl→ch10_causal) leaking into stdout files, script docstrings, and tex footnotes. **CLOSED 2026-05-20** — 12 files patched (5 sim scripts, 6 stdout files, 1 build script).
4. **Hallucinated / incorrect `refs.bib` entries**: confirmed and expanded 2026-05-20. Beyond AdusumilliEckardt2022 (fixed in 7b286c0): five additional entries in ch07_bandits/tex/dynamic_pricing.tex had fabricated metadata — Tullii2024, Fan2024, Liu2024strategic, Agrawal2024ref all fixed today. Chen2025fairness remains open: the recorded paper does not exist; real "Dynamic Pricing with Fairness Constraints" is by Cohen-Miao-Wang (Operations Research 2025, not arXiv) and does not establish the cited Θ(T^{2/3}) regret. Needs user decision on whether to replace citation, drop the claim, or find the actual paper that established the rate.
5. **Diagram-only sims are mostly clean** (median 20%, max 25% at the cap).
