# Sim Audit Index — 2026-05-20 (post-polish)

35 in-paper simulation scripts audited via the 7-point Simulation Audit defined in `CLAUDE.md`. Each audit was run by an opus subagent in hostile-reviewer mode. Diagram-only sims are capped at 25%.

Three passes have run:

1. **Audit (2026-05-19)** — all 35 sims scored. Six flagged at ≥50% (halt code work).
2. **Fix + Recovery (2026-05-19 → 2026-05-20)** — the six ≥50% sims fixed via mixed strategy (bug fixes always; relabel/disclose for algorithm-identity mismatches). Three got a substantive Phase 2 recovery (Coase DP sweep, full Bareinboim TS_C, rebalanced offline-RL behavioral).
3. **Polish (2026-05-20)** — all 33 sims with reviewer-2-level findings (everything except `identification_dags` @ 5% and `algorithm_architectures` @ 15%, both already at target) given a targeted polish pass. Single-seed runs bumped to ≥10 with SE; cosmetic / framing / caption nicks resolved; mis-labeled algorithm variants relabeled with footnotes.

**Result: every sim is now ≤15%.** No ≥50% (halt), no 30%, no 25%. The paper clears the reviewer-2 bar across all 13 chapters with sims.

## Score Distribution (post-polish)

| Bucket | Count | Sims |
|---|---|---|
| ≥50% (halt code work) | **0** | — |
| 30–49% | 0 | — |
| 20–29% | 0 | — |
| 16–19% | 0 | — |
| 15% | 5 | nfxp_ccp_td, knowledge_ladder, offline_rl_pricing, job_search_preference_learning, algorithm_architectures |
| 12% | 11 | deadly_triad_geometry, brock_mirman_bellman, benchmark_bus_engine, lq_mfg, rbc_dp_vs_drl, counterfactual_ope, causal_bandit_parallel, dtr_qlearning_vs_murphy, robust_consumption_savings, cobweb_paradigms, fishery_paradigms |
| 10% | 12 | lqc_fvi_fqi, info_geometry_npg, mm_surrogate_trpo, trust_region_lqc, overestimation_bias, cournot_bertrand_marl, durable_goods_monopoly, curve_learning_pricing, uninformative_price, confounded_ope, carbon_constrained_production, dyna_maze |
| 5% | 6 | brock_mirman_newton, td_lambda_corridor, estimation_flowcharts, regret_rates, rlhf_dpo_pipeline, identification_dags |
| 3% | 1 | dynamic_dml_snmm |

## Resolved High-Risk Findings (the original six ≥50%)

| Audit | Fix | Recovery | Polish | Sim | Disposition |
|---|---|---|---|---|---|
| **65%** | 20% | 10–15% | 10% | ch06_games / durable_goods_monopoly | Fix: rescoped 2-period sim to "Screening vs Pooling", removed hidden 0.45–0.60 tolerance, n=1→10. Recovery A1: new `durable_goods_coase.py` — closed-form DP sweep over (T, δ) genuinely demonstrates the asymptotic Coase price collapse (T=200,δ=0.99 → no-commit/commit ratio 0.23, p_T→0). Polish: Coase↔screening prose flow. |
| **55%** | 20% | 18% | 12% | ch10b / causal_bandit_parallel | Fix: relabeled `causal_thompson_sampling` → `context_conditional_thompson_sampling`. Recovery B1: implemented the full Bareinboim 2015 TS_C (consistency-axiom seeding + RDC bias weighting); empirically TS_C (regret 4.49) loses to context-conditional TS (0.66) on this MDP — an honest negative finding, Bareinboim attribution restored alongside both baselines. |
| **50%** | 10–15% | — | 5% | ch03 / td_lambda_corridor | Fix: off-by-one γ in closed-form V\*(s), γ^(19−s)→γ^(18−s); MC RMSVE 0.0091 (bias floor) → 0.0000. Polish: prose pinned to table cells (crosses RMSVE<0.05 at episode 52). |
| **50%** | 20–25% | — | 15% | ch05 / nfxp_ccp_td | Fix: repaired `sim_cache` import (script ran again), 5→10 seeds + PyTorch seed + SE columns, footnote disclosing the omitted locally-robust PMLE correction (Adusumilli-Eckardt Thm 5), bib entry corrected. Polish: footnote sharpened (§3.3 + Appendix B.3), bridging sentence on TD-Neural-vs-NFXP. |
| **50%** | 15% | — | 10% | ch06_games / cournot_bertrand_marl | Fix: corrected Bertrand FOC (stray +e·c → p\*=4), dropped phantom "Conv. iter" column, named three pure Nash on the Cournot integer grid, added Calvano 2020 cite. Polish: AskerEtAl2020 cite, hu2003nash backup footnote. |
| **50%** | 25% | 15% | 15% | ch08 / offline_rl_pricing | Fix: owned three algorithm-identity drifts (IQL→IQL-argmax, BCQ→BCQ-D, DT fused-token), added Fujimoto2019b. Recovery C1: rebalanced behavioral [10,10,10,10]→[5,7,8,9] + triangular kernel; four-way collapse gone; new rank RvS 97.0 > BC 96.8 > DT 96.3 > CQL 92.6 > BCQ-D 92.0 > IQL-argmax 91.8 > FQI 24.7. Polish: verified Phase 1/2 mismatch closed. |

## Full Index (sorted by post-polish score, descending)

| Score | Chapter | Sim | Audit | Polish | Diagram-only |
|---|---|---|---|---|---|
| 15% | ch02_rl_algorithms | algorithm_architectures | [audit](ch02_rl_algorithms__algorithm_architectures_2026-05-19.md) | not polished (already ≤15%) | yes |
| 15% | ch05_econ_models | nfxp_ccp_td | [audit](ch05_econ_models__nfxp_ccp_td_2026-05-19.md) | [polish](ch05_econ_models__nfxp_ccp_td_polish_2026-05-20.md) | no |
| 15% | ch07_bandits | knowledge_ladder | [audit](ch07_bandits__knowledge_ladder_2026-05-19.md) | [polish](ch07_bandits__knowledge_ladder_polish_2026-05-20.md) | no |
| 15% | ch08_offline_rl | offline_rl_pricing | [audit](ch08_offline_rl__offline_rl_pricing_2026-05-19.md) | [polish](ch08_offline_rl__offline_rl_pricing_polish_2026-05-20.md) | no |
| 15% | ch09_rlhf | job_search_preference_learning | [audit](ch09_rlhf__job_search_preference_learning_2026-05-19.md) | [polish](ch09_rlhf__job_search_preference_learning_polish_2026-05-20.md) | no |
| 12% | ch03_theory | deadly_triad_geometry | [audit](ch03_theory__deadly_triad_geometry_2026-05-19.md) | [polish](ch03_theory__deadly_triad_geometry_polish_2026-05-20.md) | yes |
| 12% | ch03b_deeprl_practice | brock_mirman_bellman | [audit](ch03b_deeprl_practice__brock_mirman_bellman_2026-05-19.md) | [polish](ch03b_deeprl_practice__brock_mirman_bellman_polish_2026-05-20.md) | no |
| 12% | ch04_control_problems | benchmark_bus_engine | [audit](ch04_control_problems__benchmark_bus_engine_2026-05-19.md) | [polish](ch04_control_problems__benchmark_bus_engine_polish_2026-05-20.md) | no |
| 12% | ch06_macro | lq_mfg | [audit](ch06_macro__lq_mfg_2026-05-19.md) | [polish](ch06_macro__lq_mfg_polish_2026-05-20.md) | no |
| 12% | ch06_macro | rbc_dp_vs_drl | [audit](ch06_macro__rbc_dp_vs_drl_2026-05-19.md) | [polish](ch06_macro__rbc_dp_vs_drl_polish_2026-05-20.md) | no |
| 12% | ch10_causal | counterfactual_ope | [audit](ch10_causal__counterfactual_ope_2026-05-19.md) | [polish](ch10_causal__counterfactual_ope_polish_2026-05-20.md) | no |
| 12% | ch10b_rl_for_ci | causal_bandit_parallel | [audit](ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md) | [polish](ch10b_rl_for_ci__causal_bandit_parallel_polish_2026-05-20.md) | no |
| 12% | ch10b_rl_for_ci | dtr_qlearning_vs_murphy | [audit](ch10b_rl_for_ci__dtr_qlearning_vs_murphy_2026-05-19.md) | [polish](ch10b_rl_for_ci__dtr_qlearning_vs_murphy_polish_2026-05-20.md) | no |
| 12% | ch11_dist_robust_constrained | robust_consumption_savings | [audit](ch11_dist_robust_constrained__robust_consumption_savings_2026-05-19.md) | [polish](ch11_dist_robust_constrained__robust_consumption_savings_polish_2026-05-20.md) | no |
| 12% | ch12_world_models | cobweb_paradigms | [audit](ch12_world_models__cobweb_paradigms_2026-05-19.md) | [polish](ch12_world_models__cobweb_paradigms_polish_2026-05-20.md) | no |
| 12% | ch12_world_models | fishery_paradigms | [audit](ch12_world_models__fishery_paradigms_2026-05-19.md) | [polish](ch12_world_models__fishery_paradigms_polish_2026-05-20.md) | no |
| 10% | ch03_theory | lqc_fvi_fqi | [audit](ch03_theory__lqc_fvi_fqi_2026-05-19.md) | [polish](ch03_theory__lqc_fvi_fqi_polish_2026-05-20.md) | no |
| 10% | ch03_theory | info_geometry_npg | [audit](ch03_theory__info_geometry_npg_2026-05-19.md) | [polish](ch03_theory__info_geometry_npg_polish_2026-05-20.md) | yes |
| 10% | ch03_theory | mm_surrogate_trpo | [audit](ch03_theory__mm_surrogate_trpo_2026-05-19.md) | [polish](ch03_theory__mm_surrogate_trpo_polish_2026-05-20.md) | yes |
| 10% | ch03_theory | trust_region_lqc | [audit](ch03_theory__trust_region_lqc_2026-05-19.md) | [polish](ch03_theory__trust_region_lqc_polish_2026-05-20.md) | yes |
| 10% | ch03b_deeprl_practice | overestimation_bias | [audit](ch03b_deeprl_practice__overestimation_bias_2026-05-19.md) | [polish](ch03b_deeprl_practice__overestimation_bias_polish_2026-05-20.md) | yes |
| 10% | ch06_games | cournot_bertrand_marl | [audit](ch06_games__cournot_bertrand_marl_2026-05-19.md) | [polish](ch06_games__cournot_bertrand_marl_polish_2026-05-20.md) | no |
| 10% | ch06_games | durable_goods_monopoly | [audit](ch06_games__durable_goods_monopoly_2026-05-19.md) | [polish](ch06_games__durable_goods_monopoly_polish_2026-05-20.md) | no |
| 10% | ch07_bandits | curve_learning_pricing | [audit](ch07_bandits__curve_learning_pricing_2026-05-19.md) | [polish](ch07_bandits__curve_learning_pricing_polish_2026-05-20.md) | no |
| 10% | ch07_bandits | uninformative_price | [audit](ch07_bandits__uninformative_price_2026-05-19.md) | [polish](ch07_bandits__uninformative_price_polish_2026-05-20.md) | yes |
| 10% | ch10_causal | confounded_ope | [audit](ch10_causal__confounded_ope_2026-05-19.md) | [polish](ch10_causal__confounded_ope_polish_2026-05-20.md) | no |
| 10% | ch11_dist_robust_constrained | carbon_constrained_production | [audit](ch11_dist_robust_constrained__carbon_constrained_production_2026-05-19.md) | [polish](ch11_dist_robust_constrained__carbon_constrained_production_polish_2026-05-20.md) | no |
| 10% | ch12_world_models | dyna_maze | [audit](ch12_world_models__dyna_maze_2026-05-19.md) | [polish](ch12_world_models__dyna_maze_polish_2026-05-20.md) | no |
| 5% | ch03_theory | brock_mirman_newton | [audit](ch03_theory__brock_mirman_newton_2026-05-19.md) | [polish](ch03_theory__brock_mirman_newton_polish_2026-05-20.md) | no |
| 5% | ch03_theory | td_lambda_corridor | [audit](ch03_theory__td_lambda_corridor_2026-05-19.md) | [polish](ch03_theory__td_lambda_corridor_polish_2026-05-20.md) | no |
| 5% | ch05_econ_models | estimation_flowcharts | [audit](ch05_econ_models__estimation_flowcharts_2026-05-19.md) | [polish](ch05_econ_models__estimation_flowcharts_polish_2026-05-20.md) | yes |
| 5% | ch07_bandits | regret_rates | [audit](ch07_bandits__regret_rates_2026-05-19.md) | [polish](ch07_bandits__regret_rates_polish_2026-05-20.md) | yes |
| 5% | ch09_rlhf | rlhf_dpo_pipeline | [audit](ch09_rlhf__rlhf_dpo_pipeline_2026-05-19.md) | [polish](ch09_rlhf__rlhf_dpo_pipeline_polish_2026-05-20.md) | yes |
| 5% | ch10_causal | identification_dags | [audit](ch10_causal__identification_dags_2026-05-19.md) | not polished (already clean) | yes |
| 3% | ch10b_rl_for_ci | dynamic_dml_snmm | [audit](ch10b_rl_for_ci__dynamic_dml_snmm_2026-05-19.md) | [polish](ch10b_rl_for_ci__dynamic_dml_snmm_polish_2026-05-20.md) | no |

## Triage Thresholds (from CLAUDE.md)

- **100%**: career-defining. Halt all sibling sim work. Re-run + hash check.
- **≥50%**: halt code work on this sim until user sees verdict.
- **25%**: reviewer 2 catches it but substance survives.
- **0%**: hostile reviewer reads twice, finds nothing.

## Polish-Pass Substance vs Form (2026-05-20)

The polish pass was mostly reviewer-2-level form (captions, footnotes, label conventions), but it surfaced and fixed several real items:

- **carbon_constrained_production**: the tex claim "λ overshoots to 3.2" was verified false against the instrumented `lambda_trajectory` — actual peak 1.407 ± 0.003. A fabricated number, now corrected.
- **info_geometry_npg**: the matplotlib `Ellipse` rotation was 90° wrong (long axis along the steep direction of F). Fixed; verified analytically.
- **benchmark_bus_engine**: the DP-vs-DQN evaluation consumed the RNG asymmetrically across methods. Fixed with a shared `initial_states` set — the DQN-vs-DP gap is now exactly 0.0% under paired evaluation (the old 0.0–0.4% was eval noise).
- **fishery_paradigms**: added a true myopic / open-access agent — the textbook bioeconomic-collapse tragedy (753 regret, 100% stock collapse) is now actually demonstrated.
- **Single-seed → ≥10-seed** with SE: lqc_fvi_fqi, brock_mirman_bellman, benchmark_bus_engine, carbon_constrained_production (5 seeds, documented deviation), robust_consumption_savings. Most affected sims now report defensible mean ± SE.

## Cross-Cutting Patterns (status)

1. **Algorithm-identity / paper-name mismatch** (durable_goods_monopoly, causal_bandit_parallel, cournot_bertrand_marl, offline_rl_pricing, nfxp_ccp_td). **Resolved** — fixed (relabel + disclose) and, for three sims, substantively recovered.
2. **Sub-10-seed reporting**. **Resolved** — all flagged sims bumped to ≥10 seeds with mean+SE, except carbon_constrained_production (5 seeds, documented deviation justified by per-seed cost under sustained machine contention; SE on every quantity).
3. **Stale paths from chapter renames** (ch11→ch10b, ch08_rlhf→ch09_rlhf, ch12_forecasting_rl→ch10_causal). **Resolved** — patched across sim scripts, stdout files, tex footnotes, build scripts.
4. **Hallucinated / incorrect `refs.bib` entries**. **Mostly resolved** — AdusumilliEckardt2022 + five ch07 entries fixed. **One open**: `Chen2025fairness` — the recorded paper does not exist; the real "Dynamic Pricing with Fairness Constraints" is Cohen-Miao-Wang (Operations Research 2025, not arXiv) and does not establish the cited Θ(T^{2/3}) regret. Needs a user decision: replace the citation, drop the rate claim, or find the paper that actually established the bound.
5. **Diagram-only sims**: all clean (5–12% post-polish).

## Remaining open items (not blocking)

- `Chen2025fairness` bib/claim mismatch (pattern #4) — user decision needed.
- `dtr_qlearning_vs_murphy`: the high-dim DQN cache (`dqn_hd.pkl`) predates the paired-seed edit; the polish report flags two options — re-run with `--force dqn_hd`, or narrow the caption to "tabular paired; high-dim independent".
- Substantive reimplementations deliberately deferred across all passes: continuous BCQ (VAE + perturbation), advantage-weighted IQL, three-token DT, MARL-based Coase. All are disclosed in tex footnotes as simplifications.
