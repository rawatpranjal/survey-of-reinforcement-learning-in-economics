# Audit (DELTA): ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py

**Date:** 2026-07-14
**Type:** DELTA (prior full audit 2026-05-19; polish 2026-05-20; dqn_hd paired-seed re-run 2026-05-20)
**Diagram-only:** no
**Subject focus:** (1) single coherent provenance of the committed stdout/table/figure; (2) accuracy of the paired-vs-independent-seed caption/prose as shipped; (3) full number-consistency sweep. Unchanged internals audited at skim depth.

**Files read end to end this turn:**
- `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py`
- `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt`
- `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex`
- `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.png` (viewed)
- `/Users/pranjal/Code/rl/ch10b_rl_for_ci/tex/rl_for_ci.tex` (§subsec:gmethods_bridge, lines 25-78, plus §synthesis lines 110-371)
- git diffs across `8a57f6e` (create) → `fc5c791` → `32107f7` → `0a604f5` (HEAD) for the script and stdout
- prior audits: `ch10b_rl_for_ci__dtr_qlearning_vs_murphy_2026-05-19.md`, `..._polish_2026-05-20.md`, `dtr_dqn_hd_rerun_2026-05-20.md`

## Delta summary (what changed since the 2026-05-19 audit)

Two commits touch the sim after the 2026-05-19 audit:

- **`32107f7` (polish, 2026-05-19 20:55):** re-paired the RNG in `run_qlearn_N_sweep`, `run_qlearn_epochs_sweep`, and `run_dqn_hd_sweep` — cohort now drawn from `default_rng(N*k+s)` (matching the Murphy/NN-FQI cohort) and epoch/minibatch shuffles from a separate `+7/+13`-offset stream. Relabelled "Murphy" → "Plug-in g-computation" in stdout, figure legend, and results-table rows. The tabular `qlearn_N`/`qlearn_epochs` caches were refreshed in this pass; the `dqn_hd` cache was **not**.
- **`0a604f5` (fix, 2026-05-20 02:15):** added `'seed_scheme': 'cohort_N100_paired_v2'` to `DQN_HD_CONFIG` (script lines 337-342) and force-recomputed `dqn_hd` so its cache matches the paired-seed source.

**Provenance verdict — single coherent config: CONFIRMED.** The tabular pairing genuinely took effect: at `32107f7` the committed stdout Q-learn numbers changed (N=100: 0.9755→0.9739, N=300: 0.9899→0.9914, N=3000: 0.9966→0.9979; Q2 epoch-1: 0.9825→0.9842), proving those caches were regenerated under the paired seeds, while Murphy's unchanged numbers (0.9743, 0.9907, ...) confirm its cohort seed was already `N*1000+s`. The `dqn_hd` numbers did **not** change at `32107f7` (500/2000/5000 stayed 0.7994/0.8335/0.9126), confirming the stale-cache gap, then changed at `0a604f5` to 0.7936/0.8326/0.9100 with tightened SEs — the signature of pairing. The currently committed `_stdout.txt` contains two back-to-back runs: the first with `forcing recompute of: ['dqn_hd']` (lines 6-14) and the second an all-cache-hit run (lines 52-58); both emit byte-identical result tables (lines 20-46 vs 60-90). That is direct evidence the on-disk cache reproduces the reported numbers and that all seven components derive from one config.

## Step-3 thesis statement

(i) **Theoretical claim this part of the chapter advances:** the backward recursion at the heart of Murphy's g-methods estimator for optimal dynamic treatment regimes is the same Bellman backup that drives Watkins's Q-learning (rl_for_ci.tex line 59: "The recursion ... is the same Bellman backup that drives Q-learning"). Batch backward regression (saturated plug-in g-computation = tabular Fitted-Q-Iteration) and online Q-learning (stochastic approximation on the same conditional expectation) therefore recover the same optimal regime, and the identity survives the move to function approximation (Neural-FQI vs DQN).

(ii) **What the sim is evidence FOR:** that the two estimators recover statistically indistinguishable normalized policy value `V(π̂)/V*` across three cuts — (Q1) growing tabular cohort size, (Q2) growing Q-learning replay budget at fixed N, (Q3) high-dimensional continuous state where only the neural analogues are feasible. It is a demonstration of the equivalence, not a claim that one method dominates.

## Criteria verdicts

### (a) CORRECTNESS — PASS

- Murphy estimator (`murphy_estimate_tab`, lines 205-231) computes the saturated empirical conditional mean of Y over `(S1,A1,S2,A2)` cells, maxes over A2, and recurses to stage 1 — exactly plug-in g-computation on a fully-saturated history-augmented model, i.e. tabular FQI. The label was corrected from "Murphy (FQI)" to "Plug-in g-computation (Murphy 2003 reference baseline)" (tex line 70), with a footnote (line 65) disclosing that the regret/blip estimator of Murphy 2003 §3-4 is deliberately not implemented. Honest.
- Q-learning (`qlearn_estimate_tab`, lines 234-256) is constant-α (0.1) TD on the same two coupled tables, `target1 = max_a Q2[s1,a1,s2,a]`. Constant α explains the residual sub-oracle gap (0.9968 at N=10000), correctly named "the constant-α bias" (tex line 65).
- Neural-FQI (lines 435-463) fits Q2 to Y then Q1 to the V2 bootstrap; DQN (lines 466-497) trains both nets jointly by minibatch TD with a detached target. DQN carries no target network or separate replay buffer, but for a two-stage finite horizon this is a defensible simplification and the tex does not overclaim it.
- Theory-consistency holds: all estimators approach V* as N grows (Q1, Q3), Q-learning catches the g-computation reference by 3 replay epochs (Q2), and no method substantively exceeds the oracle. See finding F1 for a float-epsilon table-sort artifact that is not a real overshoot.

### (b) PRESENTATION / NUMBERS — PASS (two cosmetic nits)

Full stdout ↔ .tex ↔ figure ↔ prose sweep, all consistent to 4 decimals:

| Quantity | stdout | results.tex | figure | prose |
|---|---|---|---|---|
| Plug-in g-comp tabular N=10000 | 1.0000 (0.0000) | 1.0000 (0.0000) | blue on oracle line, Q1 | "approaching V*" (l.65) |
| Q-learning tabular N=10000, 100 replays | 0.9968 (0.0007) | 0.9968 (0.0007) | orange ~0.997, Q1 | "small residual gap ... constant-α bias" (l.65) |
| Q2 epoch-1 vs g-comp ref 0.9907 | 0.9842 < 0.9907 | — | orange below dashed, Q2 | "below ... by three epochs caught up" (l.65) |
| NN-FQI high-dim N=5000 | 0.9310 (0.0024) | 0.9310 (0.0024) | blue ~0.93, Q3 | "slightly more sample-efficient" (l.65) |
| DQN high-dim N=5000 | 0.9100 (0.0018) | 0.9100 (0.0018) | orange ~0.91, Q3 | — |
| Oracle V* tabular / high-dim | 3.5820 / 0.7857 | 1.0000 / 1.0000 (normed) | dashed line at 1.0 | — |
| Behavior policy high-dim | 0.3350 → 0.426 normed | — | dotted ~0.43, Q3 | "behavior-policy baseline" (l.65) |

Table is rank-ordered by `V(π̂)/V*` descending (CLAUDE.md requirement), values descend 1.0000, 1.0000, 1.0000, 0.9968, 0.9310, 0.9100. The gradient-budget footnote arithmetic checks: NN-FQI at N=5000 sees 5000×200×2 ≈ 2.0M sample-visits vs DQN 8000×64 ≈ 0.51M, ratio ≈ 3.9 ≈ "roughly four times" (tex line 65 footnote). Nits: F1 (float-eps sort places the estimator above the oracle) and F2 (figure legend "Neural-FQI (plug-in)" vs caption "Neural Fitted Q-Iteration") below.

### (c) CHAPTER FIT — PASS

The sim demonstrates precisely the §subsec:gmethods_bridge thesis stated in step 3: the two estimators track each other toward V* in tabular (Q1), Q-learning needs replay budget to close the one-shot-regression gap (Q2), and both neural analogues rise from the behavior baseline toward V* under function approximation (Q3). Figure and Table 2 are both `\input`/`\includegraphics`'d into the section (tex lines 69, 78) and the prose interprets each panel. Directly on-claim.

### (d) EFFICIENCY / STANDARDS — PASS (one latent gap)

Per-component `compute_or_load` layering (lines 530-571), `add_component_args` flags with `--data-only`/`--plots-only`/`--algo` force, seeds 50 (tabular) and 20 (high-dim) both ≥10, means with 1.96·SE 95% CIs, stdout is facts-only with a config header and per-question tables. Strict boundaries respected: `compute_data` writes no figures/tables, `generate_outputs` trains nothing. Gaps: F3 (asymmetric seed-scheme cache-buster) and the carried-over compute-time prints (F4).

## 7-point checklist

1. **Algorithm identity** — PASS. Plug-in g-computation = saturated tabular FQI (honestly relabelled, regret/blip disclosed as out of scope); Q-learning = constant-α TD backup; NN-FQI/DQN reasonable analogues. DQN lacks a target network — acceptable at H=2, not overclaimed.
2. **Environment/MDP fidelity** — PASS. Two-stage DTR: tabular 5-state ordinal with logistic behavior and threshold-interaction outcome (lines 58-161); high-dim p=10 Gaussian AR transition with first-coordinate treatment effect (lines 305-358). Matches tex.
3. **Data integrity** — PASS. Numbers trace to live cache contents; the double stdout run (force + cache-hit) reproduces identical values; no hardcoded results.
4. **Comparison fairness** — PASS. Cohort seeds now paired within-seed for all panels (Murphy `N*1000+s` = qlearn cohort `N*1000+s`, lines 264/277; NN-FQI `N*100+s` = DQN cohort `N*100+s`, lines 505/518). FQI-vs-DQN gradient budget is not equated but the asymmetry is disclosed in the tex line 65 footnote rather than hidden.
5. **Theoretical sanity** — PASS. Convergence to V*, constant-α residual bias, NN methods clear the behavior baseline; no substantive oracle violation (F1 is a 1e-15 artifact).
6. **Information leakage** — PASS. Estimators receive only `(S1,A1,S2,A2,Y)`; oracle and `evaluate_policy_*` are evaluation-side only, standard OPE-vs-oracle protocol (lines 192-202, 413-432).
7. **Seed/reproducibility** — PASS with caveat (F3). Seeds fixed and ≥10 with SE; but the tabular components' config hash does not encode the seed offset, so a future seed-scheme change there would survive on a stale cache silently (the exact failure that bit dqn_hd), only guarded for `dqn_hd`.

## Prior-audit open-item disposition

From **2026-05-19** (five Reviewer-2 catches):
1. Caption "30 seeds" vs code 50/20 → **RESOLVED.** tex now reads "50 Monte Carlo seeds" (line 70), "50 seeds" (Q2), "20 seeds" (Q3), and "50 / 20 Monte Carlo seeds" in the table caption (line 76).
2. NN-FQI ~4× more gradient signal than DQN confounds "more sample-efficient" → **RESOLVED (disclosed).** Footnote at tex line 65 states they are "not gradient-step-equated" and the gap "reflects combined sample-and-compute efficiency rather than sample efficiency alone." Confound acknowledged, not eliminated — acceptable for a survey demonstration.
3. "Murphy" is a generous label → **RESOLVED.** Relabelled to "Plug-in g-computation (Murphy 2003 reference baseline)" with the plug-in-vs-blip footnote.
4. Murphy and Q-learning use different cohort seeds → **RESOLVED.** Paired in code (`32107f7`) and caches refreshed for tabular; `dqn_hd` force-recomputed (`0a604f5`).
5. `compute_data` print side-effects in oracle components → **STILL OPEN (cosmetic, F4).** `compute_oracle_tab` (lines 183-185) and `compute_oracle_hd` (lines 395-396) still print during compute.

From **2026-05-20 polish** (F1-F3):
- F1 stale `dqn_hd` cache vs "paired" caption → **RESOLVED.** Force-recompute landed the paired values (0.7936/0.8326/0.9100), the `seed_scheme` key now prevents recurrence for that component, and the caption "paired across estimators" (tex lines 70, 76) is accurate as shipped. The delta re-run genuinely closed the caption-honesty question.
- F2 figure legend "Neural-FQI (plug-in)" vs caption "Neural Fitted Q-Iteration" → **STILL OPEN (harmless).** Script line 673 vs tex line 70.
- F3 `MURPHY_CONFIG` cache key retains "Murphy" internally → **STILL OPEN (intentional).** Preserves the Murphy-sweep cache hit; internal only.

## Findings, severity-ordered

**F1 (low, presentation).** The results table lists "Plug-in g-computation (tabular, N=10000) & 1.0000" *above* "Oracle V* (tabular) & 1.0000" (results.tex lines 5-6). The rank-sort `rows.sort(key=lambda r: -r[1])` (script line 707) is stable, so the estimator can only precede the two oracle rows if its mean exceeds 1.0 before display rounding — i.e. `evaluate_policy_tab` on the recovered (optimal) policy returns a value a few ULP above `V_star` because it sums outcome terms in a different order than `compute_oracle_tab`. Not a real oracle overshoot (both cells read 1.0000; π̂ = π* to machine precision), but an adversarial reviewer glancing at the table sees an estimator ranked above the ground-truth oracle it is normalized against. Cosmetic. Optional fix: clamp `m_means` at 1.0 for the sort key, or pin the two oracle rows to the top.

**F2 (low, cosmetic).** Panel-3 figure legend "Neural-FQI (plug-in)" (script line 673) vs figure caption "Neural Fitted Q-Iteration" (tex line 70). Harmless wording drift, carried unresolved from the 2026-05-20 polish.

**F3 (low, latent-maintainability).** The `seed_scheme` cache-buster that fixed the `dqn_hd` stale-cache bug (script line 342) was added only to `DQN_HD_CONFIG`. `QLEARN_N_CONFIG` (lines 88-91) and `QLEARN_EPOCHS_CONFIG` (lines 92-95) had their cohort seed scheme changed in the very same commit (`32107f7`) yet carry no offset in their hashed config. The committed tabular numbers are correct (caches were manually refreshed, confirmed by the number change at `32107f7`), but the guard is asymmetric: a future edit to the tabular seed scheme would silently survive on a stale cache — the exact failure mode that the whole `0a604f5` fix was written to prevent. Invisible to a reviewer; a robustness note for the next editor.

**F4 (trivial, cosmetic).** `compute_oracle_tab`/`compute_oracle_hd` print inside the compute path (lines 183-185, 395-396); flagged in the 2026-05-19 audit, still present. The CLAUDE.md strict boundary bans `plt`/`.tex` writes in `compute_data`, not prints, so this is a soft style nit.

No substance-breaking issue. The equivalence claim, identification (sequential ignorability by construction), analytical oracle, convergence story, and information-leakage hygiene all hold. The delta re-run coherently closed the one open substantive item (paired-seed caption honesty), and the committed artifact is a single, reproducible provenance confirmed by the force + cache-hit double run in stdout.

**Bullshit score: 15%** — the sharpest hostile catch is a rank-sort that lists the estimator one row above the oracle (both 1.0000, a float-epsilon artifact); with the harmless legend/caption wording drift alongside it, that is a snarky-footnote-level nit, and the substance, provenance, and paired-seed claims all survive.
