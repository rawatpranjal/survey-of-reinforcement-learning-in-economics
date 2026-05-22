# Re-audit: ch12_world_models/sims/fishery_paradigms.py

**Date:** 2026-05-22
**Target file:** `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms.py` (mtime 2026-05-19 15:26)
**Companion env:** `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_env.py`
**Stdout:** `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms_stdout.txt` (mtime 2026-05-19 15:26)
**Tex prose:** `/Users/pranjal/Code/rl/ch12_world_models/tex/s09_dual_sim.tex` §9.2
**Prior scores:** original 2026-05-19 = 30%; polish 2026-05-20 = 12%
**This re-audit:** see final line.

## Polish-fix verification (spec §B1)

| # | Spec item | Status | Evidence |
|---|---|---|---|
| B1.1 | Rename Model-Based LQ → Model-Based DP | **LANDED** | `MBPOPolicy.name = 'Model-Based DP'` (line 304); registry, `PARADIGM_ORDER`, `PARADIGM_COLORS`, and tex prose §9.2 all consistent. Stale `Model-Based_LQ.pkl` removed; fresh `Model-Based_DP.pkl` (May 19 15:23) in cache. |
| B1.2 | Open-access / myopic `h = p/c` showing collapse | **LANDED** | `MyopicPolicy` (lines 90-110) returns `min(s, p/c) = min(s, 10)`. Stdout shows Myopic final regret 753.11 ± 1.75 with 100 % collapse fraction. Textbook tragedy is now visible. |
| B1.3 | GA election operator stock-dynamics term | **FOOTNOTED only** | Code lines 271-289 still scores child vs parent on `pi = p_*h - 0.5*c_*h**2` evaluated at `last_s` (a static one-period reward). The discounted bioeconomic-objective fix the spec asked for was not implemented; tex footnote (s09_dual_sim.tex line 70) discloses the simplification. |
| B1.4 | Q-Learning / GA / MBPO action-grid prior `h_max = 1.5·r·K/4` documented or removed | **FOOTNOTED only** | Lines 189 (`QLearningPolicy.reset`), 245 (`ArifovicGAPolicy.reset`), 332 (`MBPOPolicy.reset`) all still set `self.h_max = 1.5 * params['r'] * params['K'] / 4.0` from the true `r, K`. Tex footnote (s09_dual_sim.tex line 70) discloses this as "fixed problem-specific scaffolding". Disclosure, not removal. |
| B1.5 | Parameter-recovery table | **LANDED** | `_extract_param_estimates` (lines 412-420) pulls `(r̂, K̂)` from RLS and Model-Based DP; `generate_outputs` writes `fishery_paradigms_recovery.tex` and prints recovery in stdout. RLS recovers `r̂ = 0.398 ± 0.002, K̂ = 10.034 ± 0.064`; MB-DP recovers `r̂ = 0.400 ± 0.002, K̂ = 9.965 ± 0.050`. |
| B1.6 | Stock + harvest trajectory figure | **MISSING** | `compute_paradigm` (lines 470-500) stores only `regret_curves` and `final_regret`; it never records `s_t` or `h_t`. `generate_outputs` writes one figure (`fishery_paradigms.png`) and that figure is the regret panel only (verified by reading lines 554-571). The spec asked specifically for stock + harvest trajectories so a reader can see the Myopic collapse on step one and the structured-learner stabilization near `s* ≈ 4.4`. Nothing in the polish audit's "Files changed" lists a trajectory figure, and the polish-audit narrative skips B1.6 entirely. |

Four landed, two disclosure-only, one outright missing.

## 1. Algorithm Identity

**FINDING.** Six learning paradigms + oracle + naive constant rule. Identities:
- Oracle: grid-DP with Gauss-Hermite quadrature (`fishery_env.solve_oracle_dp`). Faithful.
- Naive: constant `h = 0.5`. Trivial.
- Myopic: `h_t = min(s_t, p/c) = min(s_t, 10)`. Correct unconstrained-interior open-access agent.
- RLS: information-form recursive LS on `(s, -s²)` against `Δs + h`, recovers `(r, K)` and re-solves DP every 25 steps with known `(p, c, σ)`. Faithful to Marcet-Sargent 1989 with the stated cost prior.
- Q-Learning: tabular ε-greedy on a 30×21 grid, α = 0.1, ε decay 0.3→0.01 over T = 500. Canonical Watkins-Dayan.
- Arifovic GA: fitness-proportional selection, single-point crossover, bit-flip mutation. Election operator uses *only* myopic one-period profit `ph - (c/2)h²` evaluated at the last observed stock. The spec asked for a stock-dynamics term; the code still uses static period profit. The tex now footnotes this as a deliberate simplification.
- Model-Based DP: batch LS on growth and reward residuals; re-solves grid-DP. Name now matches the planner.

**EVIDENCE.** Class definitions lines 58-388 of `fishery_paradigms.py`; helper `solve_oracle_dp` lines 46-104 of `fishery_env.py`; election operator lines 266-296.

**VERDICT.** Identity holds for six of seven paradigms. The Arifovic GA election operator is faithful to the *renamed* description (the tex now says "partial election operator with known cost parameters") but not to the spec's B1.3 ask. A bioeconomist reviewer who reads "Arifovic election operator" and inspects the code will note the discounted-rollout step is replaced by a one-shot reward; the polish footnote anticipates the complaint without resolving it. Reviewer-2-level finding.

## 2. Environment / MDP Fidelity

**FINDING.** State `s ∈ [0, 1.5K]`, action `h ∈ [0, min(s, 1.5·rK/4)]`, dynamics `s_{t+1} = max(0, s_t + r s_t (1 - s_t/K) - h_t + ε_t)` with `ε_t ∼ N(0, σ²)` truncated at `s_max = 1.5K`. Reward `r_t = p h_t - (c/2) h_t²`. Parameters `r = 0.4, K = 10, p = 2, c = 0.2, σ = 0.3, γ = 0.95, T = 500, s_0 = K`.

**EVIDENCE.** `FisheryEnv` lines 8-43 of `fishery_env.py`; `ENV_PARAMS` line 23 of `fishery_paradigms.py`; tex §9.2 line 64 ("again linear-quadratic"; explicitly documented).

**VERDICT.** Dynamics are clean Schaefer logistic; reward is the non-standard linear-quadratic form (not the classical Schaefer-Clark `ph - c(s)h`), but the choice is disclosed in tex. The hard `s_max = 1.5K` cap is a numerical guardrail; with σ = 0.3 it rarely binds. A hostile reviewer might still snipe that the linear-quadratic reward is a convenient teaching device that lets the *static* myopic optimum equal `p/c = 10` cleanly, which is the very fact the Myopic baseline now exploits to collapse the stock on step one. A reviewer who wants gradual open-access collapse rather than instantaneous one-shot collapse will ask for `p h_t - c_0 h_t` with `c_0 < p`, which would give a per-period optimum at the boundary and yield depletion over several steps rather than one. Out of scope but Reviewer-2 territory.

## 3. Data Integrity

**FINDING.** `compute_data` follows the canonical per-component-cache pattern: `compute_or_load(..., 'shared', ...)` rolls out the oracle once per seed; each paradigm has its own `compute_or_load` keyed on its config dict. Stdout shows seven cache hits ("Cache hit: shared / Oracle / Naive / Myopic / RLS / Q-Learning / Arifovic_GA / Model-Based_DP"). The cumulative-regret numbers printed in stdout exactly match the LaTeX table `fishery_paradigms_results.tex` (Oracle 0.00, RLS 13.67 ± 0.43, MB-DP 14.69 ± 0.73, QL 274.71 ± 24.36, Naive 447.35 ± 3.24, GA 706.13 ± 16.65, Myopic 753.11 ± 1.75). Recovery numbers in the tex (`RLS 0.398 ± 0.002 / 10.034 ± 0.064`, `MB-DP 0.400 ± 0.002 / 9.965 ± 0.050`) match stdout exactly.

**EVIDENCE.** `fishery_paradigms_stdout.txt` lines 16-46 vs `fishery_paradigms_results.tex` lines 6-12 and `fishery_paradigms_recovery.tex` lines 6-7. Cache mtimes: RLS cache 2026-05-19 15:20, MB-DP 15:23, Myopic 15:17, all *newer* than the code's last edit at 15:26 — wait, this is a stale-cache *risk*. Re-checking: code mtime 15:26, RLS cache 15:20, MB-DP cache 15:23, Myopic cache 15:17. **All three caches were written *before* the final code edit at 15:26.**

This raises a stale-cache concern. The polish audit narrative claims "Model-Based DP's re-run reproduces its original 14.69 ± 0.73 because the cache key was a name change and the algorithm itself is identical." However, the spec.md fix sheet listed B1.5 (parameter recovery) as a code change, and `_extract_param_estimates` plus the `r_hats / K_hats` fields are *added* by that change. If those caches were written before the helper was added, the cached `compute_paradigm` outputs would not contain `r_hats` and `K_hats`. Yet the stdout shows recovery values being printed. So either (i) caches were re-written at 15:20-15:23 *after* the recovery helper was added but *before* a final cosmetic edit at 15:26 (a no-op), or (ii) the recovery-table run hit cache and the values came from a previously-computed run that happens to be deterministic. The recovery values are deterministic given seeds; either path produces the same numbers.

**VERDICT.** No correctness defect — the per-paradigm cache is keyed on the config dict and the run is deterministic seed by seed. But a hostile reviewer auditing the artifact would object that the cache mtimes predate the code mtime, which violates the project's stated "config-keyed cache invalidates on hyperparameter change" guarantee at the file level (the config dict didn't change, so cache stays valid, but file mtime ordering creates an audit ambiguity). Project-hygiene complaint, not a correctness issue.

## 4. Comparison Fairness

**FINDING.** All seven paradigms call `rollout(paradigm, params, T=500, gamma=0.95, seed=s)` for `s in range(20)`. `FisheryEnv(seed=s)` is re-instantiated per paradigm per seed, so the noise sequence is fixed by `s`. Regret is computed as `cumsum(oracle_rewards[s] - paradigm_rewards[s])`, a paired comparison against the same per-step noise realization. Same `T`, same env params, same horizon, same `N_SEEDS = 20`.

**EVIDENCE.** `rollout` lines 395-409, `compute_paradigm` lines 470-500.

**VERDICT.** Paired-noise protocol is correct. Hyperparameter budgets differ across paradigms (Q-Learning grid, GA population, RLS refit cadence, MBPO warm-up), and there is no hyperparameter-sweep evidence. The tex (line 70) bills the experiment as "deliberately favorable to structured learners" so this is disclosed. A reviewer might still ask "what would Q-Learning do with `g_s = 100, g_h = 51, T = 5000`?" That is the standard tabular-RL underbudget complaint and is preempted by the disclosure. Reviewer-2 catches the lack of a sensitivity panel but the comparison itself is fair.

## 5. Theoretical Sanity Checks

**FINDING.** Ordered checks against theory.
- **Oracle on top:** regret 0.00 ± 0.00 by construction. Passes.
- **Open-access collapse:** Myopic at 753.11 with 100% collapse fraction. `h = p/c = 10` exceeds `s_0 = K = 10`, so the stock is depleted on step 1 and `r_t = 0` thereafter. The asymptotic regret is `T · V*_per-step ≈ 500 · 1.5 ≈ 750`, consistent with the observed 753.11. Passes the textbook bioeconomic-tragedy prediction.
- **Structured learners near oracle:** RLS at 13.67, MB-DP at 14.69. Both well within an order of magnitude of zero on a 500-step horizon. Consistent with `(r, K)` recovery to within 2-3% — once the model is right, the planner is the oracle.
- **GA worse than Naive:** GA finishes at 706, Naive at 447. Both encode constant-rule policies, but GA's stochastic population search dithers into harvest rates near `p/c = 10` for 60% of seeds (collapse fraction in stdout) while Naive's fixed `h = 0.5` does not. The tex now explains this explicitly ("population-search noise causes a non-trivial fraction of seeds to drift into harvest rates that themselves induce a collapse"). Plausible mechanism.
- **No oracle-beating:** verified — minimum regret is 0 (Oracle by construction).

**EVIDENCE.** Stdout lines 16-26 (regret) and 28-38 (collapse incidence).

**VERDICT.** All theoretical anchors pass. The Myopic-collapse prediction is now visible, which is a genuine improvement over the original audit's "this sim does not exhibit the textbook tragedy" complaint. A hostile reviewer might still object that GA underperforming Naive on the same chromosome class is "thin evidence" (the original audit's wording), but the tex now offers a specific mechanism, which is an acceptable response.

## 6. No Information Leakage

**FINDING.** Item-by-item.
- Oracle: knows everything by construction.
- Naive: knows nothing.
- Myopic: knows `(p, c)`. Open-access framing typically assumes prices are observable; defensible.
- RLS: knows `(p, c, σ)`. Tex discloses this ("with known cost parameters").
- Q-Learning: action grid bounded by `h_max = 1.5 · r · K / 4` using **true** `r, K` (line 189). The spec's B1.4 asked for removal-or-documentation; the polish pass chose documentation only (tex footnote). A hostile reviewer would still flag this: a "model-free" learner that knows MSY = `rK/4` to within a 1.5× factor has a strong action-support prior.
- Arifovic GA: chromosome decode range uses true `r, K` (line 245); election operator uses true `(p, c)` (line 272). Same footnote covers the action-support leak.
- Model-Based DP: exploration clip uses true `r, K` (line 332). Same footnote.

**EVIDENCE.** Lines 189, 245, 332 (action-bound leak); lines 134-136, 104-105, 269-272 (cost-parameter knowledge).

**VERDICT.** The structural-prior leak into the action support is disclosed but not removed. The polish footnote anticipates the reviewer's complaint and frames the bound as "fixed problem-specific scaffolding". A bioeconomist reviewer may accept; a methods reviewer may push back with "then write a sentence about how a method-of-moments estimate of `rK/4` from initial exploration would substitute." Disclosure-only, not Reviewer-1-level.

## 7. Seed & Reproducibility

**FINDING.** `N_SEEDS = 20` (meets ≥10). Means and standard errors reported. Seeds for the env are `range(20)`; paradigm-internal RNGs are `seed + offset` (deterministic per paradigm). `compute_paradigm` also calls the legacy `np.random.seed(s)` at line 480, which is a no-op for the default_rng-using paradigms but a defensible defensive default.

**EVIDENCE.** `compute_paradigm` lines 470-500, `SHARED_CONFIG` line 24-27.

**VERDICT.** Reproducible. Standard-error magnitudes (e.g. Naive ± 3.24, Myopic ± 1.75, MB-DP ± 0.73) are consistent with the noise budget; not suspiciously small, not suspiciously large. Reviewer-2 may complain that the `np.random.seed(s)` call alongside `default_rng(seed + offset)` is a code-smell sloppy pattern; cosmetic.

## Hostile-reviewer summary

The polish pass landed four of six B1 items cleanly (B1.1, B1.2, B1.5 in code; B1.3 and B1.4 disclosed in tex footnotes). One item (**B1.6, stock + harvest trajectory figure**) was *not addressed at all*. The polish audit's "Files changed" list does not mention any trajectory figure, the `generate_outputs` function produces only the regret panel, and `compute_paradigm` does not even record stock or harvest trajectories. The spec explicitly listed B1.6 as a Phase B1 deliverable.

A hostile reviewer reading the chapter writeup will look for the trajectory figure that the prose at s09_dual_sim.tex line 73 implicitly promises ("harvests the stock to zero on the first step and earns nothing thereafter") and find only the regret panel. The visual verification of the textbook collapse exists only as a derived statistic ("100% collapse fraction" in stdout) and never reaches the reader's eye through the figure. This is a real gap, smaller than the original-audit "no open-access agent" complaint, but visible.

The B1.3 (GA election operator) and B1.4 (action-grid prior) footnote-only fixes are acceptable disclosures of pre-existing assumptions but represent the kind of "we said we'd fix this; we wrote a footnote instead" trade that Reviewer 2 notices. Neither is severe.

The cache-mtime-precedes-code-mtime ordering is a project-hygiene observation, not a correctness issue.

The substance of the regret comparison, the oracle/structured-learner near-zero regret, the visible collapse tragedy, and the parameter recovery table are all clean.

**Phase B1 close eligibility:** Five of six items satisfied (one through code, four through code or footnote). B1.6 unmet. Strict reading of the spec ("close B1 when all six are LANDED / FOOTNOTED / disclosed") fails on B1.6. Lenient reading (the sim is publishable without it because the collapse is verified by the collapse-fraction diagnostic) allows close, but Reviewer 2 will still write a snarky comment.

## Deferred to next session

- **B1.6 (≥25% on its own).** Add a stock + harvest trajectory panel (3 subplots: oracle vs MB-DP vs Myopic, or a 7-panel grid). Requires `compute_paradigm` to also store per-seed `s_curves` and `h_curves` (currently discarded). Cache invalidation needed for all paradigms, which is a real cost. Suggest: store the *seed-0* trajectory only to keep cache small, and add a second panel to `generate_outputs`. Roughly 30-60 minutes of work.

No defect ≥50% surfaced in this re-audit.

**Bullshit score: 30%** — Reviewer 2 catches the missing B1.6 trajectory figure plus the disclosure-only treatment of B1.3 and B1.4, and writes a "revise and resubmit with the trajectory panel" comment, but the regret-comparison substance and parameter-recovery story survive intact.
