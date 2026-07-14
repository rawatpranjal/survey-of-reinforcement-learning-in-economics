# Audit (DELTA): ch12_world_models/sims/fishery_paradigms.py

**Date:** 2026-07-14
**Type:** DELTA. Prior: original 2026-05-19 (30%), polish 2026-05-20 (12%), re-audit 2026-05-22 (30%, open item B1.6). This is the first audit of the post-reaudit fix.
**Delta summary:** Re-audit closed at 30% with one outright-missing item, **B1.6 (stock + harvest trajectory figure)**. Commit `7ca8556` ("ch12 B1.6: fishery stock+harvest trajectory figure (closes phase B)", 2026-05-22 20:21) is the fix under audit. It (i) adds a `track_traj` path to `rollout`, (ii) records the seed-0 `s_traj`/`h_traj` in `compute_paradigm`, (iii) turns the single regret plot into a two-panel figure (`FIG_DOUBLE`) whose right panel overlays seed-0 stock (solid, left axis) and harvest (dashed, right axis) for all seven paradigms with `K=10` and `rK/4=1.0` reference lines, and (iv) rewrites the figure caption in `s09_dual_sim.tex` line 78. No numbers changed; regret table and recovery table are byte-identical to the re-audited version. No new figure *file* was added (the existing `fishery_paradigms.png` was regenerated with a second panel; confirmed via `git show 7ca8556 --stat`).

**Files read end to end:**
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms.py`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_env.py`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms_stdout.txt`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms_results.tex`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms_recovery.tex`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms.png` (viewed)
- `/Users/pranjal/Code/rl/ch12_world_models/tex/s09_dual_sim.tex`
- `/Users/pranjal/Code/rl/ch12_world_models/sims/fishery_paradigms_audit.md`
- prior audits: `..._2026-05-19.md`, `..._polish_2026-05-20.md`, `..._reaudit_2026-05-22.md`
- `git show 7ca8556`

---

## Step 3: what this part of the chapter claims, and what the sim is evidence for

(i) **Theoretical claim.** §9.2 (the fishery panel of the dual simulation) places learning paradigms on an inductive-bias-vs-data frontier in an *exogenous* non-linear environment (a logistic-growth renewable stock, where the agent's action depletes the stock but does not feed back through expectations, unlike the self-referential cobweb). The claim is that on a clean identification problem each $(s_t, s_{t+1})$ pair directly informs $(r, K)$, so structured model-based learners (RLS, model-based DP) reach near-oracle regret and recover parameters within a few percent; tabular Q-learning is an order of magnitude worse; a wrong-functional-form genetic algorithm and a myopic open-access agent are worst; and the myopic open-access agent reproduces the textbook bioeconomic tragedy (fishery collapse).

(ii) **What the sim is evidence for.** (a) the sample-efficiency ordering by inductive bias; (b) parameter-recovery convergence for the two structured learners; (c) the bioeconomic-collapse comparison, i.e. that the myopic open-access agent collapses the stock and earns the worst regret. The new right panel added by the delta is specifically the visual evidence promised for (c): the re-audit's deferred note (line 117) asked for it so "a reader can see the Myopic collapse on step one and the structured-learner stabilization near $s^\star \approx 4.4$."

---

## Criteria verdicts

### (a) CORRECTNESS — one real defect; substance otherwise holds

The regret pipeline, oracle DP, paired-noise protocol, and recovery extraction are all correct and unchanged from the re-audited version (data-integrity and comparison-fairness re-verified in the 7-point section below). The delta's `track_traj` addition is a clean, side-effect-free instrumentation of `rollout` (`fishery_paradigms.py:395-417`) and does not alter any computed number.

The defect is in the **myopic-collapse story, which the new figure was meant to verify and instead contradicts.** Both the code and the prose assert the myopic agent harvests $p/c = 10$ and drives the stock to zero on the first step:

- `MyopicPolicy` docstring (`fishery_paradigms.py:90-96`): "h = p / c ... so the fishery collapses on the first step when s_0 = K = 10 (full harvest) and persists at zero stock thereafter."
- Tex `s09_dual_sim.tex:70`: "the unconstrained interior solution $h_t^{\textsc{my}} = p/c = 10$, large enough to drive the stock to zero on the first step under the present parameters."
- Tex `s09_dual_sim.tex:73`: "the myopic open-access agent ... accumulates seven hundred and fifty-three units of regret because it harvests the stock to zero on the first step and earns nothing thereafter."

This is false. `FisheryEnv.step` (`fishery_env.py:34`) clips every agent's realized harvest to `min(self.s, self.h_max)`, and `self.h_max = 1.5 * h_msy = 1.5 * rK/4 = 1.5` (`fishery_env.py:22-23`). The myopic agent *requests* 10 but the environment caps the realized harvest at 1.5 (= 1.5x MSY). Deterministic trace (harvest clipped to 1.5, logistic growth $rs(1-s/K)$):

```
step  requested_h(=h_traj)  realized_h(env)  s_next
0     10.0                  1.5              8.50
1     8.50                  1.5              7.51
...
6     5.07                  1.5              4.57   <- ~oracle steady state, not zero
11    2.27                  1.5              1.47
12    1.47                  1.47             0.50
14    0.19                  0.19             0.075  <- effectively collapsed
```

The stock reaches zero only after roughly 12-15 steps, not on the first step. For those first ~12 steps the myopic agent harvests 1.5 and earns $ph - (c/2)h^2 = 2\cdot1.5 - 0.1\cdot1.5^2 = 2.775$ per step (**higher** than the oracle's ~1.9), so it does not "earn nothing thereafter." The realized behavior is "harvest 50% above MSY every step -> gradual collapse," a legitimate bioeconomic story but not the "unconstrained $p/c$, instantaneous collapse" the prose dramatizes. The 753.11 regret number is real and unaffected (the sim clips correctly); only the mechanistic explanation is wrong.

The prior audits (including the re-audit, §5 line 69: "$h = p/c = 10$ exceeds $s_0 = K = 10$, so the stock is depleted on step 1") repeated this same false claim without tracing the env clip. It is a shared, uncorrected blind spot, not something the delta introduced. But the delta's new figure now makes it visible (see (b)).

### (b) PRESENTATION / NUMBERS — every published number traces; the new harvest trace is misleading

All published numbers trace to generated artifacts and are mutually consistent (full sweep):

- Regret prose (`:73`) vs `fishery_paradigms_results.tex` vs stdout: RLS 13.67, MB-DP 14.69, Q-Learning 274.71, Naive 447.35, GA 706.13, Myopic 753.11 — all match; table and figure legend are rank-ordered (Oracle, RLS, MB-DP, Q-Learning, Naive, GA, Myopic).
- Recovery prose ("$r$ within two percent, $K$ within three percent, MB-DP slightly tighter") vs `fishery_paradigms_recovery.tex`: RLS $|r\_err|=0.007$ (1.75%), $|K\_err|=0.219$ (2.19%); MB-DP $0.006$ (1.5%), $0.180$ (1.8%). All within claimed tolerances; MB-DP tighter on both. Match.
- Caption reference lines: $K=10$, $rK/4=1.0$ — arithmetic correct.

**The new right panel misplots the myopic harvest.** `rollout` records `h_traj[t] = h` *before* the env clip (`fishery_paradigms.py:406-409`). For the myopic agent `act` returns `min(s, 10)`, which equals the current stock whenever $s<10$, so its recorded harvest trace equals its own stock trace step-for-step and, on a right axis sharing the [0,10] range with the left axis, the orange dashed harvest line lies exactly on top of the orange solid stock line during the entire decline. The realized harvest (flat at 1.5 for ~12 steps) is never shown. For the one paradigm the panel was added to showcase, the dashed "harvest" line is the *requested* harvest retracing the stock, not what the agent actually took from the fishery. All six other paradigms request harvests within $[0, \min(s, 1.5)]$, so their traces are accurate; the defect is myopic-specific.

Secondary presentation notes: the right panel overlays 14 lines (7 solid + 7 dashed) on twin axes, which is cluttered; the re-audit's deferred note suggested a 3-subplot or 7-panel layout instead. And because ~15 steps out of 500 is a near-vertical line at the left edge, the gradual collapse reads as visually "instant" at this zoom, so the figure does not *loudly* contradict the "first step" prose to a casual reader; a code-reading reviewer catches it immediately.

### (c) CHAPTER FIT — largely demonstrates the claim, with the collapse-mechanism caveat

The sim demonstrates the inductive-bias ordering (a) and parameter recovery (b) cleanly and consistently with the prose. For the collapse claim (c), the Myopic agent is genuinely the worst paradigm and the fishery genuinely collapses, so the "textbook tragedy is recovered" conclusion survives. The delta closes B1.6 in the literal sense (a trajectory figure now exists). But the figure the delta shipped, read carefully, disproves rather than confirms the specific "collapse on the first step" claim it was commissioned to verify, and its harvest trace for the collapsing agent is the one uninformative line on the panel. Fit is met at the level of the ordering and the qualitative tragedy; it is not met at the level of the specific mechanism the prose and caption assert.

### (d) EFFICIENCY / STANDARDS — clean

Per-component `compute_or_load` caching with `SHARED_CONFIG` + per-paradigm configs (`:519-544`); `--data-only` / `--plots-only` via `add_component_args`; `N_SEEDS = 20` (>=10) with SE reported throughout; stdout is fact-only, tabular, no opinions. Colors come from the centralized `COLORS` palette via a local `PARADIGM_COLORS` map (acceptable; not `ALGO_COLORS`, pre-existing). Minor pre-existing standards deviations, not delta-scope: `\paragraph*{Setup./Models./Results.}` headers in the tex (project memory says avoid `\paragraph*` headers); the legacy `np.random.seed(s)` at `:490` alongside `default_rng` (flagged in every prior audit, cosmetic). `compute_shared` re-solves the deterministic oracle DP 20 times (once per seed); wasteful but correct.

---

## 7-point checklist

1. **Algorithm identity** — PASS with one caveat. Six learners + oracle match their sources (verified across prior audits; unchanged). The Myopic class computes the correct unconstrained optimum $p/c$, but the environment's `h_max` clip means the *realized* myopic policy is "harvest at 1.5" not "harvest at $p/c=10$"; the class name and docstring describe the request, not the realized behavior. See (a).
2. **Environment / MDP fidelity** — PASS. Schaefer logistic dynamics, linear-quadratic reward, parameters match tex `:67`. The `h_max = 1.5*rK/4` env cap is real and binds for the Myopic agent; it is disclosed in the tex footnote only for Q-Learning/GA/MB-DP exploration (`:70`), **not** for the env-level clip that also caps Myopic. Under-documented, and it is the direct cause of the finding in (a).
3. **Data integrity** — PASS. Regret and recovery numbers in stdout, `_results.tex`, and `_recovery.tex` are mutually identical; no hardcoded values; the delta added instrumentation only.
4. **Comparison fairness** — PASS. All seven paradigms share `FisheryEnv(seed=s)` noise per seed; paired within-seed regret against the oracle's realized return; same $T$, same params, same 20 seeds.
5. **Theoretical sanity** — PASS on the ordering (structured near-oracle, Q-learning mid, GA/Myopic worst; no oracle-beating). CAVEAT: the "100% collapse fraction" diagnostic for Myopic is regret-based (`final_regret >= 0.95 * myopic_floor`), i.e. tautological for the floor paradigm; it does not verify the stock actually reached zero, so it is not independent evidence for the collapse claim.
6. **Information leakage** — PASS (disclosed). Action-support bound and cost-parameter knowledge use true $(r,K,p,c)$; disclosed in tex footnotes (`:70`). Unchanged from prior audits.
7. **Seed & reproducibility** — PASS. 20 seeds, deterministic per-seed RNGs, means and SEs reported; SE magnitudes are consistent with the noise budget.

---

## Prior-audit open-item disposition

| Item (from re-audit 2026-05-22) | Disposition | Evidence |
|---|---|---|
| **B1.6** stock + harvest trajectory figure (was MISSING; the one blocking item) | **RESOLVED (literal) / PARTIALLY REGRESSED (intent).** A trajectory panel now exists (`fishery_paradigms.py:585-610`, right panel of the PNG). But it plots pre-clip requested harvest, so the Myopic dashed harvest retraces the stock and never shows realized harvest; and the panel shows a ~15-step decline, contradicting the "collapse on step one" claim it was meant to verify. | `git show 7ca8556`; `:406-409` (h_traj pre-clip); deterministic trace in (a) |
| B1.1 rename Model-Based LQ -> DP | STILL RESOLVED | `MBPOPolicy.name='Model-Based DP'` (`:304`); registry/order/colors/tex consistent |
| B1.2 myopic agent showing collapse | STILL PRESENT but mechanism misdescribed | `MyopicPolicy` (`:90-110`); collapse is gradual (env clip), not first-step — see (a) |
| B1.3 GA election operator (stock-dynamics term) | STILL FOOTNOTED-ONLY | `_evolve` scores on static $ph-(c/2)h^2$ at `last_s` (`:287-289`); tex footnote `:70` |
| B1.4 action-grid prior from true $(r,K)$ | STILL FOOTNOTED-ONLY | `:188-189, 245, 332`; tex footnote `:70` |
| B1.5 parameter-recovery table | STILL RESOLVED | `_extract_param_estimates` (`:420-428`), `_recovery.tex` written |
| Cache-mtime-precedes-code-mtime hygiene note | N/A this pass (caches re-written by the delta run 2026-05-22 20:18, consistent with code 20:06) | stdout mtimes |

New issue not in prior audits: the env-level `h_max` clip on the Myopic agent, which invalidates the "harvest $p/c=10$ / collapse on step one" narrative that all three prior audits repeated. The re-audit even used it to justify the regret magnitude ("$500\cdot1.5\approx750$"); the 753.11 number is correct but arrived at via a mechanism the audits never verified against the env clip.

---

## Findings, severity-ordered

1. **[correctness/presentation] Myopic agent does not harvest $p/c=10$ and does not collapse the stock on the first step; the environment caps realized harvest at $h_{\max}=1.5$.** `fishery_env.py:34` clips harvest to `min(s, h_max)` with `h_max=1.5` (`:22-23`). Realized myopic harvest is 1.5/step, the stock declines over ~12-15 steps, and the agent earns 2.775/step (above the oracle) during the decline. This contradicts the prose (`s09_dual_sim.tex:70` "drive the stock to zero on the first step", `:73` "harvests the stock to zero on the first step and earns nothing thereafter") and the code docstring (`fishery_paradigms.py:90-96`). Fix: correct the prose/docstring to "harvests at the maximum allowed rate $1.5\,h_{\text{MSY}}$, above MSY, driving the stock to zero within ~15 steps," and document the env-level clip. The regret number and ranking are unaffected.

2. **[presentation] The new right panel plots requested (pre-clip) harvest, so the Myopic dashed harvest line retraces its stock and never shows the realized flat-1.5 harvest.** `rollout` stores `h_traj[t]=h` before `env.step` clips it (`fishery_paradigms.py:406-409`); `MyopicPolicy.act` returns `min(s,10)=s` while $s<10$. On the shared [0,10] twin axis the orange dashed line sits on the orange solid stock line. Fix: record realized harvest (the clipped value returned/derivable from `env.step`) for `h_traj`. Affects only the myopic trace; the other six are accurate.

3. **[presentation, minor] Right panel overlays 14 lines on twin axes (cluttered) and has no $s^\star=K/2=5$ / discounted-optimum reference line**, so the structured learners' ~4.3 stabilization is unlabeled. Not a correctness issue; the caption does not overclaim. Consider the re-audit's suggested 3-subplot layout.

4. **[sanity, minor] The "collapse fraction" diagnostic is regret-based, not stock-based**, so "Myopic 100% collapsed" is tautological for the floor paradigm and is not independent evidence that the stock reached zero (`fishery_paradigms.py:640-650`).

---
**Bullshit score: 30%** — Reviewer 2, or any reviewer who reads the env, catches that `FisheryEnv` caps realized harvest at $1.5\,h_{\text{MSY}}$, so the myopic agent neither harvests $p/c=10$ nor collapses the stock on the first step (it declines over ~15 steps and out-earns the oracle meanwhile), directly contradicting the prose and the very trajectory figure this commit shipped to verify it; the 753 regret, the paradigm ranking, and the recovery story are all correct, so the substance survives revision. Base 25% (a specific wrong sentence + caption), rounded up for the figure-prose contradiction living in the artifact the delta added.
