# Audit: ch10_causal/sims/counterfactual_ope.py

**Date:** 2026-05-19
**Diagram-only:** no — Monte Carlo simulation with 20 seeds across 4 sample sizes and 2 scenarios.
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch10_causal/tex/causal_rl.tex` §`subsec:cfope_sim` (lines 288–323), references `subsec:cfope_scm` (line 229) and `alg:buesing_cfope` (line 238).
**Cited paper PDFs read:**
- `ch10_causal/papers/2006.02579v1.md` (Sun et al. 2021 survey, "Counterfactually-guided policy evaluation", §5.1, Algorithm 1, lines 197–242). Confirms abduction-action-prediction structure.
- `ch10_causal/papers/d.md` (Causal RL survey draft) — confirms three-step abduction-intervention-prediction characterisation.
- Original Buesing 2019 ("Woulda, coulda, shoulda") *not present* in `papers/`; cited via refs.bib key `buesing2019woulda` (refs.bib:4105). Robins-Rotnitzky-Zhao primary not present either (Robins1994 at refs.bib:4160 is the IPW paper, not the DR paper).

---

## 1. Algorithm Identity

The script implements three estimators on a single-step (contextual-bandit) substrate:

- **IS** — per-decision IS with deterministic target: `rho = 1{a == pi_tilde(x)} / pi_obs(a|x)`, `V_IS = mean(rho * y)` (script lines 142–154). This is the standard IPW form for a deterministic target. ✓ matches the equation in tex line 300.

- **MB** — OLS plug-in: fit `theta` on full data, evaluate `mean(f_hat(x, pi_tilde(x)))` (lines 157–162). ✓ matches tex.

- **CF** — labelled "counterfactual-augmented" / "doubly-robust" (lines 165–187):
  `V_CF = mean(f_hat(x, pi_tilde(x))) + mean(rho_i * (y_i - f_hat(x_i, a_i)))`.
  This is algebraically the AIPW / classical doubly-robust estimator (Robins-Rotnitzky-Zhao 1994, also Bang-Robins 2005), not the Buesing (2019) abduction-action-prediction rule. The script header (lines 6–11) and the tex footnote both acknowledge this honestly: the naive Buesing-style average of `y'_i = f_hat(x_i, pi_tilde(x_i)) + (y_i - f_hat(x_i, a_i))` collapses to `V_MB` exactly under OLS with intercept because residuals sum to zero. The importance-weighted residual correction restores bias cancellation. This is mathematically correct and the documentation is transparent.

  **Caveat — a hostile reviewer will notice:** the estimator labelled "CF" is the textbook AIPW / DR estimator. Calling it the "counterfactual" estimator (and citing Buesing) overstates the connection. Buesing's algorithm operates at the trajectory level under a known/learned SCM with abduction over exogenous noise; here there is no exogenous-noise abduction, no per-trajectory rollout, and no SCM beyond the linear outcome model. The tex acknowledges this in a footnote ("coincides with the classical doubly-robust estimator of Robins-Rotnitzky-Zhao") but the section title, figure labels, and table caption still say "Counterfactual" / "CF estimator". A skeptical reader will read the section as overclaiming a Buesing implementation when the actual content is single-step AIPW.

## 2. Environment / MDP Fidelity

DGP code (`generate_data`, `outcome_mean`, `behavior_prob_a1`) matches the tex equations exactly:
- Code line 84: `beta0 + bx1*x1 + bx2*x2 + ba*a + bxa1*a*x1` = tex (eq. cfope_dgp) `y = 1 + 0.5 x1 - 0.3 x2 + 2 a + a*x1 + u`. ✓
- Behavior logit `0.5 + 1.0*x1 - 0.5*x2` = tex `sigma(0.5 + x1 - 0.5 x2)`. ✓
- Target `1{x1 + x2 > 0}`. ✓
- `u ~ N(0, 0.5^2)`. ✓

**Issue (cosmetic but real):** Script-header comment (line 2) says "Chapter 12 (forecasting and reinforcement learning), §5.3 simulation." The file has since moved to `ch10_causal/`. Git status confirms the move from `ch12_forecasting_rl/`. The stdout file at `counterfactual_ope_stdout.txt` (lines 7–11) still reports paths under `ch12_forecasting_rl/sims/`, so it was not regenerated after the move — a direct violation of the `feedback_update_stdout.md` rule recorded in MEMORY.md.

## 3. Data Integrity

`compute_data` runs the Monte Carlo end-to-end:
- `compute_oracle` runs 10^6 draws under the DGP and the target policy (line 99–104). ✓
- `run_scenario` loops over (n, seed), regenerates data, computes all three estimators per cell, stores arrays of estimates (lines 193–220). ✓
- Bias / std / RMSE are computed against the cached `oracle` (lines 215–217). No hardcoded numbers; everything traces to seeded RNG and the oracle MC.

The reported numbers in `_stdout.txt` (Bias / Std / RMSE) match the `_table.tex` to three decimals (e.g., Well-spec MB bias −0.001, RMSE 0.056). ✓

## 4. Comparison Fairness

For each (n, seed) cell the same `data` dict is passed to all three estimators (lines 209–211). Same `(x, a, y)`. ✓
Same number of seeds (20) across estimators and scenarios. ✓
Same propensity input across IS and CF. ✓

**Minor:** seeds across n's are disjoint (`seed = 1000*(i+1) + s`, so each n has its own seed block). This means n=200 and n=2000 are *not* paired (the n=2000 sample is not a superset of the n=200 sample). For a clean log-log RMSE plot you'd prefer nested samples, but it does not affect the apples-to-apples comparison across estimators within an n.

## 5. Theoretical Sanity Checks

Stdout at n=1000, 20 seeds:

| Scenario | Estimator | Bias | Std | RMSE |
|---|---|---|---|---|
| Well-spec | IS | +0.0174 | 0.0996 | 0.0987 |
| Well-spec | MB | −0.0009 | 0.0572 | 0.0558 |
| Well-spec | CF | +0.0031 | 0.0665 | 0.0649 |
| Misspec | IS | +0.0174 | 0.0996 | 0.0987 |
| Misspec | MB | −0.0716 | 0.0605 | 0.0928 |
| Misspec | CF | +0.0024 | 0.0777 | 0.0757 |

- Under well-specification all three estimators are roughly unbiased. ✓ Consistent with theory.
- Under misspecification, MB picks up the OLS projection bias from omitting `a*x1`. The CF (AIPW) estimator with the same misspecified outcome model but the *correct* propensity remains nearly unbiased. ✓ This is exactly the double-robustness textbook result.
- IS is invariant across scenarios (it doesn't use the outcome model). ✓
- CF has higher variance than MB under well-spec (Std 0.0665 vs 0.0572) — the cost of the importance-weighted correction. ✓ standard finite-sample trade-off.

**Hostile reviewer note:** The "double robustness" claim is only one-sidedly tested here. The propensity is *given exactly* by the DGP, not estimated. To genuinely demonstrate DR you'd need a misspecified propensity (constant or wrong logit) paired with a correct outcome model, and show CF still recovers. The simulation only stresses the outcome-model side. The tex sentence "unbiased under correct propensity *or* correct outcome model" is asserted but only half-tested. The script header (lines 9–11) admits the design but the tex does not flag the limitation.

**RMSE plateau under misspecification:** The figure caption (tex line 319) says "MB curve flattens under misspecification at the population bias". The CF curve continues to decline; the IS curve declines but is dominated by variance. This is what one would expect: at large n, MB-RMSE → |bias| ≈ 0.072 ≈ 0.07, while CF-RMSE → 0 at parametric rate. The log-log slopes are not explicitly reported, but qualitatively the figure is consistent with theory.

## 6. Information Leakage

**Real issue, partly acknowledged.** All three estimators receive `data['p_a1']` — the *true* behavior-policy propensity from the DGP. The estimator functions never estimate propensities; they consume the oracle propensity (script lines 149, 176, 183).

In a publication context this means:
- The "IS" estimator is *known-propensity* IPW, not the typical empirical IPW that estimates the propensity from data.
- The "CF" / DR estimator's bias-cancellation is the easy-mode version: given true `pi_obs`, the AIPW correction is exact in expectation.
- A standard OPE benchmark (e.g., Dudik-Langford-Li 2011, Jiang-Li 2016) would estimate the propensity by logistic regression and include that estimation error in the variance and bias decomposition.

The tex sentence "unbiased under correct propensity *or* correct outcome model" is technically defended because the propensity *is* exactly correct here. But the experiment never stresses the propensity side, so the "doubly-robust" framing oversells what the numbers actually demonstrate. A hostile reviewer would write: "the experiment shows AIPW with known propensity beats OLS plug-in under outcome misspecification; this is a textbook fact, not the doubly-robust property you advertised."

No leakage of `u` (the SCM exogenous noise) into estimators, no peeking at counterfactual y, no cross-contamination. The narrow form of leakage (oracle propensity) is the only one present.

## 7. Seed & Reproducibility

- `numpy.random.default_rng(seed)` used per cell (line 89). ✓
- Oracle uses `default_rng(0)` (line 101). ✓
- 20 seeds, meets the ≥10 standard. ✓
- Means and standard deviations reported (Std column in the table is `ddof=1`, line 216). ✓
- The figure plots point estimates of RMSE (over 20 seeds) without confidence bands. Minor — for log-log RMSE the difference matters more in slope than in level.

Stdout file is stale (refers to `ch12_forecasting_rl/sims/cache/...`); the cache files actually exist at `ch10_causal/sims/cache/`. The stdout would need a rerun for the published artefact to be internally consistent. Listed in MEMORY.md as a hard project rule (`feedback_update_stdout.md`).

---

## Hostile-Reviewer Summary

This is a competent textbook-grade AIPW demonstration with a misleading label. The actual experiment is: linear-DGP contextual bandit, IPW vs OLS plug-in vs AIPW with known propensity, well-spec vs outcome-misspec. The math is right, the code matches the tex equations, the numbers match the table, the well-spec / misspec contrast is the expected one. The CF estimator is genuinely the doubly-robust estimator and the tex footnote admits this. So the substance survives.

What a reviewer will mark down:
1. **Section title and figure labels say "Counterfactual" / "CF" but the estimator is AIPW.** The connection to Buesing's abduction-action-prediction is asserted but the experiment does not implement abduction over exogenous noise — there is no SCM-level counterfactual rollout, only a residual-weighted DR. The tex footnote concedes this; the section title does not. Reviewer 2 will accuse the section of overselling its Buesing connection.
2. **Double robustness is only half-tested.** Propensity is the DGP oracle; outcome model is what is varied. To support the "correct propensity OR correct model" claim, the simulation needs a third scenario: correct outcome model, misspecified propensity (e.g., constant propensity).
3. **Stdout file stale** (paths still point to ch12_forecasting_rl), violating an explicit project rule. Cosmetic but documents a process failure.
4. Script-header comment still says Chapter 12. Cosmetic.

None of these undermine the numbers; they reframe what the numbers actually demonstrate. The grade should reflect "the algorithm shown is not exactly the algorithm named, but the artifact is honest about it in the footnote and the result is qualitatively correct."

**Bullshit score: 25%** — Reviewer 2 catches that the CF estimator is AIPW dressed as Buesing, that only the outcome side of double robustness is stressed, and that the stdout file is stale. The footnote disclosure in tex line 307 prevents this from reading as deception; the numbers are real and the qualitative story (MB plateaus under misspec, CF/AIPW does not) is theoretically correct. A name change ("doubly-robust off-policy evaluation under outcome misspecification") plus a second misspecified-propensity scenario would drop this to 0%.
