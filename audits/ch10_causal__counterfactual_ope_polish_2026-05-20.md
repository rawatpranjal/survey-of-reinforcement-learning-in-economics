# Polish Report: ch10_causal / counterfactual_ope

**Date:** 2026-05-20
**Original score:** 25% (2026-05-19)
**After polish:** 12%

## Scope

Light polish addressing the three "remaining nicks" flagged in the 25% audit:

1. Misleading "CF" label vs Buesing abduction-action-prediction — chose option (B): keep the "CF" label and sharpen the footnote.
2. Stale `ch12_forecasting_rl` paths in script docstring and stdout (script moved chapters).
3. Half-tested double-robustness (only outcome side, propensity always oracle) — add an honest one-line caveat.

No code paths changed; no recomputation needed beyond a stdout rerun to refresh the cached path strings.

## Files modified

- `ch10_causal/sims/counterfactual_ope.py` — header docstring rewritten (lines 1-21). Chapter reference corrected from "Chapter 12 (forecasting and reinforcement learning)" to "Chapter 10 (causal inference for reinforcement learning), §cfope_sim". The "CF" estimator block in the docstring now states explicitly that the implementation is algebraically the Robins-Rotnitzky-Zhao doubly-robust / AIPW estimator and is *not* the Buesing 2019 abduction-action-prediction estimator. Adds a closing line stating the propensity is held fixed at the DGP value so only the outcome-model side of DR is stressed.
- `ch10_causal/tex/causal_rl.tex` — footnote on line 307 rewritten. The first sentence of the footnote now reads: "The estimator labelled $\widehat V_{\mathrm{CF}}$ in equation~\eqref{eq:cfope_dr} is algebraically the Robins-Rotnitzky-Zhao doubly-robust / AIPW estimator adapted to off-policy evaluation; it is *not* the Buesing (2019) abduction-action-prediction estimator, which operates at the trajectory level under a known or learned structural causal model with abduction over exogenous noise and is not implemented here." Footnote concludes with "We retain the 'CF' label for continuity with Section~\ref{subsec:cfope_scm}." A new closing sentence to the paragraph (after the misspec-feature-set explanation) acknowledges the half-tested DR: "The propensity $\pi_{\mathrm{obs}}$ is held fixed at the true DGP value in both scenarios, so only the outcome-model side of the double-robustness claim is stressed here; a misspecified-propensity scenario paired with a correct outcome model would test the other side of the claim and is left as a future check."
- `ch10_causal/sims/counterfactual_ope_stdout.txt` — regenerated. Now points to `ch10_causal/sims/cache/` and `ch10_causal/sims/counterfactual_ope.png` rather than the stale `ch12_forecasting_rl/sims/...` paths. All cache hits (no recompute), numbers byte-identical to prior run.
- `docs/ch10_causal.pdf` — recompiled (21 pages, 1,247,261 bytes). Only undefined reference is `section:rl_for_ci` (cross-chapter, resolves in full build); no new errors.

## What did *not* change

- No code logic in `compute_data`, the three estimators, the oracle, the DGP, or the configs. The numerical results are unchanged: Well-spec MB bias -0.0009, RMSE 0.0558; Misspec MB bias -0.0716, RMSE 0.0928; Misspec CF bias 0.0024, RMSE 0.0757; etc.
- Section title "Counterfactual OPE under a Misspecified SCM" preserved per the user's option (B) instruction — relabeling everything in code, figure, and table is more churn than warranted given the disclosure now appears in both the script header *and* the first sentence of the footnote.
- Estimator labels in figure (`EST_LABELS['CF'] = 'CF (counterfactual)'`) and table preserved. The honest disclosure is in the footnote and the docstring, not the legend.

## Bullshit detector axis check

- **Algorithm Identity (point 1):** The previous 25%-grade complaint was that the CF section title and labels asserted a Buesing connection that the implementation did not justify. The footnote now opens by stating the estimator is the Robins-Rotnitzky-Zhao DR / AIPW form and is *not* Buesing 2019, leaving zero room for a reviewer to read overclaim into the prose. The decision to retain the "CF" label is now defended explicitly ("for continuity with Section~\ref{subsec:cfope_scm}") rather than left implicit.
- **Environment Fidelity (point 2):** The cosmetic Chapter-12 reference in the script header is corrected to Chapter 10. The stdout file's stale `ch12_forecasting_rl` path references — flagged in the audit as a direct `feedback_update_stdout.md` violation — are gone. The artifact now uniformly points to `ch10_causal/`.
- **Theoretical Sanity (point 5):** The half-tested-DR concern is now disclosed in the tex paragraph itself, framed as a future-check rather than a defended claim. The chapter no longer asserts "doubly robust" without also saying "only one side of the DR claim is stressed here."

## Residual issues

- The "CF" label is still in the section title, table caption, figure label, and legend. A more aggressive pass would rename to "AIPW" or "DR-AIPW" everywhere; the user's explicit instruction was to take option (B) and not do that. Reviewer 2 may still write a one-sentence comment ("why not call it AIPW outright?") but the footnote already answers the question on the same page.
- A second misspecified-propensity scenario remains future work; the tex now acknowledges this rather than silently sidestepping it.
- The script-level estimator label `EST_LABELS['CF'] = 'CF (counterfactual)'` (sims script line 273) is purely a figure-legend cosmetic; left unchanged to avoid invalidating the cached figure path. Disclosure lives in the footnote and the script header.

## Score rationale

Anchored at 25% in the original audit. The three drivers were (a) overclaim of Buesing connection without on-page disclosure, (b) stale stdout paths violating an explicit project rule, (c) half-tested DR claim. (a) is now resolved at the strongest opening position of the footnote; the section title still says "Counterfactual" but the very next sentence after the equation makes the algorithmic identity unambiguous. (b) is fully resolved. (c) is now acknowledged in the body paragraph as future work.

The hostile-reviewer reaction collapses from "Reviewer 2 catches that the CF estimator is AIPW dressed as Buesing" to "Reviewer 2 might prefer the section title be renamed, but the footnote concedes the algorithmic identity outright and the half-tested DR is acknowledged." That is a one-sentence quibble, not a substantive overclaim — closer to a 10-15% grade than 25%.

**Bullshit score: 12%** — Reviewer 2 still has room to write "rename the section AIPW already" but the footnote and the closing sentence of the simulation paragraph defuse the substance of the overclaim. Numbers are unchanged, stdout paths are now consistent with the file's actual chapter location, and the half-tested DR caveat is on-page.
