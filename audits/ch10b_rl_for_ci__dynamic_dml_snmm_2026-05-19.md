# Audit: ch10b_rl_for_ci/sims/dynamic_dml_snmm.py

**Date:** 2026-05-19
**Diagram-only:** no (200 MC reps x 5 sample sizes x 3 estimators; substantive simulation)
**Cited tex file(s):** `ch10b_rl_for_ci/tex/rl_for_ci.tex` (Sec. `subsec:dml_dte` and `subsec:simstudy`, sub-block "Sim 1: dynamic DML on a 2-stage SNMM", lines 286-318).
**Cited paper PDFs read:** `papers/lewis2021dml.md` (Lewis & Syrgkanis 2021, "Double/Debiased ML for Dynamic Treatment Effects via g-Estimation," arXiv:2002.07285 / NeurIPS 2021). Also referenced (no separate read) `papers/robins2000msm.md`, `papers/schulte2014qlearning.md`.

## 1. Algorithm Identity

The script claims to implement Algorithm 1 of Lewis-Syrgkanis 2021 on a two-stage SNMM. Verified term by term:

- **Cross-fitting.** `_crossfit_predict` uses sklearn's `KFold(n_splits=5, shuffle=True)`, fits on training fold, predicts on held-out fold. Matches "K-fold cross-fitting" of Algorithm 1, with K=5 (configurable).
- **Stage-2 (final period) moment.** Code computes `q2_hat = E[Y | H2]` and `p2_hat = E[T2 | H2]` with `H2 = (X1, T1, X2)`, then `psi_2_hat = sum(Y_til_2 * T_til_22) / sum(T_til_22^2)`. This is the residual-on-residual Neyman-orthogonal moment of Eq. (2) in the paper (and equivalently the Robinson 1988 partial-linear estimator at t=m). Correct.
- **Peel-off / calibrated outcome for psi_1.** The code does NOT literally form `Y_bar_1 = Y - psi_2_hat * T_2` and regress on residualized T_1. Instead it residualizes the *raw* Y on X_1 (i.e. `q1_hat = E[Y|X_1]`) and substitutes the peel-off as a correction term inside the numerator: `((Y_til_1 - psi_2_hat * T_til_21) * T_til_11).sum() / sum(T_til_11^2)`. This is in fact the exact Eq. (3) of Algorithm 1 in Lewis-Syrgkanis when the moment is expanded: `tilde_Y_t = Y - q_t(X_t)`, `tilde_T_{j,t} = T_j - p_{j,t}(X_t)` for j >= t, and the estimating equation at period t plugs in `psi_2_hat` on the j=t+1 future-treatment term. The comment in the code (lines 338-344) explicitly notes this is the algorithm's form, not the "calibrated outcome" shorthand. Algebraically equivalent to peel-off when q_t is the regression of Y (not Y_bar_t) on X_t -- this is the form Algorithm 1 actually uses (the paper at line 122-124 of the .md confirms "residual-on-residual estimation approach with target outcome the 'calibrated' outcome" but the formal Algorithm 1 spec residualizes Y directly and absorbs the peel-off into the moment).
- **Three nuisances at stage 1.** `q1_hat = E[Y|X_1]`, `p11_hat = E[T_1|X_1]`, `p21_hat = E[T_2|X_1]`. All three are required because both treatments appear in the t=1 moment. Matches Algorithm 1, lines 152-156 of the paper digest ("for j = t, ..., m, fit p_{j,t}").
- **ML for nuisances.** Lasso (LassoCV with 10 alphas, internal 3-fold CV) for the outcome regression; L1-logistic regression (LogisticRegressionCV with Cs=5) for both stage propensities. Matches the "high-dimensional sparse Lasso nuisance" instantiation discussed in Section 5 of the paper (where Lasso rates of o(n^{-1/4}) are derived under sparsity s = o(n^{1/4})).
- **Variance estimator.** Z-estimator sandwich `IF_i = T_til * resid / J`, `se = sqrt(sum(IF^2))/n`. This is the standard influence-function-based asymptotic variance from Theorem 4. Correct construction (note that Z-estimator SE for a 1-parameter residual-on-residual scalar regression coincides with Eicker-Huber-White on the residualised regression).
- **Plug-in vs orthogonal contrast.** The script does NOT explicitly run a "plug-in / non-orthogonal" baseline for comparison. The Naive OLS and MSM baselines play the role of "non-DML" comparators. The narrative claim "naive panel regression and MSM both fail" is what the paper demonstrates in its Section 6 simulations. This is a defensible interpretation of the audit prompt's "plug-in vs orthogonal vs oracle" -- the role of "plug-in" is played by Naive OLS (which would correspond to a non-residualised plug-in estimator with no propensity correction).

One minor design choice worth flagging: the stage-2 propensity `p2_hat = E[T_2 | X1, T_1, X_2]` is correct because T_1 is part of history at t=2. The code includes T_1 as a column in `H2`. Good.

No fake/stub implementation. The estimator is the real Lewis-Syrgkanis recursion.

## 2. Environment / MDP Fidelity

The DGP in `generate_panel` matches the tex equation (eq:simB1_dgp, lines 295-300):
- `X1 ~ N(0, I_p)` with `p=20` (matches "state dimension p=20").
- `T1 ~ Bernoulli(sigma(gamma' X1))` with `||gamma||_2 = 1.5` supported on first s=5 coords (matches).
- `eta1 ~ N(0, sigma_eta^2 I_p)` with `sigma_eta = 0.6` (matches).
- `X2 = B X1 + alpha T1 + eta1` with `||B||_op = 0.5` and `alpha = e_1 * ALPHA_NORM = e_1 * 1.0` (tex says `alpha = e_1`, but with implicit unit norm; code's `alpha[0] = ALPHA_NORM = 1.0` matches).
- `T2 ~ Bernoulli(sigma(gamma' X2))` (matches).
- `Y = psi_1* T1 + psi_2* T2 + mu' X1 + nu' eta1 + eps` with `eps ~ N(0, 0.5^2)` (matches eq:simB1_dgp exactly).
- `nu = 2.0 * gamma / ||gamma||` (matches "outcome-coupling vector nu = 2.0 gamma / ||gamma||").
- `(psi_1, psi_2) = (1.0, 0.5)` (matches).

The DGP is the partially-linear Markovian model of Lewis-Syrgkanis 2021 Section 2.1 (paper digest line 90: "if we view the problem as a simultaneous treatment problem... we essentially have a problem of unmeasured confounding"). The `nu' eta_1` term in Y is the canonical treatment-confounder-feedback construct: eta_1 enters Y AND drives X_2 which determines T_2.

The tex caption claims "treatment-confounder feedback" and the code implements exactly that. The `nu` alignment with `gamma` (rather than independent) is a slightly aggressive choice that maximizes the bias signal but is described explicitly in both the code comment (lines 141-146) and the tex (line 300, "the outcome-coupling vector nu = 2.0 gamma / ||gamma|| is aligned with the propensity direction"). Honest, not hidden.

No environment mismatch detected.

## 3. Data Integrity

- `compute_data` is wrapped in the standard `compute_or_load` per-component cache. `SHARED_CONFIG` includes all DGP parameters; `NAIVE_CONFIG`, `MSM_CONFIG`, `DML_CONFIG` extend `SHARED_CONFIG` with `'method'` tag. Config changes invalidate caches. Correct usage.
- `run_estimator` actually runs the estimator: it generates a fresh panel per (n, seed), calls the appropriate `fit_*` function, and stores `psi_hat` and `se_hat`. No hardcoded "expected" values dropped into tables.
- The stdout/table values flow through `summarize -> make_table/make_figure/print_stdout`. `summarize` computes `bias = mean(psi) - PSI_TRUE`, `rmse = sqrt(mean((psi - PSI_TRUE)^2))`, `coverage = mean((lo <= true) & (true <= hi))`. All standard, no shortcuts.
- The .tex table values (e.g. Naive OLS bias=-0.191 on psi_1, +0.933 on psi_2; DML coverage 0.96 on psi_1, 0.93 on psi_2 at n=4000) exactly match the stdout. Tex prose claim "0.93 on psi_2" matches +0.933 from the table.

Hostile reviewer's only concern: the stdout file references `ch11_rl_for_ci/sims/...` paths, not `ch10b_rl_for_ci/...`. This is leftover from the chapter rename and does not affect the numbers, but it is a cosmetic data-integrity smell (the file was generated under the old path). A re-run under the new path would replace these strings.

Tex prose also still says "Sim 1 source: \texttt{ch11\_rl\_for\_ci/sims/dynamic\_dml\_snmm.py}" in the footnote (line 304). Wrong directory. Will get caught by a reviewer who tries to navigate to the file.

## 4. Comparison Fairness

- Same `(n, seed)` pair generates one fresh panel in `run_estimator`; the panel is shared across estimators because the same `(n*10_007 + s)` seed is used in every method's loop. Each method sees the same DGP draws -- no method gets "easier" data.
- Same N_SEEDS=200 across methods, same N_GRID. Same evaluation protocol (sandwich-style SE -> Wald 95% CI -> empirical coverage indicator).
- Same propensity-model family (L1-logistic CV) is used for MSM/IPTW and for DML's stage propensities, so the comparison is not biased by giving DML a stronger ML stack. (Both use `LogisticRegressionCV` with similar regularization sweeps.)
- All three methods get the full panel `(X1, T1, X2, T2, Y)` and choose which to use: Naive OLS deliberately omits X_2 (this is the headline "fail" case from L-S Section 6), MSM uses X_1 for stage-1 propensity and (X1, T1, X2) for stage-2 (the standard Robins-Hernan-Brumback specification), and DML uses the full history.

Fair comparison. The hostile reviewer's complaint here would be: Naive OLS is a "straw man" that omits X_2, and a more sophisticated econometrician would have controlled for X_2 even in a naive run. The code's docstring (line 192) and tex (line 302) both make clear this is the "init-ctrls" baseline of the paper, exactly the contrast Lewis-Syrgkanis 2021 Section 6 sets up. Defensible but not the strongest possible baseline. A stronger version would also report "controls on X_2 too" OLS (which is also biased because X_2 is post-treatment for T_1, the post-treatment-bias channel).

## 5. Theoretical Sanity Checks

Lewis-Syrgkanis Theorem 4 predicts:
- (a) sqrt(n)-consistency of Dynamic DML: bias should shrink ~1/sqrt(n), RMSE should shrink at 1/sqrt(n).
- (b) Asymptotic normality with influence-function variance => empirical 95% coverage near 95%.
- (c) Naive plug-in (here: Naive OLS omitting X_2) has irreducible bias (omitted variable: post-treatment X_2 drives both Y via mu/nu and T_2 via gamma).
- (d) MSM is consistent under sequential ignorability (which holds here: no unobserved confounders, just high-dimensional observed state), but has slow finite-sample convergence in high dimensions because IPTW weights are noisy.

Empirical results vs theory:

| Check | Predicted | Observed (n=4000) | Verdict |
|-------|-----------|-------------------|---------|
| DML psi_2 RMSE shrinks ~1/sqrt(n) | 0.084 -> 0.018 (factor ~4.7) over n=250->4000 (sqrt-ratio = 4) | Matches | Pass |
| DML psi_1 RMSE shrinks ~1/sqrt(n) | 0.195 -> 0.046 (factor ~4.2) | Matches | Pass |
| DML coverage ~95% | 0.91-0.97 across grid | Mostly within 2 SE of 0.95 (sqrt(0.95*0.05/200) ~= 0.015, so 2-SE band is [0.92, 0.98]) | Pass |
| Naive OLS bias on psi_2 doesn't shrink | 0.92 stable across grid | Matches | Pass |
| Naive OLS coverage approaches 0 | 0.00 across grid | Matches | Pass |
| MSM consistent but slow | psi_2 bias falls 0.68 -> 0.15, coverage 0.38 -> 0.73 | Matches paper's Section 6 qualitative finding | Pass |

The bias direction on Naive OLS for psi_1 is negative (-0.19), as the code comment predicts: "bias on psi_1 of approximately mu'alpha (the indirect path T_1 -> X_2 -> Y absorbed into the T_1 coefficient)". And on psi_2 it is positive (+0.93), consistent with the omitted post-treatment confounder X_2 channel that includes the strong nu'eta_1 coupling.

DML coverage of psi_2 at n=250 is 0.91 (under-coverage by ~4 points). This is the kind of finite-sample undercoverage you would expect from a recursive nuisance-stacked estimator at small n; it tightens to 0.93-0.97 as n grows. Not a problem.

No "DML beats oracle" or "all methods identical" pathologies. Theoretical predictions hold tightly.

## 6. Information Leakage

- The estimators see only `(X1, T1, X2, T2, Y)` -- the observable panel. The true parameters `(psi_1*, psi_2*)`, the population matrices `(B, alpha, gamma, mu, nu)`, and the unobserved shocks `(eta_1, eps)` are not passed to any estimator function. Verified in `run_estimator` signature: `fn(X1, T1, X2, T2, Y)`.
- `make_population_params` is called inside `compute_shared` and only returns `pop` which is consumed by `generate_panel`. The estimators never receive `pop`.
- The script does NOT include a separate "oracle" estimator (e.g. one that uses true propensities or true `mu, nu`). The audit prompt mentions "oracle sees the true blip" but the script makes a different design choice: PSI_TRUE is used only for evaluation (bias, coverage) and not as an input to any fit. This is fine -- there is no oracle row in the table that would be at risk of leakage.
- Cross-fit predict uses `KFold` with `shuffle=True, random_state=rng_seed`. The held-out fold is never used for fitting. No within-fit-fold leakage.
- The variance estimator uses the same held-out residuals as the point estimate (`Y_til_2`, `T_til_22`). This is correct -- the influence function is evaluated on cross-fitted residuals, not on in-sample-fit residuals. No bootstrapping-on-training-data leak.

No leakage detected.

## 7. Seed & Reproducibility

- Population-level seed `DGP_SEED = 12345` fixed in `compute_shared`. Same population params (B, alpha, gamma, mu, nu) across all Monte Carlo runs.
- Per-MC seed: `seed = (n * 10_007 + s) & 0xFFFFFFFF` is a deterministic function of (n, s). Reproducible across runs. The factor 10_007 is a prime that prevents seed collisions between different n values for the same s.
- 200 Monte Carlo replications per (method, n). Comfortably above the 10-seed minimum.
- Mean and standard-error implied (SE from MC variance can be derived from RMSE/sqrt(N_SEEDS); the table reports the empirical bias and RMSE which together encode MC variance).
- The coverage point estimates carry MC standard errors of about sqrt(0.95*0.05/200) ~= 0.015, so the n=4000 DML coverages (0.965, 0.930) are within 2 SE of nominal.
- Cross-fitting `rng_seed` for DML is also threaded through (the `rng_seed` argument to `fit_dynamic_dml` propagates into `KFold(random_state=...)`, `LassoCV(random_state=...)`, and `LogisticRegressionCV(random_state=...)`). Reproducible.

One minor reproducibility wart: `compute_or_load` keys off the config dict, but `SHARED_CONFIG` includes `PSI_TRUE.tolist()` which is fine. If anyone changed `S_SPARSE` or any DGP knob, the shared cache would correctly invalidate.

Reproducible.

## Hostile-Reviewer Summary

The simulation implements the Lewis-Syrgkanis 2021 dynamic DML algorithm correctly: cross-fitted Lasso/L1-logistic nuisances, residual-on-residual moment at stage 2, peel-off form at stage 1, Z-estimator influence-function SEs. The DGP is the partially-linear Markovian model from L-S Section 2.1 with the canonical treatment-confounder-feedback structure (nu aligned with gamma). Empirical results match all four theoretical predictions: DML achieves near-nominal 95% coverage and sqrt(n)-RMSE-shrinkage, Naive OLS carries an irreducible bias and 0% coverage, MSM is consistent but converges slowly. 200 MC reps, 5 sample sizes, seeds fixed and reproducible.

Cosmetic issues a reviewer 2 would catch: (i) the tex footnote and the stdout file still say `ch11_rl_for_ci/...` instead of `ch10b_rl_for_ci/...` (chapter-rename leftover, paths are wrong but data is correct); (ii) the "naive OLS" baseline omits X_2 to maximize the bias signal -- a defensible "init-ctrls" choice from the paper's Section 6 but a stronger sim might also include a "control on X_2" baseline to show the post-treatment-bias channel separately; (iii) no explicit "plug-in DML" (non-orthogonal) row, only Naive OLS and IPTW play that role.

None of these are substantive: the algorithm is real, the bias/coverage curves are correct, the comparison is fair, no leakage. The numbers in the tex prose match the table and stdout to 3 decimals.

**Bullshit score: 10%** -- Reviewer 2 catches the `ch11_rl_for_ci` path typo in the footnote and stdout file, and might mutter that Naive OLS is a slightly straw-man baseline (omits X_2), but the substance -- algorithm identity, DGP fidelity, theoretical sanity, leakage, reproducibility -- all hold. The headline coverage and bias numbers replicate the paper's claims tightly enough that the reviewer would accept as-is after the path-rename cleanup.
