# Audit: ch07_bandits/sims/curve_learning_pricing.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch07_bandits/tex/dynamic_pricing.tex` (Section "Simulation Study: Structural Knowledge and Curve Learning", lines 158–190; cites Weaver2025, Auer2002, Thompson1933, Misra2019 in surrounding text)
**Cited paper PDFs read:** `denboer2015_dynamic_pricing_survey.{md,pdf}` (skimmed §1–§3.1.2 to confirm scope), Weaver2025 entry confirmed in `docs/refs.bib`. Other key refs present in `papers/`: `broder2012_dynamic_pricing_parametric.{md,pdf}`, `auer2002_finite_time_multiarmed_bandit.{md,pdf}`, `thompson1933_likelihood_two_samples.{md,pdf}`. Weaver 2025 PDF itself is not in `papers/`; the bib entry is.

## 1. Algorithm Identity

Five algorithms run; one (`PricingUCB`, line 131) is defined but never registered in `make_algorithms` (line 395) nor in `ALG_NAMES` (line 62). Dead code, no impact on results.

- `PricingTS` (line 158): Bernoulli/Beta TS over demand, action chosen by `argmax(p · θ̃)` where `θ̃_k ~ Beta(α_k, β_k)`. Uniform Beta(1,1) prior. Matches the tex description "independent-arm Thompson Sampling" and is canonical. Verified update rule (line 171–173): `α += sales`, `β += n - sales`. Correct.
- `DemandGP` (line 230, both UCB and TS modes): RBF-kernel GP over the K=10 price grid with lengthscale 0.18, variance 0.20, fixed. UCB uses `μ + β·σ` with `β = 1.8` (constant, not the Srinivas2010 schedule `β_t = 2 log(t² π²/(6δ))`). TS draws `θ̃ ~ N(μ, Σ)` and selects `argmax(p · clip(θ̃, 0, 1))`. The `β = 1.8` constant is a compromise relative to the theoretical schedule — a hostile reviewer will note the GP-UCB regret bound in Srinivas2010 needs the increasing schedule, so the empirical regret here is not a clean test of the theoretical rate.
- **The GP prior mean is `μ_0(p) = clip(1 - p, 0, 1)` (line 240).** This is a strong, informative prior that bakes in downward-sloping demand. Combined with the monotone variants enforcing `D'(p) ≤ 0`, the "TS vs GP-TS-M" comparison is partly a contest of "uniform-prior independent-arm TS" vs "shape-informed GP". A reviewer asking "is the GP win really from information sharing or just from the prior?" has an opening.
- `MonotoneDemandGP` (line 265): the cleanest piece. Joint GP over `[D(0), D(p_1..K), D'(0), D'(p_1..K)]` with RBF cross-covariances `cov_f_deriv`, `cov_deriv_f`, `cov_deriv_deriv` (lines 190–207). Posterior conditioned on a batch sale-rate observation `y = sales/n` via `gaussian_update` (line 209). Curve sampling: Gibbs sweep over derivatives constrained to be negative (`_sample_negative_derivatives`, line 327, via inverse-CDF truncated normal), then conditional draw of intercept `D(0)`, then trapezoidal integration to reconstruct `D(p_1..K)`. Matches the tex footnote ("truncated-normal derivative draws ... reconstruct demand by integrating the derivative path"). UCB-M uses the `ucb_quantile=0.9` of `ucb_samples=8` constrained-curve draws as the optimistic estimate — an approximation of the upper bound, not a frequentist confidence band. Reasonable engineering choice but not a theoretically-grounded UCB.

Verdict: TS and GP-TS/GP-UCB are textbook; the monotone GP is a custom but well-implemented derivative-constrained GP. β=1.8 fixed and the informative GP prior mean are the main reviewer hooks.

## 2. Environment / MDP Fidelity

- WTP `v ~ Beta(a, b)` for `(2,9), (2,2), (9,2)` — matches tex line 174.
- K=10 prices on `{0.1, 0.2, ..., 1.0}` (line 40) — matches tex.
- T=2,500 customers, batch=10 (lines 36, 39, 430) — matches tex.
- N_SEEDS=1000 (line 37) — matches tex.
- Sale rule `sales = sum(v_i >= p)` with `v_i` drawn fresh per batch (line 432, 437) — unit demand with truthful WTP, matches Weaver's setup.
- Per-batch demand observation `y = sales/n` plugged into the GP with noise variance `0.25/n_customers` (line 260, 385) — using the Bernoulli worst-case variance bound `Var(Bernoulli(p)) ≤ 1/4`. This is conservative (true variance is `p(1-p)/n ≤ 1/(4n)`) but acceptable.
- `BetaWTPDemand.true_opt_price` and `true_opt_profit` (line 114–115) computed on a 200K-point grid via `argmax(p · sf(p; a, b))`. Continuous Beta-WTP profit is unimodal so argmax is well-defined. The `Π*_P` (grid optimum) and `Π*` (continuous optimum) reported in the table are computed cleanly.

No environment / paper mismatch. Beta-WTP unit-demand with batched updates and a finite price grid is the Weaver2025 setup.

## 3. Data Integrity

- `compute_data()` (line 481) calls `load_results` first, then if no cache runs all seeds via `mp.Pool` for each scenario. Each `run_one` (line 417) actually runs `T/BATCH_SIZE = 250` batches and dispatches to all five algorithms with the same `valuations` array.
- `_print_summary` (line 461) and the table generator (line 592) pull from `profit_arrays[name][:, idx]` — computed values, no hardcoded magic numbers.
- The `version: 9` cache key forces invalidation if any algorithm-config change happens, but the script does NOT include kernel hyperparameters (`lengthscale=0.18`, `variance=0.20`, `β=1.8`, `gibbs_sweeps`, `ucb_quantile`) in `CONFIG`. A change to those would silently use the cached results. This is a real cache-staleness risk: hyperparameter edits don't invalidate. Flag.
- Stdout file is plausibly stale: "Loaded from cache." on line 1 of `curve_learning_pricing_stdout.txt`. The displayed values match `summary.tex`, so the cache reflects the printed numbers.

## 4. Comparison Fairness

- All five algorithms see the **same** `valuations` array per batch (line 432, drawn once, then used for each algorithm's chosen arm at line 437). Same horizon, same customer noise sequence per seed. This is fair.
- The base RNG `rng = RandomState(seed)` drives customer draws; each algorithm has its own RNG with seed offsets (line 397–413). Action selection uses the algorithm's own RNG. This is the correct CRN setup.
- All algorithms have the same `T = 2500` budget, same `BATCH_SIZE = 10` update cadence. No method gets more training.
- One asymmetry already noted: TS has a uniform Beta(1,1) prior while GP methods have an informative downward-sloping mean prior `1 - p`. Strictly speaking this is the algorithm spec the authors chose, not an evaluation flaw, but it is a non-level prior. If TS were given the matching shape-informed prior (e.g. Beta with higher α at low prices) it might close part of the B(2,9) gap. Not flagged as a bug; flagged as a confound in the narrative claim "curve-level information sharing drives the gap".

## 5. Theoretical Sanity Checks

- For B(2,9), `Π*_P = 0.2 · P(v ≥ 0.2 | Beta(2,9)) ≈ 0.2 · 0.86 ≈ 0.172`, so `T · Π*_P ≈ 430`. TS realized cumulative profit ≈ 0.836 · 430 ≈ 360. Regret ≈ 70. With K=10, T=2500, `O(√(KT log K))` ≈ `√(10·2500·log10) ≈ 240` upper bound, so empirical regret well below the upper bound. Consistent with TS being O(√T).
- For B(9,2), the optimal price is high and TS already reaches 98.2%. Monotone GP slightly worse (95.8%–96.1%). This is because the informative GP prior `μ = 1 - p` is most wrong here (true demand at p=0.7 is small but the prior expects ~0.3), so the GP wastes early information overcoming its prior. The monotone constraint also bites: when the true optimum is at a high price, the derivative constraint plus the integrated-trapezoid reconstruction has more room to be wrong than independent-arm TS does. Theoretically defensible.
- No formal regret-rate test in the script. There is no slope fit on `R_t vs log t` or `R_t vs √t` for any algorithm. The chapter's `regret_rates.png` (cited in tex Table tab:regret_comparison) is a *separate* diagram-only sim, not a fit to this script's output. A reviewer asking "do you confirm √T scaling empirically here?" has no chart to point to. This is a missing check.
- Performance differences across scenarios match Weaver2025's qualitative claim: curve-learning matters most when the optimal price is in a low-information region (low p*, where most price-arms are far from the optimum and independent-arm TS pays exploration cost on every arm). Cross-checked against the tex paragraph on line 176 — interpretation is consistent.

## 6. Information Leakage

- Algorithms call `select_arm(rng)` with their own RNG and **no access** to `valuations`, the demand-model object, or the true Beta parameters. `update_batch(arm, sales, n_customers)` receives only the arm chosen and the realized sales count. Clean.
- The GP prior mean `μ_0(p) = 1 - p` is *prior knowledge of shape*, not knowledge of the specific Beta parameters. Acceptable as a structural prior.
- The "oracle" `true_opt_profit` is used only for reporting denominators in tables/figures, never inside any algorithm's selection or update. Confirmed by inspection — the `BetaWTPDemand` instance is created inside `run_one` and only its `draw_wtp` and `profit` methods are exposed; algorithms never see it.

## 7. Seed & Reproducibility

- `N_SEEDS = 1000` (well above the minimum 10).
- Per-scenario `mp.Pool` over seeds with `chunksize=4`. Each seed gets a deterministic `RandomState(seed)` for customer draws and offset seeds per algorithm.
- Means and standard errors stored (`mean_profit`, `se_profit`, line 521–523). Figure shows `±2 SE` bands. Table reports means only.
- Cache config does NOT include kernel hyperparameters or β. If a reader edits the GP lengthscale and re-runs, the cache silently serves stale results. (Same flag as §3.) The author can mitigate by adding hyperparameters to `CONFIG`.
- Reproducible up to NumPy version and `mp.Pool` chunking order; profit aggregates are commutative across seeds so chunking order doesn't matter. Acceptable.

## Hostile-Reviewer Summary

What survives review cleanly: environment matches Weaver2025, fair CRN across algorithms, 1000 seeds, no information leakage, sensible regret magnitudes, table numbers consistent with stdout and tex prose.

What Reviewer 2 will write about:
1. **GP-UCB uses a fixed `β = 1.8` instead of the Srinivas2010 schedule.** A standard reviewer would ask whether the "GP-UCB beats TS by curve learning" claim depends on β. Easy fix: cite this as "GP-UCB with constant exploration scale" in the tex footnote and rerun with the theoretical schedule to confirm robustness.
2. **GP prior mean `μ_0(p) = 1 - p` is shape-informative and TS has a uniform prior.** The "information sharing across prices" narrative is partially confounded with "GP gets a downward-sloping prior". Honest tex framing would say "GP methods exploit both curve smoothness *and* a shape-informed prior."
3. **CONFIG does not capture GP hyperparameters.** Cache could silently serve stale results after a hyperparameter edit. Cosmetic but real.
4. **No empirical regret-rate fit.** The chapter elsewhere asserts O(√T) and O(log T) rates; this sim does not verify them. A panel of `(t, regret_t)` on log-log axes with fitted slopes would close the loop.
5. **Dead `PricingUCB` class** (line 131) — listed in `ALG_LABELS` and `ALG_COLORS` (lines 70, 79) but never instantiated. Just clutter; remove or run it.
6. **File named `*_regret.png` but plot is "% of price-set oracle"** — naming inconsistency, the figure is a profit-ratio plot, not a regret plot. Caption is accurate so the substance survives, but the filename is misleading.

None of these go to the substance of the result. The Weaver claim (curve-level sharing helps most when p* is low) survives. The tex paragraph on line 176 is appropriately hedged.

**Bullshit score: 25%** — Reviewer 2 catches the fixed-β GP-UCB and the shape-informed GP prior (a real confound in the "information sharing" narrative), plus the cache-config gap and the `*_regret.png` naming. The substance survives revision with a one-paragraph footnote and a clean rerun.
