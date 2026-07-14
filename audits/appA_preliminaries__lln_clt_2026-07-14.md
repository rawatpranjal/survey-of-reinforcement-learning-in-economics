# Simulation Audit: appA_preliminaries / lln_clt

- **Sim:** `appA_preliminaries/sims/lln_clt.py` (LLN + CLT for the sample mean)
- **Date:** 2026-07-14
- **Type:** FULL, condensed variant (small pedagogical appendix sim; never audited before)
- **Auditor stance:** hostile journal referee, read-only, evidence only.

**Files read:**
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lln_clt.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lln_clt_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lln_clt.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lln_clt.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 206-251)

---

## Step 3 statement

**(i) Mathematical result presented.** Theorem `thm:prelim_lln_clt` (attributed to Durrett), stated at `preliminaries.tex:210-223`, gives the two classical large-sample results for i.i.d. draws with mean $\mu$ and finite variance $\sigma^2$. The strong law of large numbers, $\bar X_n \to \mu$ almost surely (Eq. `prelim_lln`), and the central limit theorem, $\sqrt{n}(\bar X_n - \mu) \Rightarrow N(0,\sigma^2)$ (Eq. `prelim_clt`), holding whatever the shape of the underlying distribution. A proof sketch follows (Chebyshev for the weak law, characteristic-function Taylor expansion for the CLT).

**(ii) What the sim/figure is evidence FOR.** The figure and table are the empirical illustration of that theorem. Panel A of `lln_clt.png` shows 25 running-mean trajectories of $\mathrm{Exp}(1)$ draws settling on $\mu=1$ (pathwise convergence, i.e. the *strong* law the footnote at `preliminaries.tex:231` promises to show). Panel B shows the histogram of the rescaled fluctuation $\sqrt{n}(\bar X_n-\mu)$ at $n=1000$ matching $N(0,1)$. Table `tab:prelim_lln_clt` is evidence that the CLT holds for three *non-normal* sources, checking that the variance of the rescaled mean matches $\sigma^2$ and that skewness / excess kurtosis approach the normal value of zero. So the sim is decoration-plus-check for a textbook theorem, not a novel result.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS

The code computes exactly the objects the theorem is about.

- LLN: `traj[i] = np.cumsum(x)/np.arange(1,max_n+1)` (`lln_clt.py:118`) is the running sample mean $\bar X_n$, plotted in Panel A. Trajectories visibly collapse onto $\mu=1$ by $n\approx10^3$ and stay there to $10^5$; stdout line 15 reports the final-$n$ spread across 25 seeds as $[0.9941, 1.0066]$, consistent with $\bar X_n \to \mu$.
- CLT: `s = np.sqrt(n_clt)*(xbar - mu)` (`lln_clt.py:159`) is precisely the rescaled fluctuation $\sqrt{n}(\bar X_n-\mu)$. The target law $N(0,\sigma^2)$ uses the analytical `sd` from `dist_params`, not a fitted value, so the overlay/KS are genuine goodness-of-fit checks, not a self-fit.
- Numerics agree with theory. The variance of $\sqrt n(\bar X_n-\mu)$ is exactly $\sigma^2$ for any $n$ (since $\mathrm{Var}(\sqrt n\bar X_n)=\sigma^2$); empirical ratios are 0.9972 / 1.0085 / 0.9970 (stdout 17-19), all within Monte-Carlo noise of 1. Skewness matches the closed form $\mathrm{skew}(X)/\sqrt n$: predicted 0.0632 / 0.0276 / 0 vs empirical 0.0589 / 0.0235 / 0.0108 (verified by hand). The LLN mean-abs-deviation $E|\bar X_n-\mu|$ tracks its CLT prediction $\sigma\sqrt{2/\pi}/\sqrt n$ to three digits: 0.2523 vs 0.2523 at $n=10$, 0.00461 vs 0.00461 at $n=30000$ (recomputed independently, matches stdout 16). No method beats its own oracle; nothing contradicts the theorem's rate or fixed point.

### (b) PRESENTATION / NUMBERS — PASS (one cosmetic defect)

- Table `lln_clt.tex:9-11` reproduces stdout 17-19 digit-for-digit ($\sigma^2$, var ratio, skew, excess kurt, $D_{\mathrm{KS}}$). Figure caption numbers (25 seeds, $n=1000$, 40,000 replicates) match `CONFIG` (`lln_clt.py:38,44,45`) and the table caption. Axes and legends are correct: Panel A $x$=sample size (log), $y$=running mean $\bar X_n$, dashed $\mu$; Panel B $x=\sqrt n(\bar X_n-\mu)$, $y$=density, dashed $N(0,\sigma^2=1.00)$. The `40{,}000` LaTeX thousands-separator renders correctly.
- **Cosmetic defect:** `lln_clt_stdout.txt:20-22` reports save paths under `/Users/pranjal/Code/rl-theory-proofs/...`, a worktree that no longer exists (`git worktree list` shows only the primary checkout; the directory is absent). The stdout was produced in a since-removed sibling checkout. Because `CONFIG` and all seeds are fixed and identical, the numbers are reproducible regardless of path, so this is provenance noise, not a numeric mismatch. All four artifacts share mtime 2026-07-13, i.e. regenerated together.

### (c) CHAPTER FIT — PASS

The figure + caption alone teach the result to a cold reader. Panel A makes "sample mean settles on the mean" visually obvious; Panel B makes "rescaled wobble is Gaussian" obvious; the caption names both objects and the parameters. The table extends the CLT claim to non-normal sources, directly supporting the prose sentence at `preliminaries.tex:241` ("the Gaussian shape appears whatever the individual draws look like"). The prose-to-artifact wiring is correct: `\ref{fig:prelim_lln_clt}` and `\ref{tab:prelim_lln_clt}` resolve to the included PNG and `\input` table.

### (d) EFFICIENCY / STANDARDS — PASS (minor)

- Seeds fixed and deterministic: `seed_base=20250`, per-distribution offsets `20250+100+17*di` with an explicit comment (`lln_clt.py:156-157`) rejecting `hash()`. Replicate counts exceed the repo minimum everywhere stochastic: 40,000 CLT replicates, 2,000 per rate-grid point, 25 LLN trajectories.
- Flags conform: `add_component_args` supplies `--data-only`/`--plots-only`/force; `main()` routes all three (`lln_clt.py:321-328`). Caching via `compute_or_load` on a single `'lln_clt'` component keyed to `CONFIG`.
- Stdout is factual: headers, then one line per configuration, no opinion words. Compliant with the stdout format rules.
- Minor: the table reports point statistics with no standard errors, and `ks_by_n` (the rate-grid KS distances) is computed and cached but never printed, plotted, or tabulated (see Findings 2-3).

---

## 7-point checklist

1. **Algorithm identity.** PASS. No "algorithm" per se; the estimands are the running mean and $\sqrt n(\bar X_n-\mu)$, coded exactly (`lln_clt.py:118,159`). KS uses `scipy.stats.kstest` against the analytical CDF.
2. **Environment / MDP fidelity.** N/A — no MDP. Base distributions and their analytical $(\mu,\sigma^2)$ (`dist_params`, `lln_clt.py:69-76`) are correct: Exp(1) $(1,1)$, Unif(0,1) $(1/2,1/12)$, Bern(0.3) $(0.3,0.21)$.
3. **Data integrity.** PASS. `compute_data` runs `_run_experiment`; reported numbers trace to computed variables (`emp_var`, `var_ratio`, `skew`, `exkurt`, `ks`), not hardcoded. Table digits equal stdout digits.
4. **Comparison fairness.** PASS. Same $n=1000$ and same 40,000-replicate protocol for all three distributions; the only per-distribution difference is a distinct fixed seed, which is correct.
5. **Theoretical sanity.** PASS. Variance ratios $\to 1$, skew $\to \mathrm{skew}(X)/\sqrt n$, LLN deviation matches $\sigma\sqrt{2/\pi}/\sqrt n$; Bernoulli's larger KS (0.0223 vs ~0.0046) is the expected lattice/discreteness effect, consistent with theory.
6. **Information leakage.** PASS. The overlaid Gaussian uses the analytical $\sigma$ (known constant, not estimated from the sample being tested), which is the legitimate way to state the theorem's target; no forbidden peeking.
7. **Seed / reproducibility.** PASS. Seeds set at top, deterministic offsets, replicate counts well above 10. (No SEs on the summary statistics; see below.)

---

## Findings (severity-ordered)

1. **(low, presentation) Stdout was generated in a removed sibling worktree.** `lln_clt_stdout.txt:20-22` prints save paths under `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/...`, which `git worktree list` and a directory check show no longer exists. Fixed seeds + identical `CONFIG` make the reported numbers reproducible in the primary checkout, so no value is wrong; the path string is stale provenance. Re-running from `/Users/pranjal/Code/rl` would refresh the header to the canonical path.

2. **(low, standards) `ks_by_n` is computed, cached, and never surfaced.** `_run_experiment` builds the rate-grid KS distances `ks_by_n` (`lln_clt.py:137,143`) and returns them (`:186`), and the header comment (`:6-7`) advertises "KS distance falling as $1/\sqrt n$" as a check, but no figure, table, or stdout line displays it. Dead output relative to the advertised claim. Either print/plot it or drop it and the comment clause.

3. **(low, standards) Table carries no uncertainty.** `lln_clt.tex` reports var-ratio / skew / excess-kurt / $D_{\mathrm{KS}}$ as bare point estimates. The repo study-design rule asks for standard errors on reported statistics. Over 40,000 replicates the estimates are precise, so this is defensible for a pedagogical appendix, but a strict referee would ask for the Monte-Carlo error (or a note that it is negligible).

4. **(very low, prose) "Bernoulli approaches most slowly" is true only by KS.** `preliminaries.tex:241` places this sentence right after mentioning skewness and excess kurtosis, yet Bernoulli's empirical skew (0.0235) is *smaller* than the exponential's (0.0589). The claim holds on the headline metric $D_{\mathrm{KS}}$ (0.0223, ~5x the others, the lattice effect), so the substance is correct, but the juxtaposition invites a Reviewer-2 nitpick. Anchoring the sentence explicitly to $D_{\mathrm{KS}}$ would close it.

None of the four touches the mathematical substance, which is a correctly-computed, theory-matching illustration of the LLN and CLT.

**Note on the diagram-only cap:** the script genuinely computes iterates and rates (running means, KS statistics, variance ratios, and an $E|\bar X_n-\mu|$-vs-$\sqrt n$ comparison), so the 25% diagram-only cap does not apply. The score below stands on its own merits and lands well under that cap regardless.

**Bullshit score: 15%** — Reviewer 2 flags the unused KS-vs-$n$ computation, the missing standard errors, and the removed-worktree path in stdout, but the LLN/CLT estimands are coded exactly and every number reproduces the theorem's rates and fixed points.
