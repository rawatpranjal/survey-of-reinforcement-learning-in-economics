# Simulation Audit — Jensen Gap (Appendix A, Mathematical Preliminaries)

**Sim:** `appA_preliminaries/sims/jensen_gap.py`
**Date:** 2026-07-14
**Type:** FULL, condensed variant (small pedagogical appendix sim; never audited before)
**Auditor role:** hostile journal referee, evidence-only, read-only.

**Files read:**
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/jensen_gap.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/jensen_gap_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/jensen_gap.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/jensen_gap.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 166-204, the consuming subsection)
- `/Users/pranjal/Code/rl/CLAUDE.md` (rubric)

---

## Step-3 statement (what is presented, what the sim is evidence for)

**(i) The mathematical result.** Theorem `thm:prelim_jensen` (preliminaries.tex:175-183) is Jensen's inequality: for convex `φ: ℝ→ℝ` and integrable `X`, `φ(E[X]) ≤ E[φ(X)]`, with the direction reversing for concave `φ` and equality for affine `φ`. Equation `eq:prelim_jensen` is `φ(E[X]) ≤ E[φ(X)]`. The proof (lines 185-193) is the standard supporting-line argument. The surrounding prose motivates it as the source of Q-learning maximization bias and of entropic-risk / distributionally-robust value operators.

**(ii) What the sim is evidence for.** The figure and table are cited (preliminaries.tex:195) to show two things about the Jensen gap `E[φ(X)] − φ(E[X])` for `X ~ N(0, σ²)`: that the plug-in Monte Carlo estimate (a) matches the closed-form gap (`σ²` for `x²`, `e^{σ²/2}−1` for `e^x`), and (b) never turns negative, as convexity requires. The stdout additionally shows a concave contrast (`√x` on `U(0.5,1.5)`) with a negative gap, illustrating the reversal clause of the theorem.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS
- **Theorem identity.** The code computes `phi(x).mean() - phi(x.mean())` (jensen_gap.py:111), the exact empirical analog of `E[φ(X)] − φ(E[X])`, the quantity in `eq:prelim_jensen`. The object matches the theorem.
- **Closed forms are correct.** `analytical_gap` returns `σ²` for square and `e^{μ+σ²/2} − e^μ` for exp (jensen_gap.py:70-75); at `μ=0` the latter is `e^{σ²/2}−1`. Independently recomputed: exp gaps 0.1331 / 0.6487 / 2.0802 / 6.3891 for σ=0.5/1.0/1.5/2.0, matching stdout lines 18-21 exactly. Concave `E[√X]−√(E[X]) = −0.011` for `U(0.5,1.5)` recomputed and matches stdout line 23.
- **Numerics consistent with the guarantee.** `P(gap≥0)=1.000` for every convex case (stdout 14-21). This is in fact a deterministic identity, not a lucky Monte Carlo outcome: for any finite sample the empirical measure obeys Jensen, so `mean(φ(x)) ≥ φ(mean(x))` always holds. The estimator converges to the closed form as `N` grows (figure, both panels). No method beats or violates the theoretical object.
- **Heavy-tail behavior is real, not a bug.** The `e^x`, σ≥1.5 rows sit below the exact gap at N=1e5 (2.32 SE low for σ=1.5, 3.38 SE low for σ=2.0; recomputed). This is the expected slow, right-skewed convergence of a lognormal-mean estimator, and it is disclosed rather than hidden (see (b)).

### (b) PRESENTATION / NUMBERS — PASS
- Every number in the `.tex` table traces to stdout: analytical, MC±SE, and `Pr(gap≥0)` columns (jensen_gap.tex:9-17) match stdout lines 14-21 digit-for-digit.
- Figure ↔ table ↔ stdout mutually consistent: dashed asymptotes in the PNG sit at 0.25/1/2.25/4 (left) and 0.13/0.65/2.08/6.39 (right), matching the analytical column; solid MC curves terminate at the tabulated N=1e5 values (e.g. right-panel red ends just below its dashed line at 6.31, table 6.3147).
- Axes/legends/units correct: x-axis `Sample size N` (log), y-axis `\widehat{E[φ(X)]} − φ(X̄)`, legend title `solid: MC, dashed: exact`, per-σ colors from `plot_style` (jensen_gap.py:40-45), zero reference line present in both panels.
- The `.tex` caption is unusually honest: it explicitly states the σ≥1.5 exponential rows "converge more slowly and still sit a few standard errors low at this N" (jensen_gap.tex:3), which the numbers confirm. This is a strength, not a defect.
- The figure caption's closed forms `σ²` and `e^{σ²/2}−1` (preliminaries.tex:200) are correct for μ=0.

### (c) CHAPTER FIT — PASS
The figure + caption alone teach the result to a cold reader: the plotted gap is labeled, the two convex functions are named, the closed-form dashed lines are identified, and the zero floor "the inequality forbids the gap to cross" is drawn and explained. A reader with no code access sees positivity and convergence-to-exact. The two-point `0/2` warm-up example (preliminaries.tex:173) and the proof precede the figure, so the illustration lands in context.

### (d) EFFICIENCY / STANDARDS — PASS
- Stochastic where it should be: 40 seeds via `RandomState(1000+si)` (jensen_gap.py:103), well above the ≥10 minimum; mean and SE reported.
- Flags conform to the multi-component convention: `add_component_args` / `compute_or_load` / `parse_force_set` give `--data-only`, `--plots-only`, and force refresh (jensen_gap.py:145-154, 251-274). `--plots-only` recomputes-from-cache then draws, per spec.
- Caching keyed on `CONFIG` (jensen_gap.py:29-36, 147-154); stdout shows a genuine `Computing: jensen` first pass and `Cache hit: jensen` second pass (stdout 9, 37).
- Colors/figure sizes from the central palette (`COLORS`, `FIG_DOUBLE`); no hardcoded hex.
- Stdout format: header, parameters, one factual line per configuration, summary; no opinion words. Compliant.
- Nested-prefix sampling (draw `max_n` once, read prefixes, jensen_gap.py:106-108) makes each curve a genuine within-seed refinement as N grows, which is the right design for a convergence plot.

---

## 7-point checklist

1. **Algorithm identity** — PASS. The "algorithm" is the plug-in Jensen-gap estimator; jensen_gap.py:111 is term-for-term the empirical `E[φ(X)]−φ(E[X])`. No placeholder, no missing term.
2. **Environment/MDP fidelity** — N/A. No MDP; the "environment" is `X ~ N(0,σ²)` and `U(0.5,1.5)`, which match the tex (`X ~ N(0,σ²)`, preliminaries.tex:200) exactly.
3. **Data integrity** — PASS. `compute_data` runs real Monte Carlo (`_run_experiment`); reported table/figure numbers are the computed variables, not hardcoded. Cache config-keyed.
4. **Comparison fairness** — PASS. MC vs closed form uses the same `X` draws for every N (nested prefixes); no method advantaged.
5. **Theoretical sanity** — PASS. Convex gap ≥ 0 and converges to the exact closed form; concave gap < 0; heavy-tail slow convergence appears where theory predicts it.
6. **No information leakage** — PASS. The estimator uses only the sample mean, never the true `μ` (comment jensen_gap.py:109-110); verified in code.
7. **Seed / reproducibility** — PASS. Seeds fixed, 40 runs, mean ± SE reported. The concave contrast uses a single seed at one N (jensen_gap.py:134-136), but it is an stdout-only illustration, not a headline result.

---

## Findings, severity-ordered

1. **(Low) `Pr(gap ≥ 0)` column is a deterministic identity presented as an empirical frequency.** For any finite sample and convex `φ`, `mean(φ(x)) ≥ φ(mean(x))` holds by Jensen on the empirical measure, so this column can only ever read `1.000`; it is a code sanity check, not Monte Carlo evidence. The prose "Table confirms it ... never turns negative" (preliminaries.tex:195) is true but reads as if positivity were observed rather than guaranteed. Evidence: jensen_gap.py:116, jensen_gap.tex:9-17 all `1.000`. Harmless; a hostile reviewer would note the column carries no empirical information.

2. **(Low) "small-σ exponential cases agree within a standard error" is marginally overstated for σ=1.0.** The exp σ=1.0 row is 1.29 SE from the closed form (0.6478 vs 0.6487, SE 0.0007), just over one SE, not within it. Evidence: jensen_gap.tex:3 caption vs jensen_gap.tex:15. Cosmetic; the qualitative split (small σ close, σ≥1.5 slow) is correct.

3. **(Info, not a defect) Outputs were generated in a different working directory.** The stdout save paths point to `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/sims/...` (stdout 24-26), a worktree that no longer exists, while the audited artifacts live under `/Users/pranjal/Code/rl/...`. The committed `.png`/`.tex` in this repo are self-consistent with the stdout numbers, so there is no data-integrity problem; the embedded absolute paths are simply stale provenance. Consistent with the repo's mandatory-worktree workflow.

4. **(Info) y-axis hat spans the whole expectation.** `\widehat{E[\varphi(X)]}` (jensen_gap.py:200) places the empirical hat over `E[φ(X)]`; unambiguous given the caption, slightly non-standard.

**Diagram-only cap:** does NOT apply. The script computes real Monte Carlo iterates across nine sample sizes and 40 seeds and compares them to closed-form fixed values, so it is a genuine numerical experiment, not a diagram.

---

**Bullshit score: 12%** — Reviewer 2 catches that the `Pr(gap≥0)=1.000` column is a built-in identity dressed as empirical evidence and that "within a standard error" is loose for the σ=1.0 exp row, but the object matches the theorem, every number traces to the run, and the slow-convergence caveat is disclosed rather than buried.
