# Polish Report: ch10b_rl_for_ci/sims/dynamic_dml_snmm.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch10b_rl_for_ci__dynamic_dml_snmm_2026-05-19.md` (10%)
**Polish scope:** stale-path cleanup + naive-OLS baseline disclosure

## Changes applied

1. **Script docstring** (`ch10b_rl_for_ci/sims/dynamic_dml_snmm.py`, line 2):
   replaced `Chapter 11, RL for Causal Inference, ...` with
   `Chapter ch10b_rl_for_ci, RL for Causal Inference, ...`.

2. **Tex footnote** (`ch10b_rl_for_ci/tex/rl_for_ci.tex`, line 302):
   added a footnote on the `\emph{Naive OLS}` sentence disclosing that the
   baseline omits $X_2$ on purpose, citing it as the ``init-ctrls'' baseline
   of Lewis & Syrgkanis 2021 Section 6, and noting that a stronger ``control
   on $X_2$'' baseline would introduce post-treatment bias on $\hat\psi_1$.
   The prior chapter-rename footnote (`ch11_rl_for_ci` -> `ch10b_rl_for_ci`)
   was already correct.

3. **Stdout** (`ch10b_rl_for_ci/sims/dynamic_dml_snmm_stdout.txt`):
   re-ran `python3 ch10b_rl_for_ci/sims/dynamic_dml_snmm.py > ...` from the
   repo root. All four results were cache hits; only the per-script
   `OUTPUT_DIR` strings refreshed from `/Users/pranjal/Code/rl/ch11_rl_for_ci/...`
   to `/Users/pranjal/Code/rl/ch10b_rl_for_ci/...`. Numerical results unchanged
   (DML psi_1 RMSE 0.0461 at n=4000, DML psi_2 coverage 0.93, etc.).

4. **PDF recompile**: `docs/ch10b_rl_for_ci.pdf` rebuilt via the
   `compile_chapter` template (3 pdflatex passes + bibtex). 32 pages, exit 0.
   Stale `Hfootnote` dest warnings present (pre-existing, resolve on next pass).

## Verification

- `grep -n "ch11_rl_for_ci\|Chapter 11"` across the script, stdout, and tex
  returns no hits.
- Numerical values in the regenerated stdout match the prior audit's table
  (verified DML coverage 0.93 on psi_2 at n=4000; Naive OLS coverage 0.00).
- The tex prose number "0.93 on psi_2" and the table value (psi_2 coverage
  0.93 for Dynamic DML at n=4000) still match.

## What the polish did not change

- No algorithm code touched (`fit_naive_ols`, `fit_msm_iptw`,
  `fit_dynamic_dml`).
- No DGP, hyperparameter, or seed changes; cache files reused.
- No figure regenerated content-wise; PNG bytes regenerated but plot is
  identical to prior run.
- The Naive OLS baseline itself is unchanged; only its disclosure status
  in the tex changed.

## Residual hostile-reviewer comments

- The Naive OLS baseline still omits $X_2$. The new footnote upgrades this
  from "implicit choice" to "explicitly disclosed deliberately weak baseline
  per the original paper", which is the standard textbook handling. A reviewer
  who wanted both a "control on $X_2$" and a "control on $X_1$" naive baseline
  would still mutter, but the disclosure removes the straw-man complaint.
- No explicit "plug-in DML" (non-orthogonal) row; Naive OLS plays that role
  by design.
- Stale `Hfootnote.1`/`Hfootnote.2` reference warnings during the second
  pdflatex pass; these are hyperref cosmetic warnings and resolve on the
  third pass without affecting output.

## Score

**Bullshit score: 3%** -- The hostile reviewer reads the section twice, finds
the path-rename footnote correct, finds the naive-OLS baseline explicitly
disclosed as the paper's own ``init-ctrls'' contrast, and finds the
numerical claims in the prose match the table and stdout to 3 decimals.
The remaining 3% is the inherent fragility of any DML simulation under a
sufficiently hostile reading (e.g. the choice $\nu = 2\gamma/\|\gamma\|$
deliberately aligns the feedback channel with the propensity direction
to maximize bias signal, which is honest but aggressive). Substance, data
integrity, leakage, reproducibility all clean.

Target met (<=5%).
