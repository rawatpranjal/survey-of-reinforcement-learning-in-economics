# Fix Report: ch05_econ_models/sims/nfxp_ccp_td.py

**Date:** 2026-05-19
**Original score:** 50%
**New estimated score:** 20-25% — Reviewer 2 still has angles of attack (the locally robust PMLE correction is disclosed-as-omitted rather than implemented, and the θ-linear decomposition is still a reformulation), but the script now runs end-to-end, the bib entry is correct, seeds are 10 with SEs reported and PyTorch deterministic, and the framing-vs-implementation gap is acknowledged in a tex footnote. Substance unchanged; audit trail repaired.

## Files modified
- `ch05_econ_models/sims/nfxp_ccp_td.py`
  - Imports: `compute_or_load, add_component_args, parse_force_set` → `load_results, save_results, add_cache_args` (matches what the body actually calls; script now executes).
  - `CONFIG['version']` 14 → 15; `CONFIG['exp2_seeds']` 5 → 10 (forces cache invalidation and bumps to project-standard sample size).
  - `estimate_td_ccp_nn(...)` now takes `seed=0` and sets `torch.manual_seed(seed)` (plus CUDA seed when present) before constructing the four `HNet`s.
  - `run_single_estimation(...)` and the per-seed call site propagate `seed=seed` to the TD-CCP Neural estimator.
  - `generate_outputs(data)` now prints a per-method, per-K summary table to stdout including RC bias SE, RC RMSE SE, θ₁ RMSE SE, θ₂ RMSE SE (RMSE SE via delta method, `std(sq_err)/(2·RMSE·√n)`).
  - LaTeX table widened to 12 columns: RC Bias / RC RMSE / θ₁ RMSE / θ₂ RMSE each with an adjacent SE column under a `\multicolumn{2}` grouping with `\cmidrule(lr)` separators.
- `ch05_econ_models/tex/rl_in_se.tex` (§sec:ddc_estimation_sim)
  - Added footnote on `\citet{AdusumilliEckardt2022}` disclosing (a) omission of the locally robust PMLE correction (Theorem 5, the √n-consistency result), (b) the θ-linear decomposition reformulation, and (c) that the PMLE step still uses the empirical `P_keep` so the "no-transition-density" advantage is qualified.
  - "5 seeds" → "10 seeds with the PyTorch optimizer also seeded; ... means with seed-level standard errors" in the methods paragraph.
  - Table and figure captions: "Averages over 5 seeds" → "means over 10 seeds".
  - Results paragraph rewritten to remove hardcoded `179s`, `0.077`, `0.337`, `28–44s` numbers (which would drift across re-runs and across the 5-vs-10 seed difference) and replaced with directional statements that the table/figure carry.
  - Cleanup pass 2026-05-19 (this session): regenerated the results table via `--plots-only` after the bg run finished so all 12 columns (now including RC RMSE / SE) are present; updated the table caption from "RC bias, and root mean squared error for each structural parameter" to "RC bias, RC RMSE, and root mean squared error for $\theta_1$ and $\theta_2$" so the caption matches the four-column-group table.
- `docs/refs.bib` `@article{AdusumilliEckardt2022}`
  - Author: `{Adusumilli, S. and Eckardt, M. and Tate, G.}` → `{Adusumilli, Karun and Eckardt, Dita}` (hallucinated third author removed; first names corrected per the actual paper, verified against `ch05_econ_models/papers/AdusumilliEckardt2022_td_learning_ddc.md`).
  - Title: `Estimation of Dynamic Discrete Choice Models with Differentiable Temporal-Difference Learning` → `Temporal-Difference Estimation of Dynamic Discrete Choice Models`.

## Bug fixes (always-applied)
- Fixed `sim_cache` import (script now executes; verified via `python3 ch05_econ_models/sims/nfxp_ccp_td.py --plots-only` no NameError).
- Bumped seeds 5 → 10; added SE columns to both stdout summary and the LaTeX table; seeded PyTorch in TD-CCP Neural so two reruns now yield identical θ̂ for the same panel data.
- Corrected `AdusumilliEckardt2022` bib entry (removed hallucinated co-author "Tate, G."; corrected title; expanded first names from initials).
- NFXP timing inconsistency removed by switching the prose from a hardcoded `179s` to a directional statement; the canonical number is whatever the table reports (now 163.5s mean at K=4 over 10 seeds).

## Relabels / disclosures
- Added footnote on the TD-CCP citation explicitly stating:
  - Locally robust PMLE correction (Adusumilli–Eckardt 2022 Theorem 5) is omitted. The √n-consistency guarantee of the paper does not apply to the plug-in PMLE we report.
  - The implementation reformulates the bootstrap target via a θ-linear decomposition of EV rather than fitting h(a,s) directly as in the paper.
  - The PMLE step evaluates `v_0 = -c + γ·P_keep·EV` using the empirical `P_keep`, so the "no transition density" framing is qualified; a fully density-free variant is left to future work.
- Methods paragraph and captions updated to reflect 10 seeds with SE reporting.

## Re-run verification
- Script exit code: 0 (full background run, single end-to-end pass, no exceptions).
- Cache regenerated at `ch05_econ_models/sims/cache/nfxp_ccp_td.pkl`.
- Stdout regenerated at `ch05_econ_models/sims/nfxp_ccp_td_stdout.txt`; full per-seed trace for all four K values (10 seeds each) plus the new SUMMARY block.
- New stdout key values (10-seed means ± SE):

  | Method | K=1 | K=2 | K=3 | K=4 |
  |---|---|---|---|---|
  | NFXP RC RMSE | 0.097 ± 0.015 | 0.087 ± 0.018 | 0.044 ± 0.009 | 0.061 ± 0.012 |
  | CCP RC RMSE | 0.099 ± 0.015 | 0.120 ± 0.021 | — (sparse) | — (sparse) |
  | TD-CCP Linear RC RMSE | 0.101 ± 0.016 | 0.182 ± 0.021 | 0.248 ± 0.010 | 0.331 ± 0.014 |
  | TD-CCP Neural RC RMSE | 0.094 ± 0.013 | 0.090 ± 0.021 | 0.051 ± 0.009 | 0.066 ± 0.011 |

  K=4 NFXP wall-clock: 163.5s (mean over 10 seeds); TD-CCP Neural: 40.7s; TD-CCP Linear: 0.2s; CCP fails on coverage at K≥3.
- Chapter PDF compiles cleanly: `/Users/pranjal/Code/rl/docs/ch05_econ_models.pdf` (16 pages, 1,299,568 bytes — verified by this session after the table caption tweak). Three-pass build: `pdflatex → bibtex → pdflatex → pdflatex`. Only undefined-reference warning is the expected cross-chapter `section:language` that single-chapter compiles cannot resolve; no undefined citations and no error-level issues.
- Bullshit-detector axis check:
  - §1 Algorithm Identity: now passes by disclosure (footnote relabels TD-CCP variants as the simplified plug-in form; identity-claim no longer overreaches).
  - §3 Data Integrity: passes (script runs end-to-end; cache and stdout regenerated from a single execution; reported numbers traceable to the run).
  - §6 Information leakage / framing: passes by disclosure (P_keep at PMLE step explicitly acknowledged).
  - §7 Seed and reproducibility: passes (10 seeds, SEs printed and tabulated, PyTorch seeded so MLP training is now deterministic given the seed).

## Residual issues
- Locally-robust PMLE correction is disclosed-as-omitted rather than implemented (per the user's mixed-strategy decision: no substantive Adusumilli–Eckardt reimplementation locally). A hostile reviewer who wants the √n-consistency result will still flag this; the chapter no longer claims to provide it.
- The θ-linear decomposition of EV (replacing the paper's direct h(a,s) fit) remains; now disclosed in the footnote rather than silent.
- The PMLE step's use of `P_keep` is disclosed but not removed; building a fully density-free variant (sample-target average instead of `P_keep·EV`) is a future-work item.
- The M=20 bin choice (vs Rust's M=90) is not motivated in tex; soft flag from the audit, not addressed in this fix.
- TD-CCP Linear's growing RC bias with K is still attributed to "basis misspecification at higher K" in prose; the audit's alternative reading (one-shot inversion that doesn't iterate, à la NPL) is not discussed. This is a writing call, not a script bug.
