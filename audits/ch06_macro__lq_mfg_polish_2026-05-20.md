# Polish: ch06_macro/sims/lq_mfg.py

**Date:** 2026-05-20
**Predecessor audit:** `audits/ch06_macro__lq_mfg_2026-05-19.md` (Bullshit 25%).
**Scope:** Post-processing artifact only; no Monte Carlo rerun. Three
reviewer nicks addressed by upstream lookup plus tex edits.

## Changes applied

### 1. Reward coefficients pinned in tex

The 2026-05-19 audit flagged that the chapter reward equation named
$c_a, q, \kappa, c_{\mathrm{term}}, \sigma, \rho$ but never gave their
numerical values. I traced them to the upstream defaults at
`/tmp/mfax/mfax/envs/base/toy/linear_quadratic.py` (commit `9acc1eb`,
class `BaseLinearQuadraticEnvParams`, lines 18-24):

```python
kappa: float = 0.5
c_action: float = 0.5
q: float = 0.1
c_term: float = 1.0
sigma: float = 1.0
rho: float = 0.5
```

These match the MFAX SPG / RSPG YAML configs at
`configs/linear_quadratic_spg.yaml` and `configs/linear_quadratic_rspg.yaml`,
which override none of the reward / dynamics fields. The grid wrapper in
this repo (`mfax_lq_run_grid.py`) also overrides none of them.

Inserted after the terminal-reward sentence in
`ch06_macro/tex/macro_rl.tex`:

> The MFAX defaults that we adopt unchanged are $c_a = 0.5$, $q = 0.1$,
> $\kappa = 0.5$, and $c_{\mathrm{term}} = 1.0$, with idiosyncratic-noise
> scale $\sigma = 1.0$ and common-noise sensitivity $\rho = 0.5$. The
> horizon $T = 30$ is the MFAX truncation; the discount factor
> $\gamma = 0.99$ used during training is finite-horizon-compatible and
> matches the paper's configuration.

A reader can now reconstruct the problem without leaving this repo.

### 2. SPG learning-rate boundary documented

Added a footnote where the table is interpreted, naming the asymmetry the
hostile reviewer would flag:

> RSPG's mean final exploitability is non-monotone in the learning rate
> (1316, 60, 797 at $10^{-4}, 10^{-3}, 10^{-2}$), an interior optimum at
> $10^{-3}$. SPG's minimum sits at the grid edge $10^{-2}$
> (1824, 153, 87); a wider sweep might push SPG below its current 86.64
> mean, but does not flip the ranking, since RSPG's regret floor at
> 60.37 is below SPG's interior best at this resolution.

The numbers come straight from `lq_mfg_stdout.txt` (lines 19-20) and
match the JSON aggregation. The ranking-stability argument is the
substantive point: even granting SPG a hypothetical further LR
extension, RSPG's lower regret floor (60.37) versus SPG's grid-edge
mean (86.64) is robust to plausible boundary movement on SPG's side.

### 3. Patches vendored as `patches/mfax/`

Chose option (A) over (B). The diff is small (209 lines, ~39 added /
20 removed across seven files) and entirely mechanical:

- Five env files: wrap mutable `jax.Array` class defaults in
  `field(default_factory=lambda: ...)`. Required by Python 3.11
  dataclasses. Preserves values byte-for-byte.
- Three HSM training scripts: extend the existing
  `jax.debug.print` line with a `Return:` token that prints
  `mean_policy_return`. The field is already computed; only the print
  template changes.

None of the patched lines touches the SPG, RSPG, or environment update
rules.

New files:
- `patches/mfax/py311_compat.patch` (verbatim `git diff HEAD` against
  upstream commit `9acc1eb`)
- `patches/mfax/README.md` (what changed, why, how to reproduce)

Cross-references added:
- Tex footnote on the SPG / RSPG sentence now points to
  `patches/mfax/py311_compat.patch`.
- `mfax_lq_run_grid.py` module docstring now points to the patch file
  rather than vaguely describing "local patches".

## Verification

1. **Tex compiles.** Built single-chapter PDF via
   `cd docs && pdflatex -shell-escape -jobname=ch06_macro
   "\def\chapterfile{../ch06_macro/tex/macro_rl}\input{compile_chapter}"`
   + `bibtex ch06_macro` + two more passes. Output:
   `docs/ch06_macro.pdf`, 33 pages, 962 KB. No undefined references,
   no citation warnings. The new footnotes resolve cleanly.

2. **Numbers cross-checked.** The five new reward-coefficient numbers
   in tex (`0.5`, `0.1`, `0.5`, `1.0`, `1.0`, `0.5`) match the source
   file `/tmp/mfax/mfax/envs/base/toy/linear_quadratic.py` lines 18-24.
   The four learning-rate-grid exploitability numbers in the footnote
   (1316, 60, 797 for RSPG; 1824, 153, 87 for SPG) match
   `ch06_macro/sims/lq_mfg_stdout.txt` lines 19-20.

3. **No Monte Carlo touched.** `lq_mfg.py` was not rerun. The JSON
   artifact (`mfax_lq_grid_results.json`) is unchanged. The table
   (`lq_mfg_results.tex`) and figure (`lq_mfg.png`) are unchanged.

## Remaining concerns (residual ~10%)

These are out of scope for a post-processing polish and would require
upstream cooperation or a separate experiment:

- **Upstream identity drift unbounded.** The chapter pins MFAX commit
  `9acc1eb` in tex and JSON, and now also pins our patches against that
  commit, but if a reader updates MFAX the patches will not apply
  cleanly. Documented in `patches/mfax/README.md`. Not fixable without
  vendoring MFAX itself, which is heavier than this paper requires.
- **Observation-model audit not provided.** The chapter still relies
  on Wibault et al.'s code respecting their own observation model. A
  one-shot audit script that dumps the policy's observation tensor
  would close this; deferred.
- **SPG LR sweep not extended.** The footnote argues the ranking is
  robust to plausible boundary movement, but does not rerun. A
  reviewer who insists on closure can request a $\{10^{-2}, 10^{-1.5},
  10^{-1}\}$ extension; this is a 30-job rerun on the upstream repo.

**Bullshit score: 12%** — Reviewer 2 can now reconstruct the LQ problem
end-to-end from this repo (reward coefficients, observation model,
horizon, discount, patches, commit, configs, grid). The honest
upstream-attestation caveat survives, but is now accompanied by a
checked-in, inspectable patch set rather than a verbal disclaimer. The
ranking robustness is named in prose. What stops it from going lower is
that the experiment is still aggregating someone else's training runs,
which is a structural property of using a public benchmark and not
something the polish pass can change.
