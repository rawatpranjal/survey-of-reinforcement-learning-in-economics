# Simulation Audit — appA_preliminaries / spectral_radius

- **Date:** 2026-07-14
- **Type:** FULL (condensed variant — small pedagogical appendix sim, never audited before)
- **Subject script:** `/Users/pranjal/Code/rl/appA_preliminaries/sims/spectral_radius.py`
- **Outputs:** `spectral_radius_stdout.txt`, `spectral_radius.tex`, `spectral_radius.png` (same dir)
- **Consuming tex:** `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 11-43: Theorem `thm:prelim_spectral`, Figure `fig:prelim_spectral` at line 38, Table input at line 43)
- **Files read end to end:** the script, the three output files, `preliminaries.tex` §Spectral Radius (lines 1-60), `sims/plot_style.py` (COLORS, FIG_SINGLE). Numbers independently recomputed with a throwaway `python3 -c` (not the repo script).

## Step 3 — what the appendix presents, what the figure is evidence for

**(i) The mathematical result.** Theorem `thm:prelim_spectral` ("Spectral radius governs decay", cited to Horn & Johnson 2013): for a square matrix `A` with spectral radius `ρ(A) = max_i |λ_i|`, the powers vanish `A^k → 0` iff `ρ(A) < 1`, and in that case `‖A^k‖^{1/k} → ρ(A)` (Gelfand's formula). The surrounding prose (line 34) stresses the practical corollary: the norm `‖A‖` can exceed 1, so powers grow transiently, yet they still decay because `ρ(A) < 1`; the decay rate is `ρ`, not the norm.

**(ii) What the sim/figure is evidence for.** For the non-normal Jordan-type matrix `A = [[ρ, 2], [0, ρ]]` (both eigenvalues `ρ`, operator norm `> ρ`), the figure and table demonstrate the norm-vs-radius distinction on three cases `ρ ∈ {0.5, 0.8, 0.95}`: the operator-2-norm of the powers rises to a transient peak (because `‖A‖₂ > 1`) and then decays, and the per-step tail ratio `‖A^{k+1}‖/‖A^k‖` settles at `ρ` from above. It illustrates the "in that case" rate half of the theorem and the norm-is-not-the-rate point, not the "only if" (no `ρ ≥ 1` growth case is shown, nor is one claimed).

## Criteria verdicts

**(a) CORRECTNESS — PASS.** The code computes exactly the theorem's object. `make_matrix` (py:36-39) builds `[[ρ, 2],[0, ρ]]`, whose eigenvalues are both `ρ`, so `ρ(A) = ρ` by construction — the labeled `ρ` in the table genuinely is the spectral radius. `power_norms` (py:42-49) forms `A^k` by iterated multiplication and records the operator 2-norm (largest singular value), the correct object for `‖A^k‖`. The tail ratio (py:66) aligns indices correctly: `norms[-30:]/norms[-31:-1]` gives `‖A^{k+1}‖/‖A^k‖` for `k = 170..199`. Numerics match the theory: for a `2×2` Jordan block `A^k = [[ρ^k, 2k ρ^{k-1}],[0, ρ^k]]`, so `‖A^k‖ ~ 2k ρ^{k-1}`, giving a consecutive ratio `≈ ρ(1 + 1/k)` — i.e. `→ ρ` from above with an `O(1/k)` bias, exactly as the .tex caption states. All three reported tail ratios sit just above `ρ` (0.5027, 0.8043, 0.9552). The transient-peak locations match the analytic maximizer `k* = -1/ln ρ` (0.5→k=1, 0.8→k≈4, 0.95→k≈19). Independent recompute reproduced every table cell to the printed precision.

**(b) PRESENTATION/NUMBERS — PASS.** stdout (lines 9-11), the generated table (`spectral_radius.tex` lines 9-11), and the figure are mutually consistent. Every table number traces to `_run()` (py:52-79) and equals my independent recomputation: `ρ=0.5 → ‖A‖₂=2.118, peak 2.12 @ k=1, ratio 0.5027`; `ρ=0.8 → 2.281, 4.14 @ k=4, 0.8043`; `ρ=0.95 → 2.379, 15.10 @ k=19, 0.9552`. Figure axes are labeled (`Power k`, `‖A^k‖₂`), log-scaled `semilogy`, legend distinguishes solid `‖A^k‖₂` vs dashed `ρ^k` with per-`ρ` colors. The visual asymptotics agree with theory: blue solid reaches `~10^{-57}` at `k=200` vs `ρ^{200}=0.5^{200}≈10^{-60.2}`, i.e. solid parallel to and above its dashed `ρ^k` (offset ≈ `2k`), confirming shared slope. Table caption's `+O(1/k)` claim is quantitatively exact.

**(c) CHAPTER FIT — PASS.** The figure caption (preliminaries.tex line 39) states the matrix, that solid = `‖A^k‖₂`, dashed = `ρ^k`, that solids grow to a transient peak then decay, and that both share the asymptotic slope so `ρ` sets the rate despite `‖A‖₂ > 1`. The table caption restates the object and the from-above convergence. A cold reader gets the point from caption + figure alone. Prose at line 34 ties figure and table to the theorem correctly.

**(d) EFFICIENCY/STANDARDS — PASS.** Computation is fully deterministic (matrix powers, no RNG), so seeds and multi-seed runs are correctly N/A. Script uses the multi-component cache API (`compute_or_load`, `add_component_args`, `parse_force_set`), honors `--data-only`/`--plots-only`, splits `compute_data`/`generate_outputs` with the required boundary (`generate_outputs` never trains, `compute_data` never plots), writes 300-dpi PNG via `FIG_SINGLE`, and pulls colors from `sims/plot_style.COLORS` (no hardcoded hex; `blue/green/red` keys exist, plot_style.py:11-15). stdout is header + params + one factual line per config + output paths, no opinion words.

## 7-point checklist

1. **Algorithm identity** — PASS. Not an RL algorithm; the "method" is `A^k` via iterated matmul and `numpy` operator-2-norm. Both are the objects the theorem names.
2. **Environment/MDP fidelity** — PASS. The "model" is the matrix `A = [[ρ,2],[0,ρ]]`, which matches the figure/table caption exactly (shear `2`, `ρ ∈ {0.5,0.8,0.95}`).
3. **Data integrity** — PASS. `compute_data` runs `_run` (or a deterministic cache reload); table/stdout numbers all reproduced by independent recompute; no hardcoded "expected" values.
4. **Comparison fairness** — PASS. The comparison is `‖A^k‖₂` (solid) vs the reference decay `ρ^k` (dashed) on the same axis, same `k`; fair by construction.
5. **Theoretical sanity** — PASS. Tail ratios converge to `ρ` from above with `O(1/k)` bias; peaks match `k*=-1/ln ρ`; asymptotic figure slopes match `ρ^k`. All consistent with the stated theorem and Gelfand's formula.
6. **Information leakage** — N/A. Deterministic illustration, no learning agent, no train/test split.
7. **Seed/reproducibility** — N/A (deterministic; no stochasticity to seed). Config `version:2` invalidates stale cache on change.

## Findings (severity-ordered)

1. **[Low — provenance]** The committed `spectral_radius_stdout.txt` (lines 12-14) reports the cache/figure/table were written under `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/...`, a different worktree, not `/Users/pranjal/Code/rl/`. The artifacts in `rl` therefore came from a run in another checkout. Not a defect — the numbers match a fresh independent recompute exactly — but the stdout paths do not correspond to this repo, and `appA_preliminaries/sims/cache/` does not exist here (recompute is deterministic, so `--plots-only` still works).

2. **[Low — cosmetic label]** The header comment (py:5) says the script "Illustrates Gelfand's formula `‖A^k‖^{1/k} → ρ(A)`", but the reported quantity is the consecutive-step ratio `‖A^{k+1}‖/‖A^k‖`, a different sequence (both converge to `ρ` here because `‖A^k‖ ~ C·k·ρ^k`). The user-facing caption in `spectral_radius.tex` is accurate (it names the consecutive ratio explicitly); only the internal comment slightly mislabels which quantity is plotted. A hostile referee could snark at the header, but nothing shipped in the paper is wrong.

3. **[Very low — pedagogical mismatch]** The subsection's tiny-case intro (preliminaries.tex line 13) uses `A=[[0.9,5],[0,0.9]]` (shear 5, `ρ=0.9`), while the figure uses shear 2 and `ρ ∈ {0.5,0.8,0.95}`. The figure caption states its own matrix explicitly, so there is no ambiguity, but the intro example and the figure are different matrices.

4. **[Very low — scope]** The figure demonstrates only the `ρ<1` (decay) direction; the "only if" / `ρ≥1` growth case is neither shown nor claimed by the caption. Appropriate for the figure's stated purpose (norm-vs-rate), noted for completeness.

No correctness, fairness, or leakage defect found. Diagram-only 25% cap does **not** apply: the script genuinely computes iterates (`A^k`), a transient optimum (peak location), and an empirical rate (tail decay ratio), all of which reproduce independently.

**Bullshit score: 10%** — Reviewer 2 might snark that the header comment says "Gelfand `‖A^k‖^{1/k}`" while the table actually reports the consecutive-step ratio, and that the stdout paths point at a different worktree, but every shipped number is exact, the theorem object is computed faithfully, and the caption is accurate.
