# Simulation Audit — Neumann Series / Resolvent (Appendix A, Mathematical Preliminaries)

- **Sim:** `appA_preliminaries/sims/neumann_series.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed variant — small pedagogical appendix sim; never previously audited)
- **Files read (full):**
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/neumann_series.py`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/neumann_series_stdout.txt`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/neumann_series.tex`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/neumann_series.png` (viewed)
  - `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (Theorem block lines 45–90, plus prose at 81 and Lipschitz cross-ref at 438)
  - `/Users/pranjal/Code/rl/sims/plot_style.py`, `/Users/pranjal/Code/rl/sims/sim_cache.py` (convention verification)

## Step 3 — What the appendix is presenting, and what the sim is evidence FOR

(i) **Mathematical result.** Theorem "Neumann series" (`thm:prelim_neumann`, attributed to Horn & Johnson 2013), lines 49–61. For a row-stochastic `P` and `γ ∈ [0,1)`, `I − γP` is invertible, `(I − γP)^{-1} = Σ_{m≥0} (γP)^m`, and the operator-norm truncation error obeys `‖(I−γP)^{-1} − Σ_{m=0}^{M}(γP)^m‖_∞ ≤ γ^{M+1}/(1−γ)` (eq. `prelim_neumann_bound`). The surrounding prose (line 81) instantiates this for RL: the value function `V = (I−γP^π)^{-1} r^π = Σ_m γ^m (P^π)^m r^π`, so truncating at `M` terms is a finite-horizon approximation whose error falls at rate `γ`.

(ii) **What the figure/table are evidence for.** Two claims, both explicit at line 81: (a) the truncation error of the value series decays geometrically at rate `γ` (the figure's parallel semilog slopes), and (b) the per-instance bound `γ^{M+1}‖r‖_∞/(1−γ)` holds and controls how many terms buy a fixed accuracy (the table's measured-vs-predicted term counts). The figure/table are a numerical confirmation of the theorem's rate and bound, not a novel result.

## Criteria verdicts

**(a) CORRECTNESS — PASS.** The code computes exactly the object the theorem concerns, in its value-function instantiation. Ground-truth `V = np.linalg.solve(I − γP, r)` (`neumann_series.py:46`); partial sums via the correct recurrence `partial += term; term = γ·(P @ term)` (`:50–53`), so `err[m] = ‖V − Σ_{k=0}^m γ^k P^k r‖_∞`; bound `rbar·γ^{M+1}/(1−γ)` (`:76`). The plotted vector bound `γ^{M+1}‖r‖_∞/(1−γ)` is the correct corollary of the theorem's matrix bound: `‖V − V_M‖_∞ ≤ ‖resolvent − partial‖_∞ · ‖r‖_∞ ≤ (γ^{M+1}/(1−γ))‖r‖_∞`. Numerics consistent with the guarantee: on the semilog figure every solid error curve is straight with slope `log γ` and lies below its dashed bound; measured terms-to-`10^{-6}` (17/57/276) are each below predicted (20/68/326), the correct direction because the bound is a conservative upper bound (the true asymptotic constant is `|d·r| ≤ ‖r‖_∞`, the stationary-average reward, typically much smaller). No sign of an algorithm beating its own theoretical bound.

**(b) PRESENTATION/NUMBERS — PASS (minor nits).** Every reported number traces to the computation and the three artifacts agree. stdout `:9–11` (`17/57/276`, predicted `20/68/326`) == `neumann_series.tex:9–11` == figure asymptotics. The predicted column reproduces independently: inverting `rbar·γ^{M+1}/(1−γ)=10^{-6}` with a single `rbar = 40/41 = E[max of 40 U(0,1)]` gives 19.90 / 68.02 / 326.27 → 20 / 68 / 326 (exact match, checked by arithmetic). Figure caption (`preliminaries.tex:86`) states "40-state chains, averaged over 30 seeds" matching `CONFIG` `n_states=40, n_seeds=30`; axis labels `‖V−V_M‖_∞` vs "Terms kept M", legend distinguishes solid=error / dashed=bound. Nits: y-axis top runs to ~`10^6` while the largest plotted value is ≈18, leaving ~4 empty decades (curves compressed into the lower half); and the committed stdout's save-path lines point at `/Users/pranjal/Code/rl-theory-proofs/...`, a transient worktree that no longer exists (provenance only — seeds are fixed so numbers reproduce identically here).

**(c) CHAPTER FIT — PASS.** Figure + caption alone teach the result: a cold reader sees three geometrically-decaying error curves, each shadowed by its analytic bound, with steeper decay for smaller `γ` — the caption names the bound formula and ties it to the theorem. The table then answers "how many terms for `10^{-6}`" and shows measured ≤ predicted. Title "Neumann truncation error decays geometrically" states the takeaway. Together they convey rate-`γ` decay and bound-tightness without the body text.

**(d) EFFICIENCY/STANDARDS — PASS (minor).** 30 seeds (≥10), fixed at `200+si` (`:69`), fully reproducible. Uses the shared palette (`GAMMA_COLORS` off `COLORS['blue'/'green'/'red']`, `FIG_SINGLE`) — no hardcoded hex. Multi-component cache API (`compute_or_load`, `add_component_args`, `parse_force_set`) with `--data-only`/`--plots-only` per the script-structure convention; `compute_data` does no plotting, `generate_outputs` does no computation. stdout is header → one fact line per `γ` → output paths, no opinions. Gap: no standard errors or error bands are shown though 30 seeds are run (Study Design asks for "means and standard errors"); the figure plots the mean curve only and the table reports deterministic counts. Immaterial here (a near-deterministic illustration) but a hostile reviewer would note it.

## 7-point checklist

1. **Algorithm Identity — PASS.** The "method" is an exact linear solve plus Neumann partial sums; the recurrence and the sup-norm error match the theorem term-for-term (`:46–53`). No placeholder.
2. **Environment/MDP Fidelity — PASS (nit).** `P = Dirichlet(1_n)` rows, `r ~ U(−1,1)`, `n=40` (`:36–40`) — a valid row-stochastic MDP matching "random 40-state" in the caption. "Chains" is loose wording (these are dense, not sparse tridiagonal chains) but any finite Markov chain has a stochastic matrix, so not a defect.
3. **Data Integrity — PASS.** `compute_data → _run` performs the actual solve and summation; table/stdout values come from computed `mean_err`/`m_to_tol`/`m_pred`, not hardcoded; predicted column reproduced independently above.
4. **Comparison Fairness — PASS.** The three `γ` share identical seeds, `n_terms`, and env generation; the only varied quantity is `γ`. Apples-to-apples.
5. **Theoretical Sanity — PASS.** Exact `np.linalg.solve` is the reference; measured error stays under the analytic bound, decays at rate `γ`, and reaches tolerance in fewer terms than the conservative bound predicts. All consistent with known theory; nothing beats its bound.
6. **No Information Leakage — N/A.** No learning agent; this is a numerical-analysis illustration where the exact resolvent is the intended reference object, not a hidden label.
7. **Seeds/Reproducibility — PASS (minor).** 30 fixed seeds, deterministic; means reported, standard errors not (see (d)).

## Findings (severity-ordered)

1. **[Low — provenance] stdout references a nonexistent sibling repo.** `neumann_series_stdout.txt:12–14` reports cache/figure/table saved under `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/sims/`, a worktree not present in `git worktree list` (only `/Users/pranjal/Code/rl` remains). No scientific impact — seeds are fixed (`200+si`), so re-running in this checkout yields byte-identical numbers — but the committed stdout's paths don't match the repo that houses it.
2. **[Low — cosmetic] Figure y-axis wastes ~4 upper decades.** `set_ylim(1e-10, None)` (`neumann_series.py:119`) autoscales the top to ~`10^6` while max plotted data ≈18, compressing all curves into the lower half. A tighter top (e.g. `1e2`) would use the panel better.
3. **[Low — standards] No standard errors despite 30 seeds.** Figure plots the seed-mean only; table reports deterministic counts. Study Design calls for means and standard errors; here they would be negligible but are absent.
4. **[Info — no defect] Theorem is the matrix operator-norm bound; sim illustrates the ‖r‖-scaled vector corollary.** Caption (`preliminaries.tex:86`) and prose (`:81`) label this correctly and frame it as value-function truncation, so there is no claim/artifact mismatch — noted only so a reader does not expect the figure to test `γ^{M+1}/(1−γ)` directly.

**Bullshit score: 15%** — Reviewer 2 grumbles about the four empty decades of whitespace and an stdout that points at a repo that isn't there, but the math is exact, every number reproduces, and the figure teaches the theorem; substance is airtight. (Genuine computation of iterates/rates, so the 25% diagram-only cap does not apply.)
