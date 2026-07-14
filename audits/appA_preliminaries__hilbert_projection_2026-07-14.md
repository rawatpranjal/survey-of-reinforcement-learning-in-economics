# Simulation Audit — Hilbert-Space Projection Geometry

**Sim:** `appA_preliminaries/sims/hilbert_projection.py`
**Date:** 2026-07-14
**Type:** FULL (condensed pedagogical appendix sim; never audited before)
**Auditor role:** hostile journal referee, evidence-only, read-only

**Files read:**
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/hilbert_projection.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/hilbert_projection_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/hilbert_projection.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/hilbert_projection.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 131-164, theorem + consumers)
- `/Users/pranjal/Code/rl/sims/plot_style.py` (palette keys)
- `/Users/pranjal/Code/rl/scripts/run_all_sims.py` (registry line 94)

---

## Step 3 — What the appendix presents, and what the figure is evidence FOR

(i) **Mathematical result.** Section A.1.4 "Orthogonal Projection is Nonexpansive" (preliminaries.tex:131). Theorem~\ref{thm:prelim_hilbert} (attributed to Luenberger 1969): for a closed subspace $M$ of an inner-product space with orthogonal projection $\Pi$, the Pythagorean identity $\|x\|^2 = \|\Pi x\|^2 + \|x-\Pi x\|^2$ holds, and consequently $\Pi$ is nonexpansive, $\|\Pi x - \Pi y\| \le \|x-y\|$. The proof splits $x$ into projection plus orthogonal residual and drops the non-negative residual term.

(ii) **What the sim/figure is evidence FOR.** The figure (Fig.~\ref{fig:prelim_hilbert}) draws the projection geometry, and the table (Tab.~\ref{tab:prelim_hilbert}) numerically checks the two claims of the theorem on one concrete point: that the squared lengths add exactly (Pythagoras) and that the projection is no longer than the original (nonexpansiveness). The prose (preliminaries.tex:155) ties this to the downstream payoff — under on-policy sampling the projected Bellman operator $\Pi T^\pi$ stays a $\gamma$-contraction because $\Pi$ cannot lengthen a vector. The figure/table are pedagogical corroboration of a fact proven analytically in the same subsection, not a load-bearing empirical result.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS
The code computes the genuine orthogonal projection. `project(x,u)` returns $(x^\top \hat u)\hat u$ with $\hat u = u/\|u\|$ (hilbert_projection.py:29-31), the exact closed form of projection onto $\mathrm{span}(u)$. Independently reproduced every table number from `U=[1.0,0.4]`, `X=[1.6,1.7]`:
- $\|x\|^2 = 5.4500$ (matches .tex:9)
- $\|\Pi x\|^2 + \|x-\Pi x\|^2 = 5.4500$ (matches .tex:10) — Pythagoras holds to 4 dp
- $\|\Pi x\| = 2.1169 < \|x\| = 2.3345$ (matches .tex:11-12) — nonexpansiveness holds
- residual $\cdot\,\hat u = 0.000000$ — the residual is orthogonal to the subspace, confirming this is the true orthogonal (not oblique) projection.

The computed object is exactly the object of the theorem, and both numerical guarantees the theorem asserts are satisfied. No contradiction with theory.

### (b) PRESENTATION / NUMBERS — PASS with one gap
Every number in the generated .tex traces to the actual computation and reproduces exactly (verified above). Figure geometry is consistent with the numbers: $\Pi x = (1.966, 0.786)$ sits to the lower-right of $x=(1.6,1.7)$, the red residual meets the gray subspace at a rendered right angle (right-angle marker drawn at hilbert_projection.py:79-95), and the blue $x$ arrow is visibly longer than the green $\Pi x$ arrow (measured ~1315 px vs ~1152 px in the PNG, ratio 1.14 vs the true 2.3345/2.1169 = 1.10). Axes are intentionally hidden (a geometry diagram, aspect set equal at :99). **Gap:** the committed `_stdout.txt` prints only the two file paths and "Done." — it does *not* echo the four validation numbers that the .tex table reports, so stdout carries no results to cross-check (see Findings F1). Also the stdout was generated from a sibling worktree (`/Users/pranjal/Code/rl-theory-proofs/...`, stdout:5-6) rather than this checkout — cosmetic, the path is dynamic via `__file__`, numbers unaffected.

### (c) CHAPTER FIT — PASS
Figure + caption alone teach the result. The caption (preliminaries.tex:160) states the residual meets the subspace at a right angle, the three vectors form a right triangle, and $\Pi x$ is a leg no longer than the hypotenuse $x$ — exactly what the panel shows, with a right-angle marker. The subspace is labelled $\mathrm{span}(\Phi)$, correctly connecting the toy diagram to the linear-function-approximation setting the prose invokes. One minor friction: the in-text intuition (preliminaries.tex:133) walks through $x=(1,1)$ dropped onto the *horizontal axis*, whereas the figure draws a *slanted* 1-D subspace; a cold reader may momentarily expect a horizontal projection (see F2). Not an error — the figure is the more general case and the theorem is dimension/orientation-general.

### (d) EFFICIENCY / STANDARDS — PASS with minor deviations
Deterministic diagram, no stochasticity, so the "≥10 seeds / means and SEs" rule is correctly N/A — the script fixes the geometry at :25-26 and needs no seed. Palette discipline honored: imports `apply_style, COLORS, FIG_SQUARE` from `sims/plot_style.py` (:12), uses named `COLORS[...]` keys throughout, no hardcoded hex or `'C0'` shorthand; all keys exist (plot_style.py:11-22, :98). Figure saved at 300 dpi (:107). Flags `--data-only` / `--plots-only` are present and behave per the CLAUDE.md diagram-only convention (`--data-only` exits with a message :147-149; default path runs `generate_outputs`). Deviations: the stdout violates the "copious tables / print validation metrics" standard by printing no numbers (F1).

---

## 7-point checklist

1. **Algorithm Identity** — PASS. `project()` is the textbook orthogonal-projection formula $(x^\top\hat u)\hat u$; residual verified orthogonal to $\hat u$ (0.000000). It is what it claims.
2. **Environment/MDP Fidelity** — N/A. No MDP; a fixed 2-D geometry illustrating an inner-product-space theorem.
3. **Data Integrity** — PASS. `generate_outputs()` computes `px`, `resid`, and the norms live (:35-37, :112-114) and writes them straight into the .tex; no hardcoded results. Independently re-derived all four table numbers to 4 dp. Outputs share one mtime (2026-07-13 23:41) with the current script — not stale.
4. **Comparison Fairness** — N/A. No competing methods; a single analytic projection with a self-consistency check.
5. **Theoretical Sanity** — PASS. Pythagorean sum equals $\|x\|^2$ exactly (5.4500 = 5.4500) and $\|\Pi x\| < \|x\|$ — both match the theorem's guarantees rather than contradicting them.
6. **Information Leakage** — N/A. Nothing is learned or held out; deterministic geometry.
7. **Seed / Reproducibility** — PASS (by construction). Deterministic, no RNG, reproduces identically every run; multi-seed reporting correctly not applicable.

---

## Findings (severity-ordered)

**F1 (minor, standards).** `hilbert_projection_stdout.txt` prints only file paths and "Done." (stdout:1-8); it omits the four validation numbers ($\|x\|^2$, the Pythagorean sum, $\|\Pi x\|$, $\|x\|$) that the .tex table reports. The CLAUDE.md Stdout Output Format asks for the validation metrics in the console log. Fix: print the same four quantities to stdout in `generate_outputs()`. Low impact — the numbers still trace to live computation and reproduce exactly; only the console audit trail is thin.

**F2 (cosmetic, chapter fit).** The prose intuition (preliminaries.tex:133) projects $(1,1)$ onto the *horizontal axis*; the figure draws a *slanted* subspace $\mathrm{span}(\Phi)$ with a different point. Both are correct instances of the same theorem, but a cold reader may expect the figure to mirror the worked example. Fully defensible as-is.

**F3 (cosmetic, presentation).** Label crowding in the lower-right of the PNG: `span(Φ)`, `Πx`, and the green arrowhead cluster near $(1.9,0.75)$, and the `span(Φ)` text overlaps the gray subspace line. Readable, but tighter offsets would help. The committed stdout also carries a stale sibling-worktree path (`rl-theory-proofs`); harmless.

No correctness, data-integrity, or fairness defects found.

---

## Diagram-only cap note

The script is registered and header-labelled "diagram-only," but it genuinely computes an orthogonal projection (an argmin / closest-point optimum) and numerically verifies an analytic identity, writing live-computed numbers into a table. Per the audit rubric, the 25% diagram-only cap therefore does **not** strictly apply. The score lands well below 25% on its own merits regardless.

**Bullshit score: 10%** — The hostile reviewer can only grumble that the stdout log is empty of the numbers it validates and that the prose's horizontal-axis example doesn't match the slanted figure; the projection is exact, Pythagoras closes to 4 dp, nonexpansiveness holds, and the theorem identity is airtight.
