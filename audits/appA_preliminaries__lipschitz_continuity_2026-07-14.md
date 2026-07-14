# Simulation Audit — Lipschitz Continuity (Appendix A, Mathematical Preliminaries)

- Sim: `appA_preliminaries/sims/lipschitz_continuity.py`
- Date: 2026-07-14
- Type: FULL (condensed pedagogical appendix sim; never previously audited)
- Auditor stance: hostile journal referee, read-only, evidence only

Files read end to end:
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lipschitz_continuity.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lipschitz_continuity_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lipschitz_continuity.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lipschitz_continuity.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (Thm 410-447 + surrounding prose)
- `/Users/pranjal/Code/rl/sims/sim_cache.py` (flags/caching API)

---

## Step 3 — What is being presented, and what is the figure evidence FOR?

(i) **Mathematical result.** Theorem `thm:prelim_lipschitz` (preliminaries.tex:414), attributed to Nesterov (2018): a map is $L$-Lipschitz if $\|f(x)-f(y)\|\le L\|x-y\|$; for a differentiable real $f$ the least constant is $L=\sup_x\|\nabla f(x)\|$ (eq. `eq:prelim_lipschitz`); for a linear map the least constant is the induced operator norm; and in particular the policy-evaluation operator $TV=r+\gamma PV$ is $\gamma$-Lipschitz in the sup norm, while the resolvent solve $r\mapsto(I-\gamma P)^{-1}r$ is $\tfrac{1}{1-\gamma}$-Lipschitz (preliminaries.tex:421). The proof reads the constant off the mean value theorem (scalar case) and off $\|P\|_\infty=1$ / the Neumann series $(I-\gamma P)^{-1}\mathbf 1=\tfrac{1}{1-\gamma}\mathbf 1$ (preliminaries.tex:430-435).

(ii) **What the sim/figure is evidence FOR.** Two empirical confirmations of that theorem. Panel A (and the table top block): the largest difference quotient $\hat L=\max|f(x)-f(y)|/|x-y|$ over sampled pairs recovers the analytic $L=\sup|f'|$ for four scalar functions, approaching it without ever exceeding it. Panel B (and the table bottom block): the sup-norm operator norm of the resolvent $\|(I-\gamma P)^{-1}\|_\infty$ on random row-stochastic MDPs equals $1/(1-\gamma)$, and the operator norm of $\gamma P$ equals $\gamma$. The prose ties this to the Banach contraction and the Neumann-series/error-amplification bound (preliminaries.tex:438).

---

## Criteria

### (a) CORRECTNESS — theorem identity and numerical consistency: PASS

- The code computes exactly the object the theorem is about. Part (i) evaluates the secant quotient $|f(x)-f(y)|/|x-y|$ (lines 69-76), which is the literal defining ratio in the Lipschitz inequality, and takes its max over pairs. For $C^1$ $f$ this supremum equals $\sup|f'|$ (MVT), so the estimator is the most faithful possible check of the theorem statement, not a proxy.
- Numerics match the theorem's guarantee. `_empirical_lipschitz` can only ever produce a value $\le L$ (since $|f(x)-f(y)|\le L|x-y|$ by the theorem), and stdout confirms `exceeds L=False` with `max over all pairs` equal to $L$ for every function (stdout:11-14). The "rises from below, never exceeds" behavior is theorem-consistent, not merely observed.
- Part (ii) is exact algebra, correctly coded. `op_lip = max abs row sum of gamma*P` (line 126) is the induced $\infty$-norm $\|\gamma P\|_\infty=\gamma$ (P Dirichlet-sampled, row-stochastic). `amp = max abs row sum of (I-gamma P)^{-1}` (lines 128-131) is $\|(I-\gamma P)^{-1}\|_\infty$; since the resolvent is entrywise $\ge 0$ and maps $\mathbf 1$ to $\tfrac{1}{1-\gamma}\mathbf 1$, every row sums to $1/(1-\gamma)$, so the result is exact for every seed. Reported values 2, 3.3333, 10, 20, 100 equal $1/(1-\gamma)$ to machine precision (stdout:17-21).
- No analytic value leaks into any estimate: $L$ and $1/(1-\gamma)$ appear only as reference lines (lines 187-189, 200-207), never inside the estimator.

### (b) PRESENTATION / NUMBERS — traceability and mutual consistency: PASS (one minor caption looseness, see Findings)

- Every number in the `.tex` table traces to `_run_experiment` and matches stdout: top block 0.5000 / 1.0000 / 1.0000 / 3.0000 (tex:9-12 vs stdout:11-14); bottom block operator-Lip $=\gamma$ and amplification $=1/(1-\gamma)$ (tex:16-20 vs stdout:17-21). Table caption seed counts "12 sampling seeds and 20 random MDPs" (tex:3) match CONFIG `pair_seeds=12`, `n_mdp_seeds=20` (lines 38, 43).
- Figure is consistent with both. Panel A curves sit on dashed references at 0.5, 1.0 (tanh and $|x|$ overlapping), 3.0; sin(3x) rises from ~2.9 at $10^2$ pairs to 3.0. Panel B markers land on the $1/(1-\gamma)$ curve at all five $\gamma$. Axes, legends, units all present and correct; markers deliberately unconnected (comment lines 208-209) to avoid a chord reading above the convex theory curve.

### (c) CHAPTER FIT — does figure+caption teach the result cold? PASS

- Panel A titled "Difference quotient rises to $L=\sup|f'|$" with y-axis $\hat L=\max|\Delta f|/|\Delta x|$ directly instantiates eq. `eq:prelim_lipschitz`. Panel B titled "Bellman solve amplifies error by $1/(1-\gamma)$" instantiates the resolvent clause of the theorem. The figure caption (preliminaries.tex:443) names both panels' axes and the dashed references. A cold reader sees the scalar Lipschitz constant and the operator/resolvent norms, the two halves of the theorem, illustrated side by side.

### (d) EFFICIENCY / STANDARDS: PASS

- Seeds fixed and multiple: `pair_seeds=12` and `n_mdp_seeds=20`, both $\ge 10$, via `RandomState(seed_base+si)` (lines 89, 120) — reproducible.
- Flags per CLAUDE.md: uses `add_component_args` / `parse_force_set` / `compute_or_load` (lines 17, 285-287, 149-158), giving `--data-only`, `--plots-only`, `--algo`, verified against `sims/sim_cache.py:122-145`.
- Colors from the centralized palette (`COLORS`, `FIG_DOUBLE`, lines 18, 56-61); no hardcoded hex.
- Stdout is factual and tabular (one line per function / per $\gamma$), no opinion words. Header states the parameters and the two claims.

---

## 7-Point Checklist

1. **Algorithm Identity** — PASS. Estimators ARE the defining objects: secant quotient (line 75), induced $\infty$-norm as max abs row sum (lines 126, 131). Nothing placeholder.
2. **Environment/MDP Fidelity** — PASS. $P$ is row-stochastic by Dirichlet sampling (line 121), matching the "row-stochastic $P$, $\|P\|_\infty=1$" premise of the proof (preliminaries.tex:430-433). $n=40$ states; the theory value is state-count-independent, so the choice is immaterial and correct.
3. **Data Integrity** — PASS. `compute_data` runs `_run_experiment` through `compute_or_load` (lines 149-158); table/figure are written from the returned `data` in `generate_outputs`. Numbers are deterministic (exact for part ii; converged for part i), so no stale-cache risk. Cache is gitignored; tracked outputs (`.png`, `.tex`, stdout) are in git.
4. **Comparison Fairness** — N/A. No competing methods; this is a theorem-vs-measurement check, and the reference ($L$, $1/(1-\gamma)$) is analytic, not a rival algorithm.
5. **Theoretical Sanity** — PASS. Best (only) estimate converges to the known analytic constant; the "never exceeds $L$" guarantee holds exactly (stdout:11-14); amplification blows up as $\gamma\to 1$ exactly as $1/(1-\gamma)$.
6. **No Information Leakage** — PASS. Analytic constants used only as plotted reference lines, never inside the estimator (see (a)).
7. **Seed/Reproducibility** — PASS. Seeds fixed; 12 and 20 seeds. `amp_se` computed (line 136); part (i) reports means without an SE column (quantities are effectively deterministic, SE$\approx$0), see Findings.

Diagram-only 25% cap: **does NOT apply.** The script genuinely computes Monte Carlo difference-quotient maxima and matrix-inverse operator norms (rates/optima), so it is a computational experiment, not a schematic.

---

## Findings (severity-ordered)

1. **(Low, presentation) Caption "rising ... from below for four scalar functions" over-generalizes to the two flat curves.** For $0.5x$ every secant slope is exactly $0.5$, and for $|x|$ any same-sign pair gives exactly $1$, so both hit $L$ at the smallest pair count and plot as flat lines (stdout:11,13 show `L_hat` equal to $L$; blue curve is flat in Panel A). Only tanh and sin(3x) genuinely rise. The figure caption (preliminaries.tex:443) and table caption (tex:3) describe all four as "rising ... from below." A hostile reviewer would note two of four curves are pinned at $L$ for all sample sizes. Substance (never exceeds, converges to $L$) is unaffected. Fix: soften to "approaches $L$ from below, reaching it exactly for the piecewise-linear cases."

2. **(Cosmetic) `_stdout.txt` contains two concatenated runs.** Lines 1-26 are a full compute run; lines 28-39 are a second `Cache hit` run that re-emits the figure/table. Harmless, but the file is meant to capture one canonical run; the duplication is untidy.

3. **(Cosmetic) Part (i) table has no standard-error column** while the Study-Design standard asks for means and SEs across seeds. The operator block computes `amp_se` (line 136) but part (i) reports only the mean over 12 seeds. Acceptable here because the estimated quantities are (near-)deterministic, so SE$\approx$0, but a strict reading of the standard would add it.

4. **(Note, not a defect) Outputs regenerated in a sibling worktree.** stdout output paths point to `/Users/pranjal/Code/rl-theory-proofs/...` (stdout:22-24), i.e. the run happened in a worktree, not the primary `rl` checkout. Numbers are identical and deterministic; consistent with worktree discipline. No action.

---
**Bullshit score: 15%** — Reviewer 2 snarks that two of the four "rising from below" curves are flat lines already sitting on $L$, but the theorem identity is exact and every number matches theory to machine precision.
