# Simulation Audit: Banach Contraction (Appendix A, Mathematical Preliminaries)

- **Sim:** `appA_preliminaries/sims/banach_contraction.py`
- **Date:** 2026-07-14
- **Type:** FULL, condensed variant (pedagogical single-figure appendix sim; never previously audited)
- **Auditor role:** hostile journal referee, read-only, no re-execution
- **Files read (end to end):**
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/banach_contraction.py`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/banach_contraction_stdout.txt`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/banach_contraction.tex`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/banach_contraction.png` (viewed)
  - `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (Theorem env + surrounding prose, lines 455-508)
  - `/Users/pranjal/Code/rl/sims/plot_style.py` (palette)

---

## Step-3 statement (what is presented, what the figure is evidence for)

(i) **Mathematical result.** Theorem `thm:prelim_banach` (line 458) is the Banach fixed-point theorem: on a complete metric space, a γ-contraction T (γ ∈ [0,1), d(Tx,Ty) ≤ γ d(x,y)) has a unique fixed point x★, and the iterates x_{k+1}=Tx_k satisfy the geometric error bound d(x_k, x★) ≤ γ^k d(x_0, x★) (eq. `eq:prelim_banach_rate`). The prose then specializes it to the policy-evaluation operator T^π V = r^π + γ P^π V, a γ-contraction in the sup norm because P^π is row-stochastic (line 498).

(ii) **What the sim/figure is evidence FOR.** It is the numerical illustration of that specialization. It iterates T^π V = r + γPV on 30 random 50-state Markov reward processes at γ ∈ {0.5, 0.7, 0.9, 0.99}, from V_0 = 0, and plots the measured sup-norm error ‖V_k − V★‖_∞ against the analytic γ^k‖V_0 − V★‖_∞ envelope. It substantiates three claims in line 498: (a) the measured per-step contraction factor tracks γ, (b) the error stays below the γ^k envelope, (c) iterations to a fixed tolerance grow like 1/(1−γ) (Table `tab:prelim_banach`). V★ is obtained exactly by direct solve V★ = (I − γP)^{-1} r, which serves as the reference the deterministic iteration is compared against.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS

Theorem identity is exact. The code operator (`banach_contraction.py:71`, `V = r + gamma * (P @ V)`) is verbatim the policy-evaluation operator T^π V = r + γPV named in the theorem's specialization (`preliminaries.tex:498`). The fixed point is computed as the exact linear solve V★ = (I − γP)^{-1} r (`banach_contraction.py:66`); this is provably the unique fixed point since ‖γP‖_∞ = γ < 1 makes I − γP invertible via the Neumann series. Row-stochasticity of P (Dirichlet rows, `banach_contraction.py:55`) gives ‖P‖_∞ = 1, so the operator is a genuine γ-contraction in the sup norm, matching the theorem hypothesis.

Numerics are consistent with the theorem's guarantee. Because V_k − V★ = (γP)^k(V_0 − V★) and ‖P^k‖_∞ = 1, the identity forces err[k] ≤ γ^k err[0] at every k, i.e. the measured solid curve must lie at or below the dashed γ^k envelope. The figure confirms this for all four γ (blue/orange/green solid strictly below their dashed bounds; red solid essentially coincident with its bound). Measured contraction factors (0.4626, 0.6776, 0.8915, 0.9881; `stdout:14-17`) all sit just below their respective γ, consistent with the ≤ bound and with the asymptotic ones-eigenvector rate → γ approached from below as the faster sub-dominant modes die out. Spot-check: γ=0.5 back-solves to err[0] ≈ 0.86 from the predicted-iterations formula, matching the blue curve's k=0 intercept in the figure; γ=0.99 back-solves to err[0] ≈ 7.8, matching the red intercept. Internally coherent.

### (b) PRESENTATION / NUMBERS — PASS with two catchable issues

The measured contraction factors and their SEs in `banach_contraction.tex:9-12` (0.4626 ± 0.0026, etc.) match `stdout:14-17` byte-for-byte, and the figure I viewed matches (four γ, log-y, solid + dashed). All four output files share mtime 2026-07-13 23:41, so they were regenerated together (no staleness). Axes, legend, and units are correct: y = ‖V_k − V★‖_∞ on log scale, x = iteration k, legend distinguishes solid=measured / dashed=γ^k bound. Palette is the centralized one (`plot_style.py:12-15`, blue #4878A8 / orange / green / red), matching the rendered figure.

Two issues (detail in Findings): the stdout reports **measured** iterations-to-tolerance (30.4, 60.1, nan, nan; `stdout:14-17`) while the `.tex` table reports **predicted** iterations (33, 65, 223, 2495; `banach_contraction.tex:9-12`) — different estimators, and for γ=0.9/0.99 the measured value is nan because the 120-iteration run never reaches 10^{-10}, so the table's 223 and 2495 are analytic extrapolations the simulation itself never attains. And the table caption calls a dense Dirichlet transition matrix a "chain."

### (c) CHAPTER FIT — PASS

Figure plus caption teach the result to a cold reader unaided. The caption (`preliminaries.tex:503`) states exactly what is drawn: sup-norm error of the policy-evaluation iterates on 30-seed 50-state MRPs, solid = measured, dashed = the γ^k‖V_0−V★‖_∞ bound of Theorem `thm:prelim_banach`. A reader sees four log-linear decays whose slopes steepen as γ shrinks, each pinned at or below its dashed envelope, which is a direct visual reading of d(x_k,x★) ≤ γ^k d(x_0,x★). The table reinforces "measured factor ≈ γ." The illustration is legible without the body prose.

### (d) EFFICIENCY / STANDARDS — PASS

30 seeds (`CONFIG`, `banach_contraction.py:32`), exceeding the ≥10 minimum; seeds fixed and deterministic (100+si, `banach_contraction.py:99`); the same 30 MDPs are reused across all four γ, so the cross-γ comparison is controlled. Caching via `compute_or_load` with a versioned CONFIG hash; flags `--data-only` / `--plots-only` / `--force` wired through `add_component_args` / `parse_force_set` per the CLAUDE.md script-structure convention; `compute_data` writes no plots and `generate_outputs` runs no training, respecting the boundary rules. Stdout is factual and tabular (header with params, one line per γ, output paths), no opinion words. Compute is trivial (50×50 solve × 30 seeds × 120 iters).

---

## 7-point checklist

1. **Algorithm identity — PASS.** `banach_contraction.py:71` is exactly T^π V = r + γPV; V★ = (I−γP)^{-1}r at line 66 is the exact fixed point. No placeholder, no missing term.
2. **Environment / MDP fidelity — PASS (minor wording).** 50-state row-stochastic Dirichlet P, reward r ~ U(−1,1) (`banach_contraction.py:55-56`), matching "random 50-state Markov reward processes" in the figure caption. The table caption's word "chain" is loose for a fully-connected random stochastic matrix (see Findings).
3. **Data integrity — PASS.** `_run_experiment` actually runs the solve and the 120-step iteration; reported numbers trace to computed variables; `.tex` values equal stdout values.
4. **Comparison fairness — N/A.** No method-vs-method comparison. The only contrast is the measured iterate error vs the analytic γ^k bound on those same iterates, evaluated on identical MDPs and seeds across γ — fair by construction.
5. **Theoretical sanity — PASS.** Measured factor ≤ γ for all γ; error ≤ γ^k envelope everywhere (figure); iterations-to-tolerance scale in the 1/(1−γ) family (2 → 33, 3.3 → 65, 10 → 223, 100 → 2495, with a slow log correction from the growing initial error). No method beats the analytic bound; nothing contradicts theory.
6. **Information leakage — N/A.** Nothing is learned. V★ is computed by direct linear solve as the deterministic iteration's known target; using it as the reference for the error is the intended construction, not test-time leakage.
7. **Seed / reproducibility — PASS.** 30 fixed seeds, means and SEs (ratio SE) reported; seeds set explicitly; results deterministic.

---

## Findings (severity-ordered)

**F1 (medium-low, presentation). stdout and `.tex` report different iterations-to-tolerance estimators, and the table's γ=0.9/0.99 entries lie beyond the simulated horizon.** `stdout:14-17` prints the *measured* iterations to reach 10^{-10} (30.4, 60.1, nan, nan), computed at `banach_contraction.py:107-108`; `banach_contraction.tex:9-12` prints the *predicted* count log(tol/err0)/log(γ) (33, 65, 223, 2495), computed at `banach_contraction.py:123`. They are close where both exist (30.4↔33, 60.1↔65, and the measured-below-predicted gap is itself correct since measured decay is faster than the γ-bound), but the column cannot be cross-checked between the two artifacts, and for γ=0.9 and 0.99 the *measured* value is nan because the 120-iteration run (`CONFIG n_iters=120`) never reaches the tolerance — so the headline 223 and 2495 are pure formula extrapolations the simulation never demonstrates. The `.tex` header honestly labels the column "(predicted)", so this is not fabrication, but a cold reader can misread it as an achieved count, and the figure visibly stops the green/red curves well short of 10^{-10}. Consider either running long enough to reach tol for all γ, or reporting the measured count alongside the prediction.

**F2 (low, terminology). Table caption calls a dense random stochastic matrix a "chain."** `banach_contraction.tex:3` says "random 50-state chain"; the dynamics are Dirichlet(ones) rows (`banach_contraction.py:55`), i.e. a fully-connected transition matrix, not a chain in the nearest-neighbor sense. The figure caption in `preliminaries.tex:503` correctly says "random 50-state Markov reward processes." Harmonize the two captions.

**F3 (low, style / CLAUDE.md Rule 5). Table caption embeds an interpretation, and the "measured factor" is a transient-biased estimator.** `banach_contraction.tex:3` states the factor "approaches γ from below as the transient sub-dominant modes decay" — a conclusion that Rule 5 places in the results prose, not the caption. Substantively the "measured contraction factor" is a geometric mean of the per-step ratio taken from a zero start to the tolerance crossing (`banach_contraction.py:104-106`), which by construction folds in the fast sub-dominant transient and therefore undershoots γ (0.4626 vs 0.50 is a 7% undershoot). The claim "tracks γ" (`preliminaries.tex:498`) survives because the caption explains the from-below behavior, but an asymptotic (last-few-steps) ratio would sit closer to γ and would be the less contestable estimator.

---

**Bullshit score: 20%** — Reviewer 2 catches that the table's iterations column is a prediction the 120-step run never reaches for γ=0.9/0.99 and that stdout reports a different (measured) column, plus the "chain" caption slip, and writes a snarky note; the theorem identity is exact, the γ^k envelope holds, seeds are ample, and the substance is untouched. Diagram-only 25% cap does NOT apply: the script genuinely computes iterates and per-step contraction rates, not just a picture.
