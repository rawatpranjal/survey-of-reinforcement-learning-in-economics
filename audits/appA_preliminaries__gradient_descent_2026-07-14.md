# Simulation Audit — Gradient Descent Convergence (Appendix A, Mathematical Preliminaries)

- **Sim:** `appA_preliminaries/sims/gradient_descent.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed variant — small pedagogical appendix sim; never audited before)
- **Auditor posture:** hostile journal referee, evidence-only, read-only

**Files read end to end:**
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/gradient_descent.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/gradient_descent_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/gradient_descent.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/gradient_descent.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 289–334, theorem + proof + figure + prose)
- `/Users/pranjal/Code/rl/sims/sim_cache.py` (helper signatures)

---

## Step 3 — What the appendix presents and what the sim is evidence FOR

**(i) The mathematical result.** Theorem `thm:prelim_gd` (attributed to Nesterov 2018), `preliminaries.tex:295`. Gradient descent with step `1/L` on a convex `L`-smooth `f` satisfies the sublinear bound `f(x_k) − f* ≤ L‖x0 − x*‖² / (2k)` (eq. `prelim_gd_sublinear`, line 299). If `f` is additionally `μ`-strongly convex with `κ = L/μ`, the iterates contract geometrically, `‖x_k − x*‖ ≤ (1 − μ/L)^k ‖x0 − x*‖` (eq. `prelim_gd_linear`, line 304). The proof (lines 309–323) routes the linear rate through the contraction-mapping principle: the gradient map `T(x) = x − (1/L)∇f(x)` is a `(1 − μ/L)`-contraction with fixed point `x*`.

**(ii) What the sim is evidence FOR.** Two claims made in the prose (line 325) and caption (line 330): (a) that the per-step iterate contraction of GD equals exactly `1 − μ/L` at three condition numbers `κ ∈ {10, 100, 1000}`, showing iteration count to fixed accuracy scaling with `κ`; and (b) that the general smooth-convex bound `L‖x0 − x*‖²/(2k)` holds throughout for the worst-conditioned case (`κ = 1000`), i.e. the empirical gap never crosses above the bound. The sim is a theorem-illustration, not a research result. It uses strongly convex quadratics `f(x) = ½ xᵀQx`, `Q = diag(λ)`, `λ` spanning `[μ, L]` with the slowest mode isolated at `μ`, `x* = 0`, `f* = 0`.

---

## Criteria verdicts

### (a) Correctness — PASS (with one epistemic caveat)

The object computed is the GD trajectory on the stated quadratic. GD step `1/L` on `f = ½xᵀQx` with diagonal `Q` acts elementwise as `x_i ← x_i(1 − λ_i/L)`, so `contract_i = 1 − λ_i/L` (`gradient_descent.py:77`) and `x_k[i] = x0[i]·contract_i^k`. This is the GD recursion exactly, not an approximation. The slowest mode is `λ_0 = μ`, giving asymptotic iterate contraction `1 − μ/L`, which is precisely eq. `prelim_gd_linear`. Numerics: measured factors `0.90000 / 0.99000 / 0.99900` equal theory `1 − μ/L` to five decimals (`gradient_descent.tex:9–11`, `_stdout.txt:14–16`). The `O(1/k)` bound is enforced as `sup_k f_k · 2k / (L‖x0‖²) ≤ 1` (`gradient_descent.py:113–114`); measured maxima `0.105 / 0.122 / 0.120`, all `< 1`, consistent with eq. `prelim_gd_sublinear`. The theorem is an inequality; the sim demonstrates the linear rate is *tight* (achieved by the isolated slow mode), which matches the prose caveat that quadratics are used "to make the constant sharp" (line 325). No correctness defect.

Caveat, not a bug: the trajectory is evaluated in closed form (`C2K = contract^{2k}` matrix, lines 82–84) rather than by literally looping the GD update. It is bit-for-bit the same sequence, so this is a legitimate vectorization, but see finding F1 on what "measured" then means.

### (b) Presentation / numbers — PASS

Every number in the `.tex` table traces to `_stdout.txt` and to the computation. Table row `κ=10`: `μ/L=0.1000`, measured `0.90000 ± 4e-08`, theory `0.90000`, bound ratio `0.105` — matches stdout line 14 (`± 4.1e-08`, `0.105`). Rows `κ=100` (`0.99000 ± 7e-17`, `0.122`) and `κ=1000` (`0.99900 ± 2e-17`, `0.120`) match stdout lines 15–16. Caption (line 330) correctly names both panels, both dashed references, and the axis scalings (semilog-y left, log-log right). Figure axes are labelled `f(x_k) − f*` vs iteration `k`, legends present, `κ`-colour scheme consistent across panels. Left panel: measured curves run parallel-below their dashed geometric envelopes (correct: the envelope carries full `f0` mass but the tail is only the slow-mode fraction, so it is a valid loose upper bound). Right panel: red `κ=1000` gap stays under the black `1/(2k)` line throughout, as the caption states. `.png`/`.tex`/`.py` share mtime `13 Jul 23:41` → outputs in sync.

### (c) Chapter fit — PASS

Figure + caption teach the result to a cold reader: two panels map one-to-one onto the theorem's two rates (geometric in the strongly convex case, `O(1/k)` general bound), legends distinguish measured vs theory, and the log scales make the geometric slope and the `1/k` slope legible. The surrounding prose (lines 293, 325) supplies the round-bowl / narrow-valley intuition and the `κ` interpretation. Self-contained.

### (d) Efficiency / standards — PASS (minor)

Seeds set explicitly (`seed_base = 44000 + si`, line 92), 20 seeds, means and SEs reported per CLAUDE.md study-design minimum. `compute_data(force)` wraps computation via `compute_or_load`; `generate_outputs(data)` does all plotting/`.tex` writing and touches no training — the boundary rules hold. Flags `--data-only`/`--plots-only` present via `add_component_args`. Palette imported from `sims.plot_style` (`COLORS`, `FIG_DOUBLE`), no hardcoded hex. Stdout is factual, tabular, opinion-free. Minor: stdout file contains two concatenated runs (a forced recompute, lines 1–21, then a plots-only cache-hit pass, lines 22–34) rather than one clean capture.

---

## 7-point checklist

1. **Algorithm identity — PASS.** `contract = 1 − λ/L` (line 77) is the exact GD-step-`1/L` map on a diagonal quadratic; closed-form power (lines 82–84) equals the recursion.
2. **Environment fidelity — PASS.** `f = ½xᵀQx`, `Q = diag(λ)`, `λ ∈ [μ, L]`, `μ = L/κ`, `x* = 0` (lines 68–76) matches the theorem's convex/`L`-smooth/`μ`-strongly-convex setup. Note the deliberate slow-mode isolation (finding F2).
3. **Data integrity — PASS.** `compute_data` → `_run_experiment` runs the real computation; table/stdout numbers are computed, none hardcoded. Cache written under the `rl-theory-proofs` worktree (stdout lines 17–19); primary-checkout cache dir is empty (gitignored), but the committed `.tex` values match, so no staleness.
4. **Comparison fairness — N/A.** No competing algorithm; the comparison is measured-rate vs analytic-rate under identical conditions. See F1 for why this comparison is near-tautological.
5. **Theoretical sanity — PASS.** Measured factor equals `1 − μ/L` to 5 dp; `O(1/k)` bound ratio `< 1` everywhere; both align with the cited bounds. No result exceeds a theoretical guarantee.
6. **Information leakage — N/A.** No learning agent, no held-out data; a deterministic theorem-verification sim. (The closed form uses the true structure by construction — expected here.)
7. **Seed / reproducibility — PASS (weak).** Seeds fixed, 20 runs, mean ± SE reported. But for `κ=100, 1000` the SE is `~7e-17 / ~2e-17` (machine epsilon) because the measured factor is deterministic across `x0` directions; the seed loop adds no statistical content to the headline number (finding F3).

---

## Findings, severity-ordered

**F1 (moderate, epistemic — not a bug). "Measured factor" is near-tautological.** The trajectory is the analytic geometric sequence `x0·contract^k` with `contract = 1 − λ/L` (lines 77, 82–84). The "measured asymptotic per-step factor" (lines 104–111) then reads back the base of that sequence, dominated by the isolated slow mode `μ`, and unavoidably returns `1 − μ/L`. So "measured matches theory" (table caption; prose line 325) is not an empirical convergence finding — it is an algebraic identity dressed as a measurement. The substance (the theorem's two rates, the figure, all numbers) is fully correct and survives; only the word "measured" over-promises. Reviewer-2 bait.

**F2 (minor). Designed-to-succeed eigenvalue spectrum.** `λ[0]=μ` is isolated below a cluster in `[√(μL), L]` (lines 74–76) explicitly so the slow mode dominates the tail "quickly ... rather than a blend of neighboring modes" (comment lines 70–73). Any spectrum recovers `1 − μ/L` asymptotically; the isolation is chosen to make it recover cleanly within the window. Documented and honest, but it is a setup engineered to hand back the headline constant.

**F3 (minor). The `O(1/k)` bound is exercised only where it is very slack.** The sublinear bound (eq. `prelim_gd_sublinear`) is a *general* convex claim, but it is checked only on strongly convex quadratics, where the true gap decays geometrically and the bound is loose by ~10x (ratios `0.105–0.122`). The bound is never stressed near tightness on a genuinely non-strongly-convex `f`. The prose only claims it "holds throughout," which is true, so this is a weak-demonstration note, not an error.

**F4 (cosmetic). Machine-epsilon SEs and doubled stdout.** SEs of `7e-17 / 2e-17` (table lines 10–11) signal a deterministic quantity, making the 20-seed apparatus decorative for those rows. Separately, `_stdout.txt` concatenates two runs (lines 1–21 then 22–34) instead of one clean `> stdout.txt 2>&1` capture.

---

**Bullshit score: 20%** — Reviewer 2 rightly snarks that the "measured" contraction is the analytic base of a closed-form geometric sequence, not an empirical measurement, and that machine-epsilon SEs over 20 seeds dress up an identity; but the method is exactly GD as named, every number is real, correct, and self-consistent, and the substance survives revision untouched.
