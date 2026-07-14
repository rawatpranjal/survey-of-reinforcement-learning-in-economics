# Simulation Audit — Robbins-Monro Stochastic Approximation

- **Sim:** `appA_preliminaries/sims/robbins_monro.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed pedagogical appendix sim; first audit)
- **Files read (end to end):**
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/robbins_monro.py`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/robbins_monro_stdout.txt`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/robbins_monro.tex`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/robbins_monro.png` (viewed)
  - `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 505–554, theorem + proof + prose + figure + table input)
  - `/Users/pranjal/Code/rl/sims/sim_cache.py` (flag/caching interface, lines 42–135)

---

## Step 3 — What the appendix presents, and what the sim is evidence for

**(i) The mathematical result.** Section A.x (`\subsubsection{Robbins-Monro Stochastic Approximation}`, `preliminaries.tex:509`) presents Theorem `thm:prelim_robbins_monro` (Robbins-Monro convergence). For a `γ`-contraction `g` on ℝ with fixed point `x*`, the noisy iteration `x_{t+1} = x_t + α_t(g(x_t) − x_t + w_t)` with mean-zero, bounded-variance martingale-difference noise `w_t` converges almost surely to `x*` provided the two step-size conditions hold: `Σ α_t = ∞` and `Σ α_t² < ∞` (`eq:prelim_rm_conditions`, lines 523–526). A proof via the Robbins-Siegmund almost-supermartingale theorem follows (lines 530–543).

**(ii) What the sim/figure is evidence for.** The figure and table are the empirical illustration that *both* conditions are necessary, not just sufficient. Prose at `preliminaries.tex:545` states: "Figure ... runs (eq rm_update) under four schedules. Only those satisfying both conditions ... reach the root. A constant step size leaves a noise floor. An over-aggressive `1/t²` schedule stalls short, because the iterate can travel only a finite total distance." The sim runs the exact recursion of the theorem for a concrete contraction (`g(x) = γx + b`, `γ=0.5`, `b=1`, so `x* = 2`) under four schedules that each toggle one of the two conditions, and shows: both conditions → error → 0; drop `Σα² < ∞` (constant) → noise floor; drop `Σα = ∞` (`1/t²`) → stall. It is evidence FOR the joint-necessity reading of the two conditions.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS
- **Theorem identity is exact.** The sim recursion `x = x + a*((γ-1)x + b + noise)` (`robbins_monro.py:73`) equals `x_t + α_t(g(x_t) − x_t + w_t)` with `g(x) = γx + b`, a `γ`-contraction, fixed point `x* = b/(1−γ)` (`run_sa`, lines 66–67). Noise is i.i.d. `N(0, σ)` with `σ=1` (line 71), a mean-zero, variance-`σ²` martingale-difference term drawn independently of `x_t` — matches the theorem's `w_t` hypothesis exactly.
- **Condition labeling is correct** (lines 111–112): `cond1 = p ≤ 1` (`Σ 1/t^p` diverges iff `p ≤ 1`); `cond2 = p > 0.5` (`Σ 1/t^{2p}` converges iff `2p > 1`); constant → `cond1=True, cond2=False` (lines 107–109). All four rows of the table are labeled correctly (1/t: yes/yes; 1/t^0.6: yes/yes; constant: yes/no; 1/t²: no/yes).
- **Numerics consistent with the theorem's guarantee.** Final RMS error (stdout 13–16): 1/t = 0.0451, 1/t^0.6 = 0.0766, constant = 0.3488, 1/t² = 1.0827. The two schedules satisfying both conditions drive error toward 0; constant sits at a noise floor; 1/t² stalls near its initial gap. **Rates match RM theory:** on the log-log figure the 1/t trace has slope ≈ −0.46 between t=100 and t=5000 (theory: t^{−1/2}), and 1/t^0.6 ≈ −0.36 (theory: t^{−p/2} = t^{−0.3}). 1/t finishing below 1/t^0.6 is the correct ordering (the `a(1−γ)=0.5` borderline 1/t achieves the faster t^{−1/2} rate).

### (b) PRESENTATION / NUMBERS — PASS
- stdout ↔ .tex ↔ figure are mutually consistent. The four final-RMS values in the `.tex` table (lines 9–12) are byte-identical to stdout (lines 13–16). Figure endpoints (blue ≈ 0.045, green ≈ 0.066, orange ≈ 0.3 noisy floor, red ≈ 1.1 flat) match the table.
- `x* = 2` in the caption (`.tex:3`, formatted from `data['x_star']:.0f`) matches `b/(1−γ) = 1/0.5 = 2`.
- Figure: axes labeled ("Step t", "RMS error ‖x_t − x*‖"), log-log, legend with title "step size α_t", all four traces color-keyed via `SCHED_COLORS` (blue/green/orange/red) from the central palette. Colors in the rendered PNG match the legend.
- Table and figure legend are both in rank order by final RMS (best first), per project convention.

### (c) CHAPTER FIT — PASS
The figure title ("convergence needs both step-size conditions") + caption + the two boolean columns of the table (`Σα=∞`, `Σα²<∞`) let a cold reader read off joint necessity directly: yes/yes rows have low error, either-no rows have high error. Caption correctly names the two failure modes (noise floor for constant, stall for 1/t²). The figure+table+caption teach the stated result without the body prose.

### (d) EFFICIENCY / STANDARDS — PASS (minor nits)
- **Seeds:** 100 seeds, fixed as `7 + si` (line 101), well above the 10-seed minimum. RMS aggregated across seeds (line 102).
- **Flags:** uses `add_component_args`, `compute_or_load`, `parse_force_set`; supports `--data-only` / `--plots-only`, single component `'rm'`, config-hashed cache (lines 12, 128–132, 202–225). Conforms to the multi-component script convention.
- **Color/style:** `apply_style()`, `COLORS`, `FIG_SINGLE` imported and used; no hardcoded hex or `'C0'` shorthand.
- **stdout format:** header with params, one fact line per schedule, no opinion words. Conforms.
- **300 dpi, bbox_inches='tight'** on savefig (line 158).
- Nit: `sum_a`, `sum_a2` computed (lines 104–105) but never used; the condition flags are derived analytically from `p` instead (the correct choice, since a finite 5000-term partial sum cannot diagnose divergence — but then the two `np.sum` calls are dead code).

---

## 7-point checklist

1. **Algorithm identity** — PASS. Recursion (line 73) is the theorem's update verbatim with `g(x)=γx+b`; not a placeholder.
2. **Environment/MDP fidelity** — PASS. The "environment" is the noisy fixed-point iteration of the theorem; `γ=0.5, b=1, σ=1, x0=0, x*=2` all consistent between CONFIG (lines 30–33), `run_sa`, stdout, and caption.
3. **Data integrity** — PASS. `compute_data → _run_experiment` actually runs the recursion (lines 82–125); reported numbers come from `rmse[-1]`, not hardcoded. Condition booleans are analytic labels of the infinite-sum conditions (correct), not fabricated results.
4. **Comparison fairness** — PASS (notably strong). All four schedules use the same seed set and the same number of noise draws per seed (`n_steps=5000` for every schedule), so `RandomState(7+si)` yields the *same* noise realization per seed across schedules — a common-random-numbers comparison differing only in step size.
5. **Theoretical sanity checks** — PASS. Both-conditions schedules → 0; constant → floor; 1/t² → stall; empirical slopes ≈ t^{−1/2} and t^{−0.3} match RM rates. No method beats the "oracle" (root); results do not contradict theory.
6. **No information leakage** — PASS. `x*` appears only in the reported error `|x − x*|` (line 70), never in the update; the update uses only `γ, b, noise`.
7. **Seed and reproducibility** — PASS with one minor gap. Seeds fixed, 100 runs, RMS reported. No explicit standard-error / confidence band is drawn (RMS over seeds is the only spread summary); acceptable for a pedagogical a.s.-convergence illustration but short of the "means and standard errors" letter of the study-design standard.

---

## Findings (severity-ordered)

1. **(Low, provenance)** The committed `robbins_monro_stdout.txt` was produced in a different checkout: lines 17–19 report cache/figure/table saved under `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/...`, whereas the audited repo is `/Users/pranjal/Code/rl` (origin `survey-of-reinforcement-learning-in-economics`, single worktree on `main`). The RNG is fully seeded and deterministic and the stdout numbers match the `.tex` byte-for-byte, so results are unaffected; the paths are just stale to this checkout.

2. **(Low, nit / standards)** `sum_a = np.sum(alphas)` and `sum_a2 = np.sum(alphas**2)` (lines 104–105) are computed but never used; the yes/no condition flags come from the analytic rule on `p`. The analytic rule is the correct one (a finite partial sum cannot show divergence), so the fix is to delete the two dead `np.sum` calls, not to use them.

3. **(Low, standards)** No standard-error bands or confidence interval on the RMS traces; RMS over 100 seeds is the sole aggregate. Fine for a convergence-illustration appendix figure, but nominally below the "report means and standard errors" study-design line.

4. **(Nit, cosmetic)** The y-axis label and caption use norm bars `‖x_t − x*‖` for a scalar iterate, and the caption says "averaged over 100 seeds" for what is a root-mean-square aggregation. Harmless imprecision.

No correctness, identity, fairness, or leakage defect found. Diagram-only 25% cap does **not** apply: the script genuinely simulates 5000-step iterates over 100 seeds and computes RMS errors and empirical rates.

**Bullshit score: 10%** — A hostile reviewer can only carp about missing error bands and the norm notation on a scalar; the theorem identity is exact, the four schedules toggle the two conditions cleanly under common random numbers, and the empirical rates (t^{−1/2}, t^{−0.3}), noise floor, and stall all match Robbins-Monro theory.
