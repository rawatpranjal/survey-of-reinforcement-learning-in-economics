# Audit — appA_preliminaries / lagrangian_duality

- Sim: `appA_preliminaries/sims/lagrangian_duality.py`
- Date: 2026-07-14
- Type: FULL (condensed pedagogical appendix sim; never previously audited)
- Auditor stance: hostile journal referee, read-only, evidence-only

Files read end to end:
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lagrangian_duality.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lagrangian_duality_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lagrangian_duality.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/lagrangian_duality.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 336–372, theorem + proof + prose + figure/table)
- `/Users/pranjal/Code/rl/sims/plot_style.py`, `/Users/pranjal/Code/rl/sims/sim_cache.py` (API confirmation)

---

## Step 3 — what the appendix presents and what the sim is evidence for

(i) **Mathematical result.** Theorem~`thm:prelim_duality` (preliminaries.tex:340–352) is the weak/strong Lagrangian duality theorem for a constrained program `p* = min f(x) s.t. g_i(x) <= 0`. Weak duality `d* <= p*` always holds; if `f`, `g_i` are convex and Slater's condition holds (a strictly feasible point exists), then strong duality `d* = p*` holds, attained at a saddle point (eq. `eq:prelim_saddle`). The proof derives weak duality from multiplier signs and strong duality from a supporting hyperplane to the convex value set.

(ii) **What the sim is evidence for.** The prose (preliminaries.tex:363) states the sim "solves a convex quadratic program through its dual, and Table confirms the dual value stays below the primal optimum throughout and closes to it, with complementary slackness holding at the solution." So the figure/table are numerical illustrations of the theorem's three claims on a concrete convex-QP instance family: (1) weak duality (dual is a lower bound at every iterate), (2) strong duality closing the gap to zero under Slater, (3) complementary slackness at the optimum. The script builds a convex QP `min 1/2 x'Qx + c'x s.t. Ax <= b` with `Q = MM' + nI` (SPD), a strictly feasible interior point enforced by positive slack in `b` (Slater by construction, line 60), derives the closed-form dual `g(λ) = -1/2 (c+A'λ)'Q^{-1}(c+A'λ) - λ'b`, maximizes it by projected gradient ascent with step `1/L_dual`, `L_dual = λ_max(A Q^{-1} A')`, and takes `p*` independently from SLSQP as ground truth (never fed to the dual iteration).

---

## Criteria verdicts

### (a) CORRECTNESS — PASS

Theorem identity holds. The dual function is derived correctly by hand:
with `L(x,λ) = 1/2 x'Qx + c'x + λ'(Ax-b)`, the inner minimizer is `x*(λ) = -Q^{-1}(c+A'λ)` (code line 98), and substituting gives `g(λ) = -1/2 v'Q^{-1}v - λ'b` with `v = c+A'λ` (line 99) — this matches the standard QP dual exactly. The gradient `dg/dλ = A x*(λ) - b` (line 100) is the correct constraint residual. `g` is concave with Hessian `-A Q^{-1} A'`, so its smoothness constant is `λ_max(A Q^{-1} A')` (line 91); projected ascent with step `1/L` is the textbook choice and is provably convergent for an L-smooth concave objective. The projection `max(·, 0)` (line 106) correctly enforces `λ >= 0`.

Numerics are consistent with the theorem's guarantees:
- Strong duality: signed final gap mean `-6.26e-14` (stdout:30), max `|gap| = 7.44e-13` (verified from the 15 per-seed values) — the gap closes to solver precision, exactly what Slater + convexity predict.
- Weak duality: `g(λ_k) <= p*` on all iterates for 15/15 seeds within a `1e-9` tolerance (line 144, stdout:31). The code comments (lines 173–174) are honest that the tiny positive residuals arise because SLSQP's `p*` is itself only good to ~`1e-12`, so the signed gap can dip a few `1e-13` below zero; the dual is a true lower bound on the *exact* `p*`.
- Complementary slackness: `max_i |λ_i (Ax-b)_i|` mean `3.36e-15` (stdout:32), consistent with KKT at the optimum.
- The convergence is faster than a generic `O(1/k)` bound because, once the active set stabilizes, the restricted problem is strongly concave (`A Q^{-1} A'` positive definite on the few active rows), giving linear convergence to machine precision within ~150 iterations (visible in figure panel B). Plausible and not a red flag.

### (b) PRESENTATION / NUMBERS — PASS with one minor notation inconsistency

Every reported number traces to the generated artifact and stdout↔tex↔figure are mutually consistent:
- Table "Final duality gap `<= 7.4e-13`" = `final_gap_abs_max`; independently recomputed max `|final gap| = 7.44e-13` from the 15 stdout lines → rounds to `7.4e-13`. Match.
- Table "Weak duality holds 1.000" = 15/15 (stdout:31). Match.
- Table "Complementary slackness `3.4e-15` (max `1.9e-14`)" = stdout `3.36e-15` mean, seed-3 `1.9e-14` max. Match.
- Table "Active constraints (mean) 1.5 of 5" — recomputed sum of per-seed active counts = 22, `22/15 = 1.467` → `1.5`. Match.
- Figure panel A dashed `p*` sits at ~29.8, matching the stored example (seed 0, `p*=29.8068`, stdout:14). Panel B mean `|gap|` (red) plateaus near solver precision, consistent with the table.

Minor inconsistency: the theorem statement names the dual function `q(λ)` (preliminaries.tex:342), while the figure caption, table, script, and stdout all call it `g(λ)`/`g(λ_k)`. Same object, but a cold reader jumping between the theorem and the figure meets two symbols for one quantity.

### (c) CHAPTER FIT — PASS

Figure + caption teach the result to a cold reader. Panel A ("Dual rises to the primal optimum") shows the dual value climbing monotonically to the `p*` dashed line from below (weak duality, then tightness). Panel B ("Gap closes to zero (Slater)") shows `|p* - g(λ_k)|` on a log axis collapsing to solver precision across 15 instances with the mean highlighted. The caption states axes, panels, legend, seed count, and the Slater reason for closure. Legends and titles are present and correct. The figure alone plus caption conveys weak duality (lower bound) and strong duality (gap → 0), which is exactly the theorem. Complementary slackness lives only in the table, appropriately.

### (d) EFFICIENCY / STANDARDS — PASS

- Seeds: 15 (>= 10 required), deterministic `RandomState(seed_base + si)`, `seed_base=55000` (lines 40, 138). Means and SE reported (stdout:30).
- Caching: per-component `compute_or_load(..., 'duality', CONFIG, ...)` with `CONFIG` hashed (lines 185–194); confirmed `compute_or_load`/`add_component_args`/`parse_force_set` exist in `sims/sim_cache.py`.
- Flags: `add_component_args` supplies `--data-only`/`--plots-only`; `main()` handles all three branches (lines 311–318). Boundary rules respected — `compute_data()` never touches `plt`/`.tex`; `generate_outputs(data)` never trains and works from `data` alone.
- Colors: `COLORS['blue']`, `COLORS['red']`, `BENCH_STYLE`, `FIG_DOUBLE` from `plot_style` (confirmed exports); no hardcoded hex, no `'C0'` shorthand.
- Stdout: header with parameters, one fact line per seed in tabular form, summary statistics, output paths. No opinion words. Conforms to the stdout standard.

---

## 7-point checklist

1. **Algorithm identity** — PASS. Projected dual gradient ascent on the analytic QP dual; update rule (lines 96–107) matches the derivation term-by-term. SLSQP primal is a genuine independent solver, not a stub.
2. **Environment/MDP fidelity** — PASS. The QP (`Q` SPD via `MM'+nI`, linear inequalities, Slater enforced by positive slack in `b`) is a valid convex instance of the theorem; `n=8`, `m=5` as configured.
3. **Data integrity** — PASS. `compute_data()` runs the solver and ascent live; reported table/figure numbers recomputed from stdout and match. No hardcoded "expected" values.
4. **Comparison fairness** — PASS. Same instance per seed feeds both SLSQP (`p*`) and the dual iteration; the two are compared on identical `(Q,c,A,b)` (lines 138–140).
5. **Theoretical sanity** — PASS. Gap closes to `~1e-13` (strong duality under Slater); weak duality holds to tolerance; complementary slackness `~1e-15`. All align with KKT/duality theory. Nothing beats the oracle beyond solver noise.
6. **Information leakage** — PASS. The dual iteration never reads `p*` or `x_p`; comment and code confirm the SLSQP solution is ground-truth-only (lines 8, 65, 139–140).
7. **Seed/reproducibility** — PASS. Seeds fixed, 15 runs, mean±SE reported; deterministic `RandomState`.

Not diagram-only: the script genuinely computes dual iterates, an independent optimum, and a convergence rate, so the 25% diagram cap does **not** apply.

---

## Findings (severity-ordered)

1. **(minor, cosmetic) Dual-function symbol mismatch between theorem and figure/table.** The theorem writes the dual as `q(λ)` (preliminaries.tex:342), everything downstream writes `g(λ)` (figure caption line 368, table `.tex` line 9, script line 99). One object, two names; a reader cross-referencing may stumble briefly. Fix is a one-symbol rename in either the theorem or the figure/table caption.

2. **(informational, not a defect) Weak duality is enforced with a `1e-9` tolerance, not exactly.** Several seeds have `g_final` exceeding `p*` by a few `1e-13` (e.g. seed 0 `gap = -7.18e-13`, stdout:14). This is SLSQP's `p*` precision floor, not a duality violation; the true dual remains a lower bound on the exact `p*`. The code comments (lines 173–174) and the table's "(solver precision)" annotation are honest about it. No action needed, noted for completeness.

3. **(provenance, informational) Stdout was generated in a sibling worktree.** The cache/figure/table paths in `lagrangian_duality_stdout.txt:33–35` point to `/Users/pranjal/Code/rl-theory-proofs/...` (a worktree, now removed), not `/Users/pranjal/Code/rl/...`. All numeric values match the current `.tex`, config (`n_seeds=15`), and re-derived quantities, so this is the correct source-of-truth run under the repo's mandatory-worktree workflow, not stale output. No action needed.

---
**Bullshit score: 10%** — A hostile reviewer catches only the `q` vs `g` symbol swap between theorem and figure; the algorithm is the QP dual it claims to be, every number traces to the run, and the numerics sit exactly where weak/strong duality and Slater predict.
