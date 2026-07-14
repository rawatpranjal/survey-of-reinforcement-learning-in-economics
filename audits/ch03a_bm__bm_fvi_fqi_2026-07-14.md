# Simulation Audit — Brock–Mirman FVI vs FQI

- **Sim:** `ch03a_bm/sims/bm_fvi_fqi.py`
- **Date:** 2026-07-14
- **Type:** FULL (never previously audited)
- **Auditor mode:** hostile referee, read-only, no re-execution permitted

**Files read (end to end):**
- `/Users/pranjal/Code/rl/ch03a_bm/sims/bm_fvi_fqi.py`
- `/Users/pranjal/Code/rl/ch03a_bm/sims/bm_fvi_fqi_stdout.txt`
- `/Users/pranjal/Code/rl/ch03a_bm/sims/bm_fvi_fqi_results.tex`
- `/Users/pranjal/Code/rl/ch03a_bm/sims/bm_fvi_fqi_weights.tex`
- `/Users/pranjal/Code/rl/ch03a_bm/sims/bm_fvi_fqi.png` (viewed)
- `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (lines 230–329; the consuming section)
- `/Users/pranjal/Code/rl/sims/plot_style.py` (palette symbols)
- git log / mtimes for all five artifacts

**Outputs the script writes (enumerated from source):** `bm_fvi_fqi.png` (Phase 7), `bm_fvi_fqi_results.tex` (Phase 8), `bm_fvi_fqi_weights.tex` (Phase 8), plus console diagnostics captured to `bm_fvi_fqi_stdout.txt`. The sibling `bm_illustrated.py` writes `bm_convergence.png`, `bm_learning_curves.png`, `bm_policy_curves.png`, `bm_value_functions.png`, `bm_study_results.tex` — none belong to this script.

---

## Step 3 — What claim is this sim evidence for?

**(i) Theoretical claim of the surrounding section** (`\subsubsection{Finite-Sample Theory of Fitted Methods}`, `sec:fvi_fqi_theory`, lines 265–281). Fitted VI and fitted Q-iteration solve *projected* Bellman equations, so their fixed-point error is governed by the inherent approximation error of the function class (Munos–Szepesvári bound, eq. `fvi_error_bound`, with the $(1-\gamma)^{-2}$ amplification). The section argues that whether FQI converges is a property of the *geometry* — specifically whether $Q^*(\cdot,a)\in\mathrm{span}(\Phi)$ for each action — not a property of the algorithm.

**(ii) What this sim is used FOR** (`sec:bm_fvi_fqi`, lines 306–326). Brock–Mirman is the deliberately-chosen *negative* case. With a log-polynomial basis that contains $V^*$ but not the per-action $Q^*(\cdot,a)$, linear FQI is shown to stall while FVI converges; then swapping in the structurally correct log-consumption feature (Oracle-FQI with known $\alpha$, NLLS-FQI estimating $\alpha$) restores FQI convergence. The lesson the sim must sell: *FQI's failure here is a basis-representability failure, not an algorithmic one, and the same FQI algorithm succeeds once the function class contains $Q^*$.*

I verified the theory claim independently. For log utility, Cobb–Douglas $z k^\alpha$, full depreciation, $V^*(k,z)=A_z+B\log k$ with $B=\alpha/(1-\alpha\beta)=0.36/0.6544=0.5501$, and $k'=\alpha\beta z k^\alpha$. For a fixed action $k'$, $Q^*(\cdot,k')(k)=\log(zk^\alpha-k')+\text{const}=\log z+\alpha\log k-\sum_{n\ge1}\tfrac1n(k'/z)^n k^{-n\alpha}+\text{const}$. The $k^{-n\alpha}$ terms are genuinely outside $\mathrm{span}\{1,\log k,k,k^2,k^3\}$, so the tex sentence "requires fractional-power terms $k^{-n\alpha}$ outside the log-polynomial span" is correct, and the oracle basis $[\mathbbm1_{z\ell},\mathbbm1_{zh},\log(zk^\alpha-k')]$ with per-action weights represents $Q^*(\cdot,a)$ exactly (slope on $\log c$ equals 1, the $\beta B\log k'$ and $\beta\mathbb E[A_{z'}|z]$ terms fold into the per-action per-$z$ intercept). The design is sound.

---

## Criteria verdicts

### (a) Correctness — PASS
- Algorithm identity holds term-by-term. `linear_fvi` (155–183) is textbook projected VI: $\theta_{k+1}=(\Phi^\top\Phi)^{-1}\Phi^\top V_{\text{target}}$, $V_{\text{target}}=\max_a[R+\gamma P\Phi\theta_k]$, full model, no sampling. `linear_fqi` (190–239) fits a separate linear model per action $\theta_a=(\Phi_a^\top\Phi_a+\lambda I)^{-1}\Phi_a^\top Q_{\text{target},a}$ with $Q_{\text{target},a}=R(\cdot,a)+\gamma P(\cdot,a,\cdot)\max_{a'}Q_k$. `oracle_fqi` (246–310) uses the known-$\alpha$ log-consumption feature; `nlls_fqi` (317–429) profiles $\alpha$ by concentrated least squares with `minimize_scalar` and a dropout penalty (380–406) that charges infeasible-under-candidate observations `mean(target^2)`, blocking the optimizer from gaming feasibility.
- No information leakage. `V_star` (exact VI) enters only the `errors_to_vstar` diagnostic list and the printed projection floor; it is never used in any fit. Confirmed in all four methods.
- Results match theory. FVI recovers the $\log k$ coefficient 0.5512 / 0.5508 vs analytical $B=0.5501$ (stdout 25, 39–40). Linear FQI stalls at 1.65 (per-action $Q^*$ unrepresentable); Oracle-FQI and NLLS-FQI reach $2.4\times10^{-5}$; NLLS recovers $\hat\alpha=0.360000$ exactly. The non-monotone FVI curve (dips to $2.4\times10^{-4}$ at iter ~253, settles at $1.0\times10^{-3}$) is the expected fitted-VI fixed-point bias passing through $V^*$, not a bug; final error $10^{-3}$ sits within $1/(1-\gamma)$ of the $2\times10^{-4}$ projection floor. No correctness defect found.

### (b) Presentation / numbers — one caption defect, otherwise consistent
- stdout ↔ `results.tex` ↔ figure ↔ prose all agree: FQI 1.6521 / FVI 0.0010 / Oracle 0.0000 / NLLS 0.0000, $\hat\alpha=0.3600$, iters 341/339/341/341. Prose "stalls at error 1.65", "converges to 0.001", "below $10^{-4}$", "$\hat\alpha=0.3600$ in a single iteration" all trace to artifacts (stdout 45–63, `results.tex` 5–9).
- **Defect:** the figure caption (line 324) reads "Left: convergence … Right: NLLS-FQI estimated $\alpha$ trajectory, converging from $\alpha_0=0.5$ to the true $\alpha=0.36$ in one iteration." The generated `bm_fvi_fqi.png` is a **single panel** — `fig, ax1 = plt.subplots(1, 1, ...)` at line 767, only `ax1` is drawn (769–782). `alpha_traj` is computed and cached but never plotted. There is no right panel. The caption describes content that does not exist.

### (c) Chapter fit — demonstrates the claim, but the figure alone under-delivers
The sim is a clean, direct demonstration of the stated lesson: the single panel shows FQI plateauing at 1.65 while FVI, Oracle-FQI and NLLS-FQI descend, which is exactly the "basis, not algorithm" contrast. A cold reader gets the *convergence* half from figure + table. The $\alpha$-recovery half of the lesson lives only in the table row and stdout; the figure caption promises it as a "Right" panel that isn't there, so the "figure + caption alone" test partially fails on its own terms.

### (d) Efficiency / standards — several deviations, none fatal
- `--data-only` / `--plots-only` present via `add_cache_args` (866); stdout format matches CLAUDE.md (param header, per-phase tables, summary table, output paths, no opinions). Good.
- Seeds: `SEED=42` at top (51, 54) but the experiment is fully deterministic (full model, expectations, no Monte Carlo). The ≥10-seeds-with-SE rule is **N/A** — there is no stochastic $N$ to vary, matching the sibling LQC sim's own footnote (line 279). Defensible.
- Caching is monolithic `load_results`/`save_results` on a single `CONFIG` dict (436–446). CLAUDE.md's Simulation Modularity says a multi-algorithm comparison (this has four methods) should use per-component `compute_or_load`. Minor standards deviation. The cache key is `CONFIG` only, with no code hash — see Finding 2.
- Color standards violated: `'steelblue'`, `'darkorange'`, `'#2ca02c'`, `'#d62728'` hardcoded (769–778) although `plot_style` (which defines `ALGO_COLORS`, `COLORS`, `BENCH_STYLE`) is already imported. `FIG_DOUBLE` imported (19) but unused.

---

## 7-point checklist

1. **Algorithm identity** — PASS. FVI, per-action linear FQI, oracle-basis FQI, and NLLS-FQI each match their defining update term-by-term (155–429). No placeholder or penalty-always-zero.
2. **Environment / MDP fidelity** — PASS. Log utility, $z k^\alpha$ with $\alpha=0.36$, full depreciation ($c=zk^\alpha-k'$), $\beta=0.96$, $z\in\{0.9,1.1\}$ with the stated $2\times2$ transition, $N_K=50$, $N_Z=2$, $N_A=50$ (build_reward_and_transitions 65–86). Matches the tex (line 309) exactly.
3. **Data integrity** — QUALIFIED PASS / RISK. `results.tex`, `weights.tex`, `.png` (all mtime 23:36) are mutually consistent with `stdout.txt` (mtime 20:56, ~2h40m older than the `.py` at 23:36). Every published number reconciles with stdout, and every number is theory-consistent by hand-check. But cache is keyed on `CONFIG` alone; a code edit that leaves `CONFIG` unchanged would let a `--plots-only` regeneration emit table/figure from a stale cache with no failure. I cannot re-run (read-only) to prove the *current* code reproduces the committed numbers. No error demonstrated; provenance not guaranteed. See Finding 2.
4. **Comparison fairness** — PASS. All methods run on the same grid, same $R$, same $P$, same ground-truth $V^*$, same iteration cap, same tolerance. FVI fits $V$ in $\mathrm{span}(\Phi)$; FQI fits each $Q(\cdot,a)$ in the same per-function $\Phi$ — the natural framing for a representability claim. (FQI carries 50×10 parameters vs FVI's 10 and still loses, which strengthens rather than flatters the point.)
5. **Theoretical sanity** — PASS. FVI/oracle/NLLS converge toward the DP optimum; linear FQI fails exactly where theory predicts ($Q^*(\cdot,a)\notin\mathrm{span}\Phi$); recovered $B=0.551$ and $\hat\alpha=0.36$ hit analytical values; no method beats the exact-VI oracle.
6. **No information leakage** — PASS. `V_star` used only for error reporting, never in any fit; $\alpha$ is *estimated* in NLLS, only *assumed known* in the explicitly-labelled Oracle method.
7. **Seed / reproducibility** — PASS (with N/A on multi-seed). Deterministic computation; a single run is the population result. Fixed seed present though inert.

**Diagram-only cap:** does NOT apply. This is a genuine numerical experiment (exact VI plus four fitted solvers producing quantitative error metrics), not a diagram, so the 25% cap is not invoked as a ceiling — the score below is reached on the merits.

---

## Findings (severity-ordered)

1. **[Medium-high] Figure caption describes a two-panel figure; only one panel exists.** `planning_learning_v3.tex:324` promises "Right: NLLS-FQI estimated $\alpha$ trajectory," but `bm_fvi_fqi.py:767` builds a single-axis figure and never plots `alpha_traj`. Evidence: figure viewed (one panel, four convergence curves); source lines 767–782. A hostile reviewer reads this as "wrong figure attached." Affects the master arXiv survey (`ch03_theory`) and `thesis/`; `thesis_v2` dodges it by including only the table (line 138). Fix: either add the $\alpha$-trajectory subplot (data already in `alpha_traj`) or strike the "Left/Right" wording.
2. **[Medium] Stale stdout + config-only cache key = unverifiable provenance / drift risk.** Committed `stdout.txt` (20:56) predates the last `.py` edit (23:36); table/figure were regenerated afterward, plausibly from a cache keyed on `CONFIG` (436–455) that carries no code hash. All numbers are internally and theoretically consistent, so no error is shown, but a future code change leaving `CONFIG` untouched would silently republish stale numbers. Read-only mandate prevents a cold-cache re-run to close this. Fix: bump `CONFIG['version']` on any algorithm edit, and re-run to regenerate stdout alongside the table/figure in one pass.
3. **[Low-med] Color-standards violation.** Hardcoded `'steelblue'`/`'darkorange'`/`'#2ca02c'`/`'#d62728'` (769–778) instead of `ALGO_COLORS`/`COLORS` from the already-imported `plot_style`; `FIG_DOUBLE` imported but unused (19). Cosmetic, but an explicit CLAUDE.md rule.
4. **[Low] Orphan output.** `bm_fvi_fqi_weights.tex` is written every run (844–847) but is `\input` by no `.tex` in the repo (grep across all `.tex`: not referenced). Dead artifact.
5. **[Low] Modularity + reference gaps.** Multi-method script uses monolithic caching rather than per-component `compute_or_load` (CLAUDE.md Simulation Modularity). `ch03a_bm/papers/` does not exist, so the Brock–Mirman source is not on disk for validation; the model is standard and I verified its closed forms analytically, so this is a documentation gap, not a correctness one.

---
**Bullshit score: 25%** — Reviewer 2 catches the figure caption promising an $\alpha$-trajectory "Right" panel that the single-panel PNG does not contain, and grumbles that stale stdout plus a code-hash-free cache key leaves the published numbers' provenance unproven; the algorithms, the model, and every reported number are correct and theory-consistent, so the substance survives revision.
