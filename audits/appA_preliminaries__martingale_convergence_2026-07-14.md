# Simulation Audit: Martingale Convergence (Polya Urn)

- **Sim:** `appA_preliminaries/sims/martingale_convergence.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed variant, small pedagogical appendix sim; never previously audited)
- **Auditor stance:** hostile journal referee, evidence-only, read-only
- **Files read:**
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/martingale_convergence.py`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/martingale_convergence_stdout.txt`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/martingale_convergence.tex`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/martingale_convergence.png`
  - `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 252-284, theorem + prose + figure/table)

---

## Step 3: What the appendix is presenting, and what the sim is evidence for

(i) **Mathematical result.** Theorem `thm:prelim_martingale` (preliminaries.tex:256-264) is the martingale convergence theorem plus its Robbins-Siegmund almost-supermartingale extension. Part one: a supermartingale bounded in L1 converges almost surely to a finite limit `M_∞`. Part two (eq. `prelim_robbins_siegmund`): a nonnegative process obeying `E[Z_{n+1}|F_n] ≤ (1+a_n)Z_n − b_n + c_n` with summable `a_n`, `c_n` converges a.s. and `∑ b_n < ∞`. The prose (line 275) frames this as the probabilistic engine behind Robbins-Monro, Q-learning, TD(λ), and Nash-Q almost-sure convergence.

(ii) **What the sim is evidence FOR.** The figure and table (cited at preliminaries.tex:275, "Figure~\ref{fig:prelim_martingale} follows a bounded martingale... and Table~\ref{tab:prelim_martingale} confirms the martingale property, the settling of every path, and that the random limits follow the predicted law") illustrate the *first* (Doob) part only: a bounded martingale, the red-ball fraction `M_n = red/total` of a Polya urn from a (1,1) start, converges a.s. on every path to a random limit whose law is Beta(1,1) = Uniform(0,1). It is a concrete instance, not a test of the Robbins-Siegmund extension (the prose does not claim otherwise).

---

## Criteria verdicts

### (a) CORRECTNESS — PASS

- **Theorem identity.** The code simulates the exact object the theorem is about. `run_urn` (py:78-89) draws red with probability equal to the current fraction (`drew_red = u < frac`, py:80), which is the defining Polya dynamic, and tracks `M_n = red/total`. Analytically the martingale property is exact: with red `R`, total `T`, `E[M_{n+1}|F_n] = (R/T)(R+1)/(T+1) + ((T−R)/T)R/(T+1) = R(1+T)/(T(T+1)) = R/T = M_n`. Verified by hand.
- **Numerics vs guarantee.** Increment `E[M_{n+1}−M_n] = −3.80e-07 ± 9.31e-07` over 24,000,000 increments (stdout:15), a z-score of −0.41 vs the theoretical 0. The naive SE is *valid here* because martingale differences are uncorrelated (`E[d_n d_m]=0` for `n≠m` by the tower property), so `Var(∑d) = ∑Var(d)`; treating the 24M increments as effectively uncorrelated for a mean-zero test is defensible, not a hidden iid abuse.
- **Limit law.** Simulated mean 0.4985 (theory 0.5000), var 0.0839 (theory 1/12 = 0.0833), KS to Beta(1,1) = 0.0072 (stdout:16). The 5% KS critical value at n=6000 is ≈1.36/√6000 = 0.0176, so 0.0072 is comfortably inside the null. Every path settled (tail oscillation < 0.02 for 40/40 paths, mean 0.0021, max 0.0061; stdout:14). All consistent with the theorem's guarantee. No method beats theory; no leakage.

### (b) PRESENTATION / NUMBERS — PASS (two minor imprecisions)

- stdout ↔ .tex ↔ figure are mutually consistent. Every table entry (increment −3.80e-07 ± 9.3e-07, paths settled 1.000, limit mean 0.4985, var 0.0839, `D_KS` 0.0072) traces byte-for-byte to stdout:14-16. Theory column (0, 1, 0.5000, 0.0833, 0) is arithmetically correct.
- No hand-typed numbers in prose. The consuming paragraph (line 275) and figure caption (line 280) state no numeric values beyond "40 seeds" (matches `traj_seeds=40`) and "Beta(1,1)"; the table caption states "6{,}000 urns" (matches `limit_seeds=6000`). Compliant with the pipeline-first rule.
- Figure axes/legends/units correct: Panel A log-x "Step n", y "red fraction M_n", dashed `M_0` at 0.5; Panel B "limit fraction M_∞" vs "density" with the Beta(1,1) reference at density 1.0. Panel A's leftmost points sit at 1/3 and 2/3 (the value after one draw at recorded step n=1), which is correct, not an error.
- Imprecision 1 (low): stdout:14 labels the tail window "last 10% of steps", but the window is the last 10% of the *log-spaced record grid* (`tail_start = int(0.9*len(rec_idx))`, py:120), which corresponds to steps ≈16946-50000, i.e. the last ~66% of steps by count. Mislabel is stdout-only; the .tex caption does not repeat it.
- Imprecision 2 (low): the table caption (tex:3) calls the mean-zero increment "the martingale property". Averaging *unconditional* increments only verifies `E[M_{n+1}] = E[M_n]`, a necessary-not-sufficient consequence of the conditional property `E[M_{n+1}|F_n]=M_n`. The table *row label* `E[M_{n+1}−M_n]` is itself precise; the conditional property is proven analytically in the theorem, so this is a phrasing overstatement, not a wrong result.

### (c) CHAPTER FIT — PASS

The figure + caption alone teach the result to a cold reader: Panel A shows bounded paths each flattening to a distinct random limit (a.s. convergence made visual), Panel B shows those limits filling [0,1] as Uniform = Beta(1,1). Caption (line 280) names the start, the log axis, the 40 seeds, and the predicted law. Together with the gambler-in-a-fair-game intuition (line 254) the illustration lands the "bounded martingale converges to a random limit" message cleanly.

### (d) EFFICIENCY / STANDARDS — PASS

- Seeds fixed: `seed_base=31000` for trajectories, `seed_base+1` for the ensemble (py:105, 132); 40 trajectory seeds and a 6000-member ensemble, appropriate for a probability-ensemble sim. Increment reported with mean ± SE.
- Flags per CLAUDE.md: `add_component_args`/`parse_force_set`, `--data-only`, `--plots-only`, full-run default (py:296-321). `compute_data`/`generate_outputs` boundary respected — `compute_data` writes no plt/.tex (py:171-180), `generate_outputs` runs no simulation (py:188-288). Uses `compute_or_load` single-component ('martingale'); acceptable.
- Color standards: `apply_style`, `COLORS["blue"]`, `COLORS["red"]`, `BENCH_STYLE`, `FIG_DOUBLE` (py:15, 201, 204, 227); no hardcoded hex.
- Stdout format: header with params, one line per quantity, facts only, no opinion words. Compliant.

---

## 7-point checklist

1. **Algorithm identity** — PASS. `run_urn` implements the exact Polya reinforcement rule (py:80-83); the martingale property is exact analytically and numerically (−0.41 SE from 0).
2. **Environment/MDP fidelity** — PASS. Urn state (red, total), transition (draw-and-replace-plus-one), and observable `M_n = red/total` match the (1,1)-start Polya urn described in caption and table.
3. **Data integrity** — PASS. `compute_data → _run_experiment` actually runs the urns (py:101-168); every reported number references a computed variable. On-disk .tex matches stdout exactly. (Note: stdout paths read `/Users/pranjal/Code/rl-theory-proofs/...`, a temporary worktree; the numbers match the audited `/Users/pranjal/Code/rl/...` artifacts, so this is provenance, not staleness.)
4. **Comparison fairness** — N/A. No competing methods; the only comparison is simulation vs closed-form theory (Beta law, zero increment), which is the correct benchmark.
5. **Theoretical sanity checks** — PASS. Increment matches 0 within SE; limit mean/var match 1/2 and 1/12; KS 0.0072 < 0.0176 (5% crit); all 40 paths settle. Results align with, and do not exceed, the theorem.
6. **No information leakage** — PASS. Dynamics use only the urn's own current fraction; theory values enter only as post-hoc comparison, never inside `run_urn`.
7. **Seed and reproducibility** — PASS. Seeds set explicitly; 40 + 6000 realizations; mean ± SE reported for the increment. Reproducible.

---

## Findings (severity-ordered)

1. **(Low, presentation)** stdout:14 mislabels the settling window as "last 10% of steps"; it is the last 10% of the log-spaced record grid, i.e. steps ≈16946-50000 (verified: `10^(0.9·log10 50000)=16946`). The scientific claim (all paths settled) is unaffected. Confined to stdout, not the .tex.
2. **(Low, presentation)** Table caption (tex:3) equates the mean-zero *unconditional* increment with "the martingale property", which the conditional statement strictly implies but is not implied by. The row label `E[M_{n+1}−M_n]` is precise; the conditional property is proven in the theorem. Substance holds.
3. **(Informational, provenance)** Committed stdout was generated in a `rl-theory-proofs` worktree (paths at stdout:17-19); numbers are identical to the audited artifacts, so no drift, but the stdout does not self-document the canonical `rl` path.

No correctness, fairness, integrity, or leakage defects found. The martingale property is exact, the limit law is essentially exact (KS 0.0072), and the figure faithfully teaches the stated result. This is not diagram-only (it computes 40 trajectories, a 6000-member limit ensemble, an increment mean/SE over 24M draws, and a KS statistic), so the 25% diagram cap does not apply.

**Bullshit score: 15%** — Reviewer 2 can snipe at "last 10% of steps" and at calling a zero mean-increment "the martingale property", but the urn is the exact object, the numbers match theory to their standard errors, and stdout/tex/figure agree byte-for-byte.
