# Simulation Audit — appA_preliminaries / markov_stationary

- **Sim:** `appA_preliminaries/sims/markov_stationary.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed; small pedagogical appendix sim, never previously audited)
- **Diagram-only cap:** Does NOT apply. The script genuinely computes eigenvalues, stationary distributions, power-iteration iterates, and empirical convergence rates, so the 25% cap for diagram-only sims is not invoked.

**Files read (full):**
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/markov_stationary.py`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/markov_stationary_stdout.txt`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/markov_stationary.tex`
- `/Users/pranjal/Code/rl/appA_preliminaries/sims/markov_stationary.png` (viewed)
- `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (theorem block, lines 92–129)
- `/Users/pranjal/Code/rl/CLAUDE.md` (Simulation Audit + Standards)

---

## Step 3 — What the appendix presents and what the sim is evidence FOR

(i) **Mathematical result.** Theorem `thm:prelim_markov` (preliminaries.tex:96–104), "Perron-Frobenius and mixing" attributed to Levin-Peres 2017. For an irreducible, aperiodic (primitive) finite Markov chain with transition matrix `P`, there is a unique stationary distribution `d*` with `d* P = d*`, and from any start the iterates `d_k = d_0 P^k` converge to `d*` with total-variation distance bounded geometrically, `||d_0 P^k − d*||_TV ≤ C |λ₂|^k`, where `λ₂` is the second-largest eigenvalue modulus. The proof (lines 106–118) is the eigenbasis expansion: the eigenvalue-1 term is `d*`, every other term decays at its own eigenvalue, and `|λ₂|` bounds the tail.

(ii) **What the sim is evidence for.** Figure `fig:prelim_markov` and Table `tab:prelim_markov` are the numerical illustration of the theorem. They demonstrate three claims at once, on random 8-state chains at three mixing speeds: (1) the computed `d*` actually satisfies `d* P = d*` (residual column), (2) the TV distance to `d*` decays geometrically, and (3) the empirical decay rate matches `|λ₂|^k` (solid measured curves lie parallel to dashed `|λ₂|^k` reference lines on the log axis), with slower-mixing chains (`|λ₂|` near 1) taking proportionally more iterations to forget the start. Prose at line 120 cites both float and table without hand-typing any number.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS
The code computes exactly the object the theorem is about (theorem identity holds).
- Primitive `P` is built as `P = s·I + (1−s)·R` with `R` a strictly-positive Dirichlet-row stochastic matrix (py:42–49), so `P > 0` entrywise, hence irreducible and aperiodic exactly as the theorem hypothesizes.
- `d*` is the normalized left eigenvector for eigenvalue 1 (py:52–57), and stationarity is independently verified: `resid = max|d* P − d*|` reports `2.2e-16`–`2.5e-16` (stdout:9–11, tex:9–11), i.e. machine precision. This is the `d* P = d*` guarantee.
- `|λ₂|` is the second-largest eigenvalue modulus (py:60–63); measured means 0.8939 / 0.6473 / 0.3742 fall monotonically with stickiness `s = 0.85 / 0.5 / 0.1`, consistent with `λ = s + (1−s)μ` for the subdominant modulus `μ ≈ 0.29` of the shared random `R`.
- Power iteration measures true TV distance `0.5·Σ|d−d*|` from a random Dirichlet start (py:66–74) — no leakage of `d*` into the start.
- Numerics respect the theorem's rate. Independent arithmetic check: with `C ≈ 0.4·`(start error) the geometric prediction `k = ln(C/tol)/ln(1/|λ₂|)` gives 115 / 30 / 13 iterations to `1e-6`, versus measured 123 / 33 / 16. Measured is uniformly ~7–20% above the mean-`λ₂` geometric, which is the correct signature of averaging heterogeneous per-seed geometrics (the mean is dominated at large `k` by the slowest-mixing seed). No result beats a bound it should not.

### (b) PRESENTATION / NUMBERS — PASS
Every number traces to the artifact; stdout ↔ .tex ↔ figure are mutually consistent.
- `|λ₂|`: stdout 0.8939 / 0.6473 / 0.3742 = tex table 0.8939 / 0.6473 / 0.3742 = figure legend 0.89 / 0.65 / 0.37 (2-dp rounding of the same `r['lam2']`).
- Residual: stdout 2.22e-16 / 2.50e-16 / 2.22e-16 = tex 2.2e-16 / 2.5e-16 / 2.2e-16.
- Iters to `1e-6`: stdout 123 / 33 / 16 = tex 123 / 33 / 16.
- Table caption calls the residual `‖d*P−d*‖_∞`; code uses `np.max(np.abs(...))` = ∞-norm (py:92). Match.
- Figure axes/labels correct: x = "Iteration k", y = "TV distance ‖d_k − d*‖_TV", log scale, legend distinguishes solid (measured) vs dashed (`|λ₂|^k`). No hand-typed numbers in prose (line 120 routes everything through `\ref`/`\input`).

### (c) CHAPTER FIT — PASS
Figure + caption alone teach the result to a cold reader. Straight lines on the log axis read immediately as geometric decay; steeper slope for smaller `|λ₂|`; dashed reference lines lie alongside the measured curves so the "rate `|λ₂|^k`" claim is visually self-evident. Caption states `d_k = d_0 P^k`, 8 states, 30 seeds, log scale, and the solid/dashed split, and ties directly back to Theorem `thm:prelim_markov`. The deck-shuffling analogy in the consuming prose (line 120) reinforces it.

### (d) EFFICIENCY / STANDARDS — PASS (minor gaps)
- Seeds fixed and multiple: chain generation `300+si`, power start `900+si`, 30 seeds (py:90–94, CONFIG:31) — exceeds the 10-seed minimum and is reproducible. The same seed index yields the same base `R` across all three chains, so the three mixing speeds are a controlled comparison differing only in stickiness — good design.
- Flags/caching per convention: `add_component_args` / `parse_force_set` / `compute_or_load`, `--data-only` / `--plots-only` (py:112–116, 176–191). Config-hash caching via `CONFIG` with `version:2`.
- Color standards: uses `COLORS` from `sims.plot_style`, `FIG_SINGLE`, no hardcoded hex (py:13, 35–39).
- stdout: header + params + one factual line per chain, no opinions (stdout:6–11). Conforms.
- Gap: means reported but no standard errors, and the figure shows no error band, though CLAUDE.md Study Design asks for "means and standard errors." Low severity for a pedagogical appendix plot. See Findings.

---

## 7-point checklist

1. **Algorithm identity** — PASS. Left-eigenvector `d*`, `|λ₂|`, and TV power iteration match the theorem's objects term-for-term (py:52–74). No placeholder or stubbed penalty.
2. **Environment/MDP fidelity** — PASS. `P = s·I + (1−s)·R` is primitive as the theorem requires; residual `≈1e-16` confirms the constructed `d*` is genuinely stationary (stdout:9–11).
3. **Data integrity** — PASS. `_run()` computes eigen-decompositions and iterates live (py:77–109); table/figure values equal the current stdout; `.py`/`.tex`/`.png` all dated 2026-07-13.
4. **Comparison fairness** — PASS. All three chains share the seed schedule and same base `R`; identical iteration count, tol, and seed set. Apples-to-apples.
5. **Theoretical sanity** — PASS. Fixed point verified to machine eps; empirical rate tracks `|λ₂|^k` on the log axis; iterations-to-tol rank with `|λ₂|` (123 > 33 > 16) and sit ~7–20% above the mean-`λ₂` geometric, as expected from seed-averaging. Nothing contradicts theory.
6. **Information leakage** — PASS. Start distribution is random Dirichlet, not `d*`; `d*` used only to measure distance. `P` is the chain's own matrix, which the theorem is about, not privileged side information.
7. **Seed/reproducibility** — PASS on seeds (fixed, 30). Partial on reporting: no standard errors printed or drawn (see Finding 1).

---

## Findings (severity-ordered)

1. **Low — no standard errors reported or drawn.** CLAUDE.md Study Design requires "means and standard errors"; stdout, table, and figure report means over 30 seeds only, with no SE column and no shaded band on the figure (py:95–108, generate_outputs py:119–150). Does not affect correctness; a reviewer could note the omission. Fix: add an SE column to the table or a light band to the figure.

2. **Low / cosmetic — dashed reference is anchored at the mean, so measured slightly exceeds it at large k.** The dashed curve is `mean_err[0]·(mean |λ₂|)^k` (py:134–141). Because the mean of heterogeneous per-seed geometrics decays slower than the geometric at the mean rate, the solid curve sits above the dashed at large `k` (visible for red/slow at `k≈150`, ~0.7 decade gap in the PNG). The caption calls the dashed line "the `|λ₂|^k` rate," which a hostile reader could misread as an upper bound being violated. It is not a theorem violation (the bound's `C` is chain-specific); it is an averaging artifact. Optional: note "illustrative, at the mean `|λ₂|`" in the caption, or plot the per-seed median.

3. **Cosmetic — stdout paths point to a sibling worktree.** stdout:12–14 write `/Users/pranjal/Code/rl-theory-proofs/appA_preliminaries/...`, not the audited `/Users/pranjal/Code/rl/...`. Expected under the mandatory-worktree workflow; the numbers match the committed `.tex`, so no integrity impact. Purely a provenance note.

4. **Cosmetic — misleading local variable name.** In `_run` the loop variable `eps` (py:85, 90) holds the stickiness `s`, not an epsilon; `random_chain(n, s, seed)` receives it as `s`. Functionally correct, mildly confusing to a reader.

---
**Bullshit score: 15%** — Reviewer 2 dings the missing standard errors and can quibble that the measured curve pokes above the "rate" line, but the sim computes the right objects, verifies `d* P = d*` to machine precision, and reproduces the `|λ₂|^k` mixing rate; the substance holds.
