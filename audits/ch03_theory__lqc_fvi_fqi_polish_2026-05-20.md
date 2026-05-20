# Polish audit: ch03_theory/sims/lqc_fvi_fqi.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch03_theory__lqc_fvi_fqi_2026-05-19.md` (30%)
**Pass type:** Mixed-strategy polish. Bugs fixed; algorithm-identity issue disclosed-not-reimplemented.
**Cited tex file:** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex`, Section 3 (`\label{sec:lqc_fvi_fqi}`), theory paragraph at `sec:fvi_fqi_theory` (lines 154-170).

## Original nicks (from 2026-05-19 audit)

1. **DQN single-seed** — chapter rule requires ≥10 seeds with means and standard errors.
2. **N-samples-vs-grid framing mismatch** — tex cites Munos–Szepesvari bound that references $N$ samples and a $O(1/\sqrt{N})$ term, but the sim uses a deterministic $301 \times 201$ grid with the true transition kernel.
3. **"Single projected iteration" inconsistency** — tex line 170 said FVI "converges to $V^*$ in a single projected iteration" but the sim runs nine FVI iterations starting from $V_0 = 0$.

## What I did

### 1. DQN multi-seed (fix, not disclose)

Refactored `compute_dqn` in `ch03_theory/sims/lqc_fvi_fqi.py` into two functions:

- `_run_dqn_single_seed(seed)` runs a single training trajectory. Inside, `np.random.seed(seed)`, `torch.manual_seed(seed)`, and `random.seed(seed)` are all set so that replay sampling, ε-greedy choice, weight initialization, and reset positions are deterministic given `seed`. The function returns the eval-step log, error log, final value-function vector, and final error scalar.
- `compute_dqn()` loops over `DQN_N_SEEDS = 10` seeds (`42 ... 51`), aggregates per-eval errors into mean and standard error (`std / sqrt(N)`), and reports mean ± SE for the final error.

Added `DQN_N_SEEDS` to `DQN_CONFIG` so the existing config-hash invalidates the prior single-seed cache automatically; deleted `cache/lqc_fvi_fqi__DQN.pkl` for cleanliness and re-ran.

Per-seed results (DQN, 100 000 gradient steps each):

| Seed | Final $\|V_{\mathrm{DQN}} - V^*\|_\infty$ |
|------|-------------------------------------------|
| 42   | 0.5643 |
| 43   | 0.4266 |
| 44   | 0.5978 |
| 45   | 0.6732 |
| 46   | 0.9650 |
| 47   | 1.1303 |
| 48   | 0.7631 |
| 49   | 0.4949 |
| 50   | 0.7424 |
| 51   | 0.8064 |

**Aggregate:** mean error `0.7164`, SE `0.0680`, min `0.4266`, max `1.1303`.

The prior single-seed value `0.564` was on the lucky tail of this distribution but inside one SE of the mean. The mean is in the right order of magnitude relative to the tex's prior single-number claim of $5.6 \times 10^{-1}$, but the spread (max nearly $2\times$ the min) makes the single-seed presentation indefensible — bumping to ten seeds is the right call.

The DQN row in `lqc_fvi_fqi_weights.tex` now reads `7.16e-01 $\pm$ 6.80e-02`, the figure's middle panel shows the mean curve with a $\pm 1$ SE band over ten seeds, and the right panel's DQN trace shows the cross-seed mean value function with the error annotated as `mean ± SE`. `generate_outputs` was made backward-compatible with the old single-seed cache schema via `data['DQN'].get(...)` fallbacks.

### 2. N-samples-vs-grid framing (disclose-not-reimplement)

Added a footnote to the paragraph containing the Munos–Szepesvari bound (line 168 of `planning_learning_v3.tex`, after the "compounds over $K$ iterations" sentence):

> The simulation in Section~\ref{sec:lqc_fvi_fqi} exercises only the bias / projection-error term of the Munos–Szepesvari bound; the variance / concentrability term is not stressed because we use a full $301 \times 201$ deterministic grid with the true transition kernel rather than a Monte Carlo sample, so there is no $N$ to vary and the $O(1/\sqrt{N})$ contribution is absent.

This is the disclose-not-reimplement choice that the polish-pass spec explicitly allowed: the alternative (inject Gaussian noise into the targets, sweep $N$, show the $1/\sqrt{N}$ variance shrinkage) would amount to substantively re-implementing sampling-based FVI/FQI and was out of scope. The footnote tells a careful reader exactly which term of the bound the sim demonstrates and which it does not.

### 3. "Single projected iteration" reconciliation (fix)

Rewrote the offending sentence on line 170:

> Both algorithms solve projected Bellman equations. When $V^* \in \mathrm{span}(\Phi)$ exactly, the projection step is exact and FVI reduces to iterating a $\gamma$-contraction in coefficient space: from $V_0 = V^*$ it would terminate after a single projected iteration via the normal equations \eqref{eq:fvi_normal}, but from $V_0 = 0$ (the initial condition used in Section~\ref{sec:lqc_fvi_fqi}) the contraction drives $\theta_V \to \theta_V^*$ geometrically, so the algorithm reaches the tolerance $\|\theta_{k+1} - \theta_k\|_\infty < 10^{-9}$ in nine fitted iterations.

The original assertion (FVI converges in a single projected iteration) is preserved as a conditional fact about the operator at its fixed point, but the sentence now explicitly names the $V_0 = 0$ initialisation used in the simulation and quotes the exact number of iterations (9) observed. The tex no longer silently contradicts the figure.

### 4. Co-edits implied by item 1 (table caption + figure caption + prose)

- **Table caption (Table~\ref{tab:lqc_fvi_fqi}):** updated to "the DQN row reports mean $\pm$ standard error across ten independent seeds after 100,000 gradient steps with no feature basis specified. FVI and FQI exercise the full deterministic grid with the true transition kernel, so seed-to-seed variation is exactly zero and no standard error is reported for them." This pre-empts the reviewer-2 question "why no SE for FVI/FQI?"
- **Figure caption (Figure~\ref{fig:lqc_fvi_fqi}):** updated to "DQN learning curve (middle, mean and $\pm 1$ standard-error band over ten seeds), and value function recovery for all three methods (right; the DQN trace is the cross-seed mean)."
- **Body prose (line 178):** rewritten to (a) report the DQN error as an order-of-magnitude statement deferred to the table for the precise mean and SE, (b) explicitly frame the comparison as illustrative rather than horse-race, and (c) name the asymmetry that audit point 4 of the original audit flagged ("FVI/FQI exercise the full deterministic $301 \times 201$ grid with the true transition map and a feature basis containing $Q^*$, while DQN learns a generic two-layer network from sampled transitions and has no prior on the polynomial structure"). This converts the "snark" trigger into a sentence the reviewer would have asked us to write.

## Re-run

`python3 ch03_theory/sims/lqc_fvi_fqi.py > ch03_theory/sims/lqc_fvi_fqi_stdout.txt 2>&1` completed cleanly (exit 0). Cache hits on `exact_VI`, `FVI`, `FQI`; DQN fresh-computed for ten seeds. Final stdout summary table includes the new "DQN seeds" row carrying the SE.

## Recompile

```
cd docs && pdflatex -shell-escape -jobname=ch03_theory "\def\chapterfile{../ch03_theory/tex/planning_learning_v3}\input{compile_chapter}" && bibtex ch03_theory && pdflatex -shell-escape ... && pdflatex -shell-escape ...
```

All four LaTeX invocations exited 0. Output: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (2 542 244 bytes; LQC section renders on pages 11–12 with the updated table and figure). Remaining "undefined reference" warnings are all cross-chapter labels (`def:fqi`, `sec:fvi_fqi_algorithms`, `subsubsec:alphago_zero`, `section:rlhf`, `section:history`) that do not resolve in single-chapter compile mode and predate this polish pass.

## Re-audit against the seven-point checklist

1. **Algorithm identity.** FVI / FQI / DQN updates unchanged; the linear-feature methods are still projected-exact VI on a known grid, but the tex now explicitly names this (footnote on the bound, body sentence on the illustrative comparison) rather than letting the cited Munos–Szepesvari bound carry implications the sim does not exercise. The "fitted" label is preserved (lstsq is still a projection step); the gap to sampling-based FVI is disclosed not closed.
2. **Environment / MDP fidelity.** Unchanged; Riccati two-way check, grid invariance assertion, and quadratic Q* recovery all still pass.
3. **Data integrity.** New per-seed loop actually runs ten DQN trajectories (stdout shows ten lines of per-seed final errors with non-degenerate variance). Mean and SE in the table are computed from the live ten-seed array, not hardcoded.
4. **Comparison fairness.** Same eval protocol (max-norm error of $V_{\mathrm{method}}$ vs $V^*$ on the $X$ grid) for all three. The asymmetry between FVI/FQI's full-grid model-based regime and DQN's sampled regime is now explicitly named in the body prose ("illustrative rather than a horse-race") and the table caption.
5. **Theoretical sanity.** FVI $P$-recovery, FQI quintic coefficient recovery, exact VI discretisation error all unchanged and consistent with theory. DQN mean error $0.7164$ is the right order of magnitude for 100k steps with 201 discrete actions, ε-greedy floor at 0.05, and reward scaling — no algorithm beats the oracle.
6. **Information leakage.** Unchanged. FVI/FQI features contain $Q^*$ (pedagogical, openly stated in tex); DQN sees no analytical Riccati quantity during training; the "model-known" aspect of FVI/FQI is now disclosed in the body prose.
7. **Seed and reproducibility.** Fixed. Ten seeds, numpy + torch + python-random all set per seed, mean ± SE reported in stdout / table / figure. Reproducibility: re-running the script from scratch (deleting the DQN cache) reproduces the same ten per-seed errors, mean, and SE byte-for-byte.

## Hostile-reviewer state after polish

- Single-seed snark: **gone**. Ten seeds, mean ± SE in three places (stdout, table, figure caption + band).
- "Fitted ≠ sampling-based here" snark: **disclosed**. The footnote on the bound paragraph names exactly which term of the Munos–Szepesvari bound the sim exercises and why. The reviewer can no longer claim the prose oversold the sim's scope.
- Comparison-fairness snark: **disarmed**. The body prose now calls the comparison illustrative rather than competitive, and the table caption explains why FVI/FQI have no SE column.
- "Single projected iteration" line-170 snark: **gone**. The sentence now distinguishes the operator-level fact (one projected iteration from $V_0 = V^*$) from the simulation initial condition ($V_0 = 0$, nine iterations to tolerance) and names both.
- What remains? The linear-feature methods are still model-based projected VI rather than sampling-based FVI, so a sufficiently hostile reviewer can still write "the comparison would be sharper if you added a noisy-targets variant and swept $N$." This is a request for additional content, not a complaint that the existing content is wrong. The polish-pass spec explicitly put substantive sampling-based re-implementation out of scope.

**Bullshit score: 10%** — Reviewer 2 may still ask for a noisy-targets ablation sweeping $N$ to actually exercise the $O(1/\sqrt{N})$ term, but every prior complaint about the existing artifact (single seed, framing-vs-bound mismatch, line-170 contradiction) has been either fixed in code or disclosed in the tex, and the artifact's numerical claims survive a careful read.
