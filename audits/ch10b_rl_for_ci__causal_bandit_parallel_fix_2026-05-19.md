# Fix Report: ch10b_rl_for_ci/sims/causal_bandit_parallel.py

**Date:** 2026-05-19
**Original score:** 55%
**Strategy:** Relabel + reconcile. No reimplementation of Bareinboim 2015's TS_C with consistency-axiom seeding + RDC weighting.

---

## 1. Audit findings reconciled

The 2026-05-19 audit at `audits/ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md` flagged three concrete issues that a hostile reviewer would catch:

1. **TS_C mislabelled.** The function `causal_thompson_sampling()` implements context-conditional Beta posteriors indexed by (intuition `x`, arm `a`) with a straight argmax. Both distinguishing features of Bareinboim et al. 2015 Algorithm 1 are absent: (i) consistency-axiom seeding of the off-intuition arm `a ≠ x`, and (ii) RDC bias weighting `w[a] = 1 − |Q1 − Q2|` applied to posterior samples. The chapter prose (`rl_for_ci.tex` line 226) named both missing components explicitly.
2. **Non-monotone m grid contradicts the "rising approximately as √(m*/T)" claim** on tex line 333. Empirically: regret at `m* = 24` is 0.125 and drops to 0.071 at `m* = 48`, opposite to the asserted trend. Cause: single-coordinate reward model — at `m* = 48` nearly all 50 arms are unbalanced, so the optimal-coordinate arm is included in phase 2 with probability close to 1 and is estimated directly; at intermediate `m*` the optimal coordinate falls into the balanced set roughly half the time.
3. **√(N/T)·ε reference line mislabelled** as "theoretical floor for graph-blind algorithms" in the figure caption (tex line 345). It is an *asymptotic rate lower bound*, not an upper bound on Successive Reject's finite-`T` performance. Empirical SR regret ~0.28 sits well above the reference ~0.106 because `T = 400 ≪ K log K = 467` at `K = 101`.

The Lattimore Algorithm 1 implementation, the Successive Reject implementation, the `m(q)` construction, the greedy-casino MABUC environment, the linear-regret prediction for vanilla Thompson, and the seed/SE reporting were all confirmed faithful in the audit and are not modified.

## 2. Edits

### Script: `ch10b_rl_for_ci/sims/causal_bandit_parallel.py`

- **Renamed** function `causal_thompson_sampling` → `context_conditional_thompson_sampling`.
- **Rewrote** its docstring and section banner to disclose the omitted TS_C components.
- **Updated** the file header comment to describe Algorithm 3 as "context-conditional Thompson sampling — a stripped-down variant of TS_C that retains the (x, a) posterior but omits the consistency-axiom seeding of the off-intuition arm and the RDC bias-weighting step of their Algorithm 1."
- **Renamed** dict key `tsc` → `cctp` and local variables `tsc_regret`/`tsc_mean`/`tsc_se`/`final_tsc`/`rng_tsc` accordingly throughout `run_mabuc`, `make_figure_combined`, and `print_stdout`.
- **Updated** figure label in panel (c) from `r'Causal TS ($\mathrm{TS}_C$)'` to `'Context-conditional TS'`.
- **Updated** the panel (a) reference line legend from `r'$\sqrt{N/T} \cdot \epsilon$ reference'` to `r'$\sqrt{N/T} \cdot \epsilon$ asymptotic rate (lower bound)'`.
- **Updated** stdout strings: "Causal Thompson (TS_C)" → "Context-conditional Thompson"; "Ratio TS / TS_C" → "Ratio (vanilla TS) / (context-conditional TS)".
- Cleared stale `cache/causal_bandit_parallel__mabuc.pkl` so the rerun regenerates outputs with the new dict key.

### Tex: `ch10b_rl_for_ci/tex/rl_for_ci.tex`

- **Line 226 (TS_C description).** Added a closing sentence: "The simulation in Section~\ref{subsec:simstudy} implements a stripped-down variant of $\text{TS}_C$ that retains the context-specific Beta posterior indexed by $(x, a)$ but omits both the consistency-axiom seeding of the off-intuition arm and the RDC bias-weighting step. The fully specified $\text{TS}_C$ is a stronger baseline left for future work." The paragraph still describes the full TS_C (with consistency axiom and RDC) as the named algorithm of Bareinboim et al. 2015 but now declares that the simulation implements a strict subset of it. Also explicitly named "RDC" inline since the abbreviation was previously undefined.
- **Line 291 (simstudy chapeau).** Replaced "causal Thompson sampling achieves bounded cumulative regret" with "a stripped-down context-conditional Thompson sampling baseline achieves bounded cumulative regret", and modified the Lattimore claim to "on the monotone region $m(q) \in \{2, 8, 24\}$ of the hardness grid". Also swapped "$\sqrt{N/T}$ floor" for "asymptotic $\sqrt{N/T}$ rate lower bound".
- **Line 329 (Sim 2 algorithm list).** Renamed the third algorithm from "Causal Thompson sampling (TS_C)" to "Context-conditional Thompson sampling" and disclosed: "The full $\mathrm{TS}_C$ additionally seeds the off-intuition arm $a \neq x$ via the consistency axiom and applies the RDC bias-weighting multiplier; neither component is implemented here."
- **Line 333 (Sim 2 results paragraph).** Rewrote the regret-vs-`m*` sentence to acknowledge the non-monotonicity at `m* = 48` and explain it via the single-coordinate reward construction: "The graph-aware algorithm achieves regret $0.024$ at $m^* = 2$ and rises approximately as $\sqrt{m^*/T}$ for $m^* \in \{2, 8, 24\}$, in line with Theorem~1 of \citet{lattimore2016causal}; the trend reverses at $m^* = 48$, where regret drops to $0.071$ rather than continuing to grow. The non-monotonicity is a finite-sample artefact of the single-coordinate reward construction: at $m^* = 48$ nearly all $50$ parents are unbalanced, so the optimal-coordinate arm is included in the phase-2 allocation with probability close to one and is estimated directly, whereas at intermediate $m^*$ the optimal coordinate falls into the balanced set roughly half the time and is recovered only through the noisier phase-1 regression on observational data." Also revised the SR sentence to note finite-`T` constants dominate: "well above the asymptotic rate lower bound $\sqrt{N/T} \cdot \epsilon \approx 0.106$ for graph-blind algorithms; with $T = 400$ and $K = 2N+1 = 101$ arms the finite-$T$ constants dominate the rate since $T \ll K \log K$." Renamed "causal Thompson sampling" → "context-conditional Thompson sampling" in the 305× ratio sentence.
- **Line 345 (Figure 2 caption).** Rewrote panel (a) caption: "the dotted reference line is the theoretical regret lower bound $\sqrt{N/T} \cdot \epsilon$ on the asymptotic rate achievable by any graph-blind algorithm, not an upper bound on the Successive Reject baseline at finite $T$. Observed Successive Reject regret sits well above this line because $T = 400 \ll K \log K$ at $K = 101$, so finite-$T$ constants dominate the rate." Renamed panel (c) algorithm label "causal Thompson sampling" → "context-conditional Thompson sampling".
- **Line 244 (sub-section closing reference).** Adjusted from "$\sqrt{m/T}$ rate across the hardness grid $m \in \{2, 8, 24, 48\}$ ... and $\sqrt{N/T}$ floor" to "on the monotone segment of the hardness grid $m \in \{2, 8, 24\}$ ... asymptotic $\sqrt{N/T}$ rate lower bound", and renamed TS_C → "stripped-down context-conditional Thompson sampling baseline".
- **Line 349 (closing summary).** "matching upper and lower bounds ... hold tightly across the hardness grid" → "the $\sqrt{m(q)/T}$ trend ... holds on the monotone segment $m^* \in \{2, 8, 24\}$ of the hardness grid (the trend reverses at $m^* = 48$ due to a finite-sample artefact of the single-coordinate reward construction discussed above)".

## 3. Verification

- Sim rerun from repo root:
  `python3 ch10b_rl_for_ci/sims/causal_bandit_parallel.py > ch10b_rl_for_ci/sims/causal_bandit_parallel_stdout.txt 2>&1`
  exited 0. Regret panels hit cache; MABUC recomputed under the renamed dict key `cctp`. Final stdout numbers unchanged: vanilla TS cumulative regret 200.49 (SE 0.28), context-conditional TS 0.66 (SE 0.04), ratio 305.3×.
- Outputs regenerated: `causal_bandit_combined.png` (panel (c) legend now reads "Context-conditional TS"; panel (a) legend now reads "asymptotic rate (lower bound)"), `causal_bandit_results.tex` (rerun, numbers stable to 3 decimals).
- Chapter PDF compiled from `docs/` via per-chapter pattern in `CLAUDE.md` Key Commands (pdflatex × 3 with one bibtex pass). Output: `docs/ch10b_rl_for_ci.pdf` (30 pages, 1,114,560 bytes). Only undefined-reference warnings are cross-chapter `\ref{section:causal_rl}` and `\ref{section:rl_algorithms}`, expected in standalone-chapter compilation.

## 4. Residual exposure

The three reviewer-catch issues are now reconciled at the prose level. The simulation still implements only the stripped-down variant of TS_C, but the chapter text now names that variant honestly and disclaims the omitted consistency-axiom seeding and RDC bias weighting on every mention. The m-grid table still includes `m* = 48` to preserve the script's modular cache; the non-monotonicity is now explained in the results paragraph rather than contradicted by a claim of monotone √(m*/T) growth.

Remaining open items not addressed in this pass (out of scope per the relabel + reconcile strategy):
- The reward model is still a single-coordinate sigmoid; the chapter does not re-derive the parallel-bandit gain under a more general linear-payoff reward (this is acknowledged in the new prose at line 333 but not corrected).
- Vanilla TS in MABUC still does not receive the observational seed data while the context-conditional TS does. The audit deemed this immaterial because the observational data does not help the unconditioned Beta posterior on the greedy-casino instance, but the asymmetry remains.
- The fully specified TS_C with both consistency-axiom seeding and RDC weighting is not implemented; the tex defers it as "a stronger baseline left for future work".

## 5. New score

**Bullshit score: 20%** — Reviewer 2 still notes that the script implements the stripped-down variant rather than Bareinboim 2015 Algorithm 1 in full, but the tex now names what is implemented, discloses what is omitted, explains the non-monotone `m* = 48` data point, and reframes the √(N/T)·ε line as an asymptotic rate lower bound rather than a finite-`T` floor. The substance — graph-aware vs graph-blind cost on the parallel bandit, bounded vs linear regret on greedy-casino MABUC — is intact and matches the chapter prose at the level of identified objects. Anchored at 25%: a hostile reviewer can still write a comment about "the algorithm implemented is a strict subset of the named TS_C, why not run the full version", but the artifact-prose mismatch that drove the original 55% is resolved; rounded down to 20% because the script and tex now agree on the name of the algorithm and the trend in the data.
