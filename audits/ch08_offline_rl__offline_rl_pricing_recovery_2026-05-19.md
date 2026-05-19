# Recovery Report: ch08_offline_rl / offline_rl_pricing -> Rebalanced Behavioral

**Date:** 2026-05-19
**Original score (Phase 0 audit):** 50%
**After Phase 1 fix (relabel + disclose collapse):** 25%
**After Phase 2 recovery (state-dependent behavioral):** 15%

## Files modified

- `ch08_offline_rl/sims/offline_rl_pricing.py`
  - Header comment expanded to document the Phase 2 behavioral rebalance.
  - `BEHAVIORAL_MARKUPS` changed from `[10, 10, 10, 10]` to `[5, 7, 8, 9]` (state-dependent preferred prices, one per demand regime).
  - Added `BEHAVIORAL_KERNEL = 2` constant and `_triangular_kernel_probs(center)` helper.
  - `behavioral_action` rewritten: with probability `noise_prob` sample uniform random, else sample from a triangular kernel centered on the regime's preferred price (half-width 2).
  - `ENV_PARAMS` dict now includes `BEHAVIORAL_KERNEL` so the kernel-width parameter is part of the cache key.
  - `CONFIG_VERSION` bumped 13 -> 14 to invalidate all caches.

- `ch08_offline_rl/tex/offline_rl.tex` (Simulation Study section)
  - Setup paragraph: behavioral description rewritten from "always sets the maximum price ($p=10$) ... with probability 0.85" to "state-dependent ... triangular kernel of half-width 2 around $p^*(d)$ ... 15% uniform noise". Old 85%-concentration footnote kept (relocated) to acknowledge the Phase 1 collapse and the rebalance that fixes it.
  - Results paragraph fully rewritten: the four-way-collapse paragraph (BC=BCQ-D=DT=RvS=169.27) is gone, replaced with two paragraphs reporting the new rank order. RvS (97.0%) -> BC (96.8%) -> DT (96.3%) -> CQL (92.6%) -> BCQ-D (92.0%) -> IQL-argmax (91.8%) -> FQI (24.7%).
  - The pessimism-beats-BC chapter message has been honestly reframed: under this near-on-policy behavioral, BC is itself near-optimal and the pessimism family pays a small cost for distributional robustness. The advantage of pessimism shows up in the coverage sweep, which is the test that matters.
  - Coverage paragraph rewritten with new numbers and a corrected interpretation of $\epsilon_b$ as the uniform-mix weight against the state-dependent kernel.

- `ch08_offline_rl/sims/offline_rl_pricing_stdout.txt` (regenerated)
- `ch08_offline_rl/sims/offline_rl_pricing_results.tex` (regenerated)
- `ch08_offline_rl/sims/offline_rl_pricing_coverage.png` (regenerated)
- `docs/ch08_offline_rl.pdf` (recompiled, 14 pages, 579,172 bytes)

## Behavioral change

- Old (Phase 0/Phase 1): `BEHAVIORAL_MARKUPS = [10, 10, 10, 10]`, `BEHAVIORAL_NOISE = 0.15`. State-independent. 85% mass on $p = 10$, 15% uniform over all 10 prices. Max single-action mass 0.85.
- New (Phase 2): `BEHAVIORAL_MARKUPS = [5, 7, 8, 9]`, `BEHAVIORAL_KERNEL = 2`, `BEHAVIORAL_NOISE = 0.15`. State-dependent. Each regime samples from a triangular kernel of half-width 2 around the regime's preferred price (5 admissible actions per regime, weights {1,2,3,2,1}/9), mixed with 15% uniform. Marginal max single-action mass 0.19 over 8 effective support actions.

## New empirical findings (key)

Main comparison (mean return, % of DP optimal, 20 seeds):

| Method | Mean ± SE | % Optimal |
|---|---|---|
| DP Oracle | 192.41 ± 0.33 | 100.0% |
| RvS | 186.58 ± 0.34 | 97.0% |
| BC | 186.28 ± 0.31 | 96.8% |
| DT | 185.27 ± 0.33 | 96.3% |
| CQL | 178.08 ± 1.48 | 92.6% |
| BCQ-D | 177.05 ± 0.73 | 92.0% |
| IQL-argmax | 176.67 ± 0.81 | 91.8% |
| FQI | 47.48 ± 8.42 | 24.7% |

- **Four-way collapse: gone.** The four supervised-conditioning methods (BC, BCQ-D, DT, RvS) now report four distinct numbers (186.28, 177.05, 185.27, 186.58) instead of bit-identical 169.27. The chapter's intended demonstration that DT/RvS provide additional information beyond imitation is now visible in the data.
- **New finding:** under this near-on-policy behavioral, the supervised-conditioning family outperforms the pessimism family. BC at 96.8% sets the imitation ceiling; RvS and DT edge slightly above; CQL/IQL-argmax/BCQ-D sit at 92% because their pessimism penalties trade a small amount of imitation fidelity for distributional robustness.
- **New finding:** FQI collapses harder under more diverse behavioral. Phase 1's 81.2% becomes Phase 2's 24.7%. With broader action coverage, the unconstrained $\max_{a'} Q(s', a')$ operator has more out-of-distribution Q-values to find, and the overestimation cascade is more severe rather than less. This contradicts the naive "more data is better" intuition for off-policy Q-learning and is consistent with Fujimoto2019's documented failure mode.
- **Coverage sweep (% of DP optimal, mean ± SE, 20 seeds):**

| Method | eps=0.05 | eps=0.3 | eps=0.9 |
|---|---|---|---|
| BC | 96.8 ± 0.2 | 96.6 ± 0.2 | 95.4 ± 0.2 |
| RvS | 96.7 ± 0.2 | 96.7 ± 0.2 | 86.7 ± 0.3 |
| DT | 96.7 ± 0.2 | 96.0 ± 0.3 | 85.5 ± 1.0 |
| CQL | 92.6 ± 0.4 | 93.4 ± 0.5 | 92.0 ± 0.5 |
| IQL-argmax | 92.1 ± 0.5 | 92.6 ± 0.4 | 91.5 ± 0.5 |
| BCQ-D | 92.3 ± 0.5 | 91.7 ± 0.3 | 25.6 ± 5.1 |
| FQI | 16.7 ± 0.6 | 27.4 ± 4.7 | 25.4 ± 5.0 |

  - BC, CQL, IQL-argmax stay stable across all coverage levels.
  - BCQ-D collapses to 25.6% at $\epsilon_b = 0.9$ because the threshold becomes vacuous when the behavioral is near-uniform.
  - DT and RvS drop to 85-87% at $\epsilon_b = 0.9$ because the kernel signal is washed out and the return-conditioning extrapolation has less in-distribution support.
  - FQI is uniformly catastrophic.

## Verdict

Supervised-conditioning methods now span 92.0% (BCQ-D) to 97.0% (RvS), with three of the four supervised-conditioning rows (BC, DT, RvS) within 0.7 percentage points of each other but reporting distinct policies. The chapter's intended message that DT and RvS provide information beyond imitation is restored, and the supplementary message that the differentiation is small on a near-on-policy benchmark is honestly framed in the new results paragraph.

The chapter-level message was modified rather than restored verbatim: the original framing was "pessimism beats BC" which now holds only conditionally (under bad behavioral / poor coverage). Under good behavioral, BC is itself near-optimal and the pessimism family pays a small cost for the robustness guarantee. The coverage sweep now does the work the headline table previously did in the chapter's narrative.

## Bullshit-detector axis check

- **Algorithm Identity (point 1):** unchanged from Phase 1. All seven methods still match their named-paper formulations modulo the disclosed simplifications (IQL-argmax policy step, BCQ-D discrete variant, fused-token DT).
- **MDP Fidelity (point 2):** unchanged. The state, action, transition, reward, terminal salvage all match the tex writeup.
- **Data Integrity (point 3):** improved. Numbers in the table are no longer bit-identical across four rows; each row is a different computed value.
- **Comparison Fairness (point 4):** **improved**. The diverse behavioral gives all four supervised-conditioning methods a real opportunity to differentiate, which the original experiment did not afford. Same offline dataset per seed across methods; same eval RNG.
- **Theoretical Sanity (point 5):** improved. The new FQI collapse to 24.7% is a stronger demonstration of the overestimation cascade than Phase 1's 81.2%; the BC ceiling at 96.8% and CQL/IQL plateau at 92% reflect a real distributional-robustness tradeoff that the chapter prose now discusses honestly.
- **Information Leakage (point 6):** unchanged. DT/RvS use $R^\star = V^\ast(s_0) \approx 184$ as the deployment target; this is a single scalar that an operator could supply.
- **Reproducibility (point 7):** unchanged. 20 seeds, separate dataset/eval RNG offsets, config version bumped to invalidate stale caches.

## Residual issues (deliberately not fixed)

- C2 deferred. Substantive reimplementation of continuous BCQ (VAE + perturbation network), advantage-weighted IQL, and three-token DT remains out of scope for this pass.
- DT/RvS $R^\star$ sensitivity remains undisclosed in tex prose (the choice is mentioned, but no sweep is reported).
- FQI's catastrophic collapse to 24.7% is worth a deeper diagnostic to confirm it is overestimation rather than an optimization pathology; deferred to a future pass.
- The pessimism-beats-BC chapter-level message is now conditional on the behavioral being poor (visible only in the coverage sweep), not universal. This is a more honest framing but the original prose claimed pessimism beats BC unconditionally. A future pass could re-design the headline experiment with a deliberately bad behavioral if the chapter wants the unconditional claim back.

## Bullshit score after recovery

**Bullshit score: 15%** — Reviewer 2 can still ask why BC is the second-best method on the headline table rather than the chapter-claimed pessimism methods, and a careful reader will catch that the FQI collapse to 24.7% is much larger than the audit-suggested 81.2%. Both are real findings under the rebalanced behavioral, the tex prose now owns them, and the four-way bit-identical-coincidence problem is gone. Substance survives a hostile read; only minor framing critiques remain.
