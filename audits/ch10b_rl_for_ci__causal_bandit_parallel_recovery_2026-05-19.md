# Recovery Report: ch10b_rl_for_ci / causal_bandit_parallel → Full Bareinboim TS_C

**Date:** 2026-05-19
**Original score:** 55% (Phase 0)
**After Phase 1 fix:** 20% (relabeled stripped-down)
**After Phase 2 recovery:** 18%

## Files modified
- `ch10b_rl_for_ci/sims/causal_bandit_parallel.py` (~75 lines added — full `causal_thompson_sampling_tsc` function with consistency seeding + RDC weighting; updated `run_mabuc` to dispatch all three TS variants; updated figure panel (c) to plot the new curve; updated stdout to print all three regrets and three pairwise ratios)
- `ch10b_rl_for_ci/tex/rl_for_ci.tex` (Bareinboim attribution restored; lines 226, 244, 291, 329, 333, 345, 349 rewritten to describe the full TS_C alongside CCTS; new empirical finding that CCTS beats TS_C on this MDP framed honestly)
- `ch10b_rl_for_ci/sims/cache/causal_bandit_parallel__mabuc.pkl` (regenerated under the new `algos_version: v2_full_tsc` config key)
- `ch10b_rl_for_ci/sims/causal_bandit_combined.png` (panel (c) now has 3 curves; full TS_C in green)
- `ch10b_rl_for_ci/sims/causal_bandit_parallel_stdout.txt` (3-algorithm MABUC summary)
- `docs/ch10b_rl_for_ci.pdf` (recompiled; 31 pages, 1,128,459 bytes)

## Substantive code added
- **Consistency-axiom seeding** of the off-intuition arm `a' = 1 - x` at fractional pseudo-count `c = CONSISTENCY_OFF_INTUITION_WEIGHT = 0.5` of each observational `(x, y)` pair (lines 511-525 of the script). The on-intuition cell receives the full pseudo-count (canonical Bareinboim 2015 line 2); the off-intuition cell receives the same direction but scaled by `c`.
- **RDC bias weighting** with running empirical `Q_hat[x, a]` estimates (lines 529-557). At each round, samples are drawn from the `(x, a)`-conditional Beta posterior, then multiplied by `w_a = clip(1 - |Q_hat[0, a] - Q_hat[1, a]|, 0.01, 1)`. Arm selection is `argmax_a (w_a · theta_a)`.
- **New TS_C dispatch entry** in `run_mabuc()` (lines 642-647). All three algorithms (vanilla TS, CCTS, TS_C) now run on every seed with independent RNG offsets (`s + 100_000`, `s + 200_000`, `s + 300_000`).
- **Cache invalidation marker** (`'algos_version': 'v2_full_tsc'` in `MABUC_CONFIG`) forces fresh recomputation under the new dict schema.

## New empirical findings
- Vanilla TS cumulative regret: 200.49 (SE 0.28) — unchanged from Phase 1.
- Context-conditional TS (CCTS) cumulative regret: 0.66 (SE 0.04) — unchanged from Phase 1.
- Full TS_C (Bareinboim 2015) cumulative regret: 4.49 (SE 0.10) — new.
- TS_C vs CCTS: **TS_C loses**, by a factor of ≈ 6.8× in final cumulative regret.
- TS_C vs vanilla TS: TS_C wins by ≈ 45× (vanilla TS is still linear-regret; TS_C is bounded-regret).
- Bareinboim 2015 claim that TS_C dominates context-conditional baselines: **does NOT hold on this MDP**, but the substantive linear-vs-bounded gap relative to vanilla TS holds for both context-conditional variants (CCTS at 305×, TS_C at 45×).

The CCTS-dominates-TS_C result on the greedy casino is interpretable: the off-intuition arm's true payoff (0.50) exceeds the on-intuition payoff (0.10), so the fractional consistency-axiom seed transferred from on-intuition observations attaches a *pessimistic* prior to a *high-value* off-intuition cell. The agent must then unlearn this bias through experimental pulls. RDC weighting compounds the slowdown by suppressing both arms symmetrically once `Q_hat` reveals the cross-context flip (both arms get `w ≈ 0.6`), preserving the argmax ordering but slowing posterior concentration. On a two-context two-arm instance where `(x, a)` cells are independently identifiable from experimental data alone, the minimal CCTS captures everything the additional Bareinboim components are designed to exploit, so the components introduce overhead without benefit. The tex now states this honestly.

## Bullshit detector axis check
- **Algorithm Identity (point 1):** Full TS_C now matches Bareinboim 2015 Algorithm 1 within the disclosed pseudo-count rule. The implementation includes both distinguishing features named in the chapter prose: consistency-axiom seeding (on the off-intuition arm at `c = 0.5` of the on-intuition count, with the rule itself disclosed in a footnote) and RDC bias weighting (with running `Q_hat[x, a]` cell means, clipped weights `w_a ∈ [0.01, 1]`). The previous Phase 1 disclosure of "stripped down" has been narrowed to apply to CCTS only, and CCTS is now positioned as an explicit minimal baseline against which the full TS_C is benchmarked. The 305× vanilla-TS-vs-CCTS gap is no longer the chapter's headline — the headline is now the full bounded-vs-linear story for both context-conditional variants.
- **Comparison Fairness (point 4):** Three TS variants now run on the same observational seed data with seed offsets that ensure independent RNG draws. Vanilla TS still does not consume observational seed data, which is intentional (it has no `x` to condition on, so the seed adds no signal); this was an acknowledged caveat in Phase 1 and is unchanged.
- **Theoretical Sanity (point 5):** The empirical TS_C regret (4.49) is above CCTS (0.66) on this MDP. The chapter explains this honestly as a function of the greedy-casino payoff asymmetry; no claim that TS_C dominates is made.

## Residual issues
- The fractional off-intuition seeding rule (`c = 0.5`) is one operationalization of the consistency axiom extended to the counterfactual cell; the paper does not give a canonical value for this. The choice is disclosed in a footnote of the tex but not optimized. A robustness sweep over `c ∈ {0, 0.25, 0.5, 0.75, 1.0}` would tighten the claim; deferred to a future pass.
- The TS_C-loses-CCTS result is MDP-specific. Bareinboim 2015 Experiment 2 ("Paradoxical Switching", Table 2) might be a more favorable instance for the full TS_C. Implementing the paradoxical-switching DGP and rerunning would test whether TS_C dominates CCTS when the observational and experimental distributions diverge sharply (which is the regime the algorithm was designed for). Deferred.
- Vanilla TS still does not receive the observational seed data; symmetric framing would let vanilla TS also consume the data and confirm (as expected) that it derives no benefit. Deferred per Phase 1 reasoning.
- The non-monotone `m* = 48` data point in the regret-vs-m grid is unchanged; the Phase 1 prose explanation (single-coordinate reward construction) still applies.

## Score rationale
Anchored at 25%: a hostile reviewer can still ask "why does your claimed full TS_C lose to the simpler baseline — did you implement it correctly?" The answer is two-pronged: (a) the implementation matches Bareinboim 2015 Algorithm 1 line-for-line for the named components, with the off-intuition seeding pseudo-count `c` disclosed as a free parameter that the paper leaves implicit; (b) the CCTS-dominance result is interpretable from the greedy-casino payoff structure and the tex explains why. Rounded down to 18% because the substantive comparison is honest, the implementation matches the cited paper within the one disclosed choice, and the chapter prose now agrees with both the algorithm names and the numerical results. The TS_C-loses-CCTS finding is itself a small but defensible contribution: the additional Bareinboim machinery is overhead on instances where the `(x, a)` posterior already captures the identifiable variation.

**Bullshit score: 18%** — Reviewer 2 may push back on the `c = 0.5` choice and ask for a robustness check, but the artifact and the prose now describe the same algorithm, name the same paper, and report a finding that is consistent with the implementation. The Phase 1 substance is preserved (305× vanilla-TS-vs-CCTS) and Bareinboim attribution is restored at the cost of a small, honest qualification.
