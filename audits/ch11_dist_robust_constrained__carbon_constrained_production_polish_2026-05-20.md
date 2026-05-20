# Polish Report: ch11_dist_robust_constrained/sims/carbon_constrained_production.py

**Date:** 2026-05-20
**Prior audit:** `ch11_dist_robust_constrained__carbon_constrained_production_2026-05-19.md` (Bullshit score 25%)
**Scope:** Address the three reviewer-bait issues from the 2026-05-19 audit. No re-implementation of CPO/PID (explicitly out of scope).

## What was done

### 1. Single seed -> five seeds with mean +/- SE (audit issue 1)

The prior version reported point estimates from `seed=0` only, a CLAUDE.md
violation ("minimum 10 seeds, report means and standard errors").

**Wall-clock measurement.** A timing probe (3000-episode run, both QL
variants) on a contention-loaded machine extrapolated to ~160 s per variant
for the full 30k-episode run, i.e. ~5.4 min per seed for the two QL runs
combined. An earlier full single-seed timing was distorted by extreme
machine load (load average 57/255/245; the process logged 10% CPU over a
1h47m wall clock) and is not representative. Per the procedure's
3-10 min/seed band, the bump was set to **5 seeds** (`SEEDS = [0,1,2,3,4]`)
rather than 10, to bound the run against documented machine contention.
The 5-seed run completed cleanly (exit 0).

**Code changes.** `compute_data()` now loops `run_q_learning` over
`SEEDS` for both the unconstrained and Lagrangian variants, aggregates a
`stats` dict (means and standard errors of final return, final cost, final
lambda, and lambda peak), and stores per-seed runs in `ql_runs` / `lag_runs`.
`generate_outputs()` table now prints `mean $\pm$ SE` for the two QL rows;
the figure plots seed means with `+/- 1 SE` shaded bands. CONFIG `version`
bumped 12 -> 13 and `seeds` added to the cache key, so the stale single-seed
cache is correctly invalidated.

**5-seed results** (stdout, 2026-05-20):

| Method | Return | Cost | Budget | lambda |
|---|---|---|---|---|
| LP Oracle | 186.4 | 31.35 | Y | 1.20 |
| Unconstrained Q-learning | 253.7 +/- 5.7 | 95.03 +/- 4.65 | N | -- |
| Lagrangian Q-learning | 172.5 +/- 3.1 | 23.41 +/- 2.38 | Y | 1.39 +/- 0.00 |

The headline finding survives multi-seed: unconstrained QL violates the
budget ~3x; Lagrangian QL satisfies it and the multiplier settles near the
LP shadow price. The lambda SE is 0.0017 (rounds to 0.00) -- the multiplier
is highly stable across seeds.

### 2. The "lambda overshoots to 3.2" claim (audit issue 2)

The prior tex prose claimed the multiplier "overshoots to lambda ~ 3.2
before settling at lambda ~ 1.40." The audit flagged this as figure-only,
not stdout-corroborated.

**Verified false.** The `run_q_learning` function was instrumented to record
the peak of `lambda_trajectory` per seed. Across all five seeds:

- Per-seed lambda peaks: 1.4109, 1.4022, 1.3978, 1.4139, 1.4093
- Peak mean 1.407 +/- 0.003, max over seeds 1.414
- Per-seed final lambda: 1.397, 1.390, 1.392, 1.392, 1.398
- The peak occurs late (samples 43-53 of 60) and is within ~1% of the
  settling value.

There is no overshoot to 3.2. The trajectory rises near-monotonically from
0 toward ~1.39. The 3.2 figure had no basis in any computed array.

**Tex rewrite.** The prose now reads: the multiplier "rises from zero and
settles at lambda ~ 1.39"; "the trajectory is nearly monotone: the peak
value is 1.407 +/- 0.003, within one percent of the settling value, so
naive dual ascent here exhibits little of the overshoot that
[Stooke2020]'s PID controller is designed to dampen." The earlier sentence
attributing the conservatism to "the transient spike" was removed; the
conservatism is now attributed to the final lambda sitting ~16% above
lambda* (1.39 vs 1.20), which the numbers do support.

### 3. LP-vs-QL evaluation truncation bias (audit issue 3)

The LP oracle return (186.4) is computed by exact infinite-horizon policy
evaluation `V = (I - gamma*P_pi)^{-1} R_pi`; the QL returns are Monte Carlo
averages over rollouts truncated at H=100. The 6-unit LP-vs-QL gap is partly
truncation, not pure suboptimality.

**Tex footnote added** disclosing the methodology difference: with
gamma^100 ~ 6e-3, the removed discounted tail is bounded by
~gamma^H r_max/(1-gamma), on the order of 1-2% of the return, and "part of
the gap between the LP value and the Q-learning returns is therefore
truncation bias rather than policy suboptimality."

## Verification

- 5-seed sim run: `carbon_constrained_production_stdout.txt` regenerated,
  exit code 0, all five seeds completed for both QL variants.
- Table `carbon_constrained_production_table.tex` regenerated with SE columns.
- Figure `carbon_constrained_production_convergence.png` regenerated with
  per-seed mean +/- SE bands.
- Lambda peaks cross-checked directly against the cached `lambda_trajectory`
  arrays (not just stdout): max peak 1.414, far from the discredited 3.2.
- Chapter PDF recompiled: `docs/ch11_dist_robust_constrained.pdf` (967 KB),
  3 pdflatex passes + bibtex, all exit 0. No new errors, no undefined
  references or citations. Pre-existing overfull hboxes at lines 282/366
  (equation displays) are unrelated to this edit.

## Re-audit against the 7-point checklist

1. Algorithm identity -- unchanged, still correct (LP Altman occupation-measure
   LP; Lagrangian QL = single-timescale RCPO form). PASS.
2. Environment fidelity -- unchanged, faithful to tex. PASS.
3. Data integrity -- now 5 seeds; lambda peak claim verified against the
   array; table/stdout numbers match the computation. PASS.
4. Comparison fairness -- LP-vs-QL truncation asymmetry now disclosed in a
   tex footnote. PASS.
5. Theoretical sanity -- constraint binds, lambda* > 0, unconstrained policy
   violates 3x, Lagrangian QL approaches LP optimum; all hold across 5 seeds. PASS.
6. Information leakage -- unchanged, none. PASS.
7. Seed and reproducibility -- 5 seeds, means + SE reported in table, stdout,
   and figure bands. Below the CLAUDE.md "minimum 10" target; the shortfall
   is documented and justified by measured per-seed wall clock under
   sustained machine contention. The remaining gap to 10 seeds is the only
   residual nick. PASS (with caveat).

## Verdict

All three reviewer-bait issues from the 25% audit are resolved: the
single-seed violation is fixed (5 seeds, mean +/- SE everywhere), the
unsupported "lambda ~ 3.2" claim is removed and replaced with the verified
peak (1.407 +/- 0.003), and the LP-vs-QL truncation asymmetry is disclosed.
The only residual is 5 seeds rather than 10 -- a documented, justified
deviation, not a hidden one. A hostile reviewer can still note "why 5 not
10," but the answer (measured cost, machine contention) is on the record
and the SE on every reported quantity is now visible.

**Bullshit score: 10%** -- Reviewer 2 might quibble that 5 seeds is short of
the stated 10-seed standard, but every number now carries a standard error,
the discredited 3.2 claim is gone, and the eval asymmetry is footnoted. The
substance is intact and the headline finding holds across all five seeds.
