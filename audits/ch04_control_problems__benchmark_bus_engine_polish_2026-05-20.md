# Polish Report: ch04_control_problems/sims/benchmark_bus_engine.py

**Date:** 2026-05-20
**Prior audit:** `ch04_control_problems__benchmark_bus_engine_2026-05-19.md` (Bullshit score 25%)
**Scope:** Five remaining nicks from the 25% audit, all addressed.

---

## What was changed

### 1. Three to ten DQN seeds, reported as mean +/- SE

`SEEDS` extended from `[42, 123, 7]` to ten values `[42, 123, 7, 11, 19, 23, 31, 47, 61, 83]`.
The script now reports `dqn_reward_se = sample_std / sqrt(n)` (with `ddof=1`) rather than
the population std. The LaTeX table, the figure error bars, and the stdout summary all
use SE. The figure legend reads `DQN (mean +/- SE)` and the figure caption states
"DQN error bars are mean +/- standard error over ten seeds."

Per-seed determinism: at N=1 and N=2 all ten seeds return the identical discounted value
(SE = 0.000), because the small-fleet problem is fully solved by DQN. SE grows with N
(0.006 at N=3, 0.022 at N=5, 1.855 at N=6), which is the expected behaviour.

### 2. Documented discount-factor change (gamma = 0.95 vs Rust's 0.9999)

Added a sentence in the main text of `applications.tex` plus a footnote: "I use gamma = 0.95
rather than Rust's monthly beta = 0.9999, which softens the effective discount horizon from
roughly 10,000 periods to roughly 20 and stabilises DQN training; the qualitative DQN-vs-DP
comparison is unaffected." The footnote adds the numerical reason (Rust's beta produces
Q-values of order 1e4 that destabilise small-replay-buffer gradient updates).

### 3. Fixed RNG-consumption asymmetry (paired evaluation)

`evaluate_dp_policy`, `evaluate_dqn_policy`, and `evaluate_heuristic` in `econ_benchmark.py`
gained an `initial_states` parameter. A helper `_start_episode` sets `env.state` directly
to a pre-sampled tuple instead of calling `env.reset()` (which draws from the global RNG).

`run_single_complexity` now samples `EVAL_EPISODES` initial states once per fleet size,
from a dedicated `np.random.RandomState(EVAL_INIT_SEED + N)`, and feeds the identical set
to DP, DQN, and both heuristics. The comparison is now strictly paired: every method is
evaluated on the same 200 episode starts. The tex notes this: "All policies are evaluated
on the same pre-sampled set of initial states so that the comparison is paired across
methods."

This explains why the DQN-vs-DP gap is now exactly 0.0% at N=1..5 (vs the audit's reported
0.0-0.4%): the residual gap in the old code was evaluation noise from different episode
draws, not a true policy gap.

### 4. Corrected the "DP infeasible at N=6" claim

The code skips DP at N=6 via a state-count threshold (`DP_FEASIBLE_THRESHOLD = 10,000`),
not because value iteration numerically failed. The tex now says: "At N=6 (46,656 states),
I omit DP rather than run it: plain-Python value iteration is not numerically infeasible at
this size, but a single backup sweep costs |S|^2 |A| in Python-loop time and exceeds the
wall-clock budget by an order of magnitude." The figure caption changed from "DP is
infeasible (no data point)" to "DP is omitted because plain-Python value iteration becomes
wall-clock prohibitive at this scale." The stdout DP-skip message was also corrected to
state it is a wall-clock heuristic, not a numerical infeasibility.

(The redundant `or env.num_states <= 10_000` clause in the VI guard was removed; the guard
is now `if env.dp_feasible`, which is the same threshold, stated once.)

### 5. Disclosed that the scaling claim is Python-loop-specific

Added a footnote: "The DP wall-clock curve in the left panel reflects this Python-loop
implementation; a vectorised backup using numpy broadcasts would shrink the absolute
timings by one to two orders of magnitude. The scaling exponent (slope of the log-time
curve in N) is the right asymptotic shape, but the absolute timings are
implementation-dependent."

---

## Other changes

- `CONFIG['version']` bumped 1 -> 2 and `eval_init_seed` added, so the stale 3-seed cache
  is invalidated automatically.
- DQN episode budgets reduced (N=1: 3000->1000, ..., N=6: 12000->4000) to keep the
  10-seed sweep tractable on a loaded CPU. The reduction is documented in a code comment;
  DQN still converges to within 0.0% of DP at N=1..5, so the headline claim is unaffected.
  This is disclosed as an honest budget choice, not hidden.
- Added a parameter header to `print_detailed_results` so the stdout is self-documenting
  (fleet sizes, gamma, seeds, horizons, paired-eval seed).
- The stdout file is now written once (by the `capture_stdout` tee); the prior CLAUDE.md
  shell-redirect command double-tees with `capture_stdout` and produced a duplicated file.
  Regenerated via `--plots-only` without a shell redirect.

---

## Verification

- Full 10-seed sweep ran to completion (N=1..6, all 60 DQN trainings). Final stdout:
  `ch04_control_problems/sims/benchmark_bus_engine_stdout.txt`.
- Headline result holds: DQN matches DP at 0.0% gap for N=1..5 under paired evaluation.
  At N=6 (DP omitted) DQN returns -334.65 +/- 1.86, beating Threshold(3) at -336.69.
- Q-error / policy-agreement: agreement 100% (N=1) down to 81.1% (N=5), reward gap stays
  at 0.0% throughout, consistent with policy-relevant states being a small subset.
- VI converged at all feasible N (residuals ~1e-8).
- Chapter PDF recompiled: `docs/ch04_control_problems.pdf`, 12 pages, exit 0. Only an
  expected undefined cross-reference to `section:language` (a different chapter in
  single-chapter compilation).

## Residual reviewer-2 surface

- DQN episode budgets were reduced from the pilot. A reviewer could ask whether the
  smaller budget still represents "DQN given adequate compute." The 0.0% gap at N=1..5 and
  the disclosed comment answer this, but the choice is visible.
- The stdout from `--plots-only` carries the results tables and parameter header but not
  the per-N VI-iteration / per-seed-timing trace (that trace only exists during a forced
  recompute). The substantive numbers are all present.
- N=6 DQN SE (1.86) is larger than at N=5 because one seed (83) returned -350.46 vs the
  pack near -332; honestly reported, not smoothed.

**Bullshit score: 12%** — A hostile reviewer notes the DQN episode budget was trimmed for
wall-clock and that the `--plots-only` stdout lacks the live training trace, but the
paired evaluation, ten-seed SE, documented discount swap, and corrected DP-infeasibility
claim remove every substantive objection from the prior audit. The headline (DQN tracks
the DP oracle to 0.0% on a combinatorial fleet problem) is now computed on identical
trajectories and survives a second read.
