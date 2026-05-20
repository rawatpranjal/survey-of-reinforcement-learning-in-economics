# dtr_qlearning_vs_murphy — dqn_hd Paired-Seed Re-run

**Date:** 2026-05-20

## Problem

The 2026-05-20 polish pass edited `dtr_qlearning_vs_murphy.py` so all estimators
use paired cohort seeds, but the high-dimensional DQN component cache
(`cache/dtr_qlearning_vs_murphy__dqn_hd.pkl`) was from 2026-05-12, generated under
the OLD unpaired DQN seed scheme. The figure caption claimed "paired across
estimators" while the on-disk panel-3 (Q3) DQN data was actually computed unpaired.

Note: the component config hash is unchanged (`f3306781d3f4c8ee4913e6a7d6429baa`)
because `DQN_HD_CONFIG` does not encode the seed-stream offset — the seed *values*
changed inside `run_dqn_hd_sweep` (line 513: `np.random.default_rng(N * 100 + s)`,
matching NN-FQI's cohort seed at line 500), not the hashed config. The stale cache
would therefore NOT have been auto-invalidated; an explicit force-recompute was
required.

## Command used

The script's per-component force flag is `--algo <component>` (via
`add_component_args` / `parse_force_set` in `sims/sim_cache.py`), NOT `--force`.

```bash
cd /Users/pranjal/Code/rl && python3 ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py --algo dqn_hd > ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt 2>&1
```

Exit code 0. Stdout confirms `forcing recompute of: ['dqn_hd']`, `Computing: dqn_hd`,
`Cache saved`; the other six components hit cache. Figure + table then regenerated
from the now-consistent caches:

```bash
python3 ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py --plots-only
```

- `dqn_hd` cache refreshed: `cache/dtr_qlearning_vs_murphy__dqn_hd.pkl`, mtime 2026-05-20 02:05.
- Outputs regenerated: `dtr_qlearning_vs_murphy.png` and `dtr_qlearning_vs_murphy_results.tex` (mtime 2026-05-20 02:06), `_stdout.txt`.

## Old vs new Q3 numbers (high-dim DQN panel)

High-dim Oracle V* = 0.7857 (behavior policy 0.3350). FQI 200 full-batch epochs,
DQN 8000 minibatch steps, 20 seeds. NN-FQI was already paired-correct (unchanged).

| N | OLD DQN mean (SE) — unpaired | NEW DQN mean (SE) — paired | NN-FQI (SE) |
|---|---|---|---|
| 500 | 0.7994 (0.0046) | 0.7936 (0.0043) | 0.8764 (0.0046) |
| 2000 | 0.8335 (0.0043) | 0.8326 (0.0032) | 0.9187 (0.0033) |
| 5000 | 0.9126 (0.0030) | 0.9100 (0.0018) | 0.9310 (0.0024) |

Raw (un-normalized) DQN V means: old `[0.6281, 0.6549, 0.7171]` →
new `[0.6236, 0.6542, 0.7151]`. Shifts are within Monte Carlo noise
(≤ 0.006 normalized). Standard errors tightened at every N (N=5000: 0.0030 →
0.0018), the expected effect of pairing the DQN cohort with the NN-FQI cohort.
Qualitative conclusion unchanged: NN-FQI ahead at small N, gap closes as N grows.

Results table `dtr_qlearning_vs_murphy_results.tex`, `DQN (high-dim, N=5000)` row:
`0.9126 (0.0030)` → `0.9100 (0.0018)`. Rank order unchanged.

## Caption status

`ch10b_rl_for_ci/tex/rl_for_ci.tex`:

- Figure caption (line 62): "(Q3) High-dimensional sweep ... 20 seeds, paired
  across estimators" — now ACCURATE. The on-disk `dqn_hd` cache is computed under
  cohort seed `N*100+s`, identical to NN-FQI's.
- Table caption (line 68): "High-dimensional setting ... 20 Monte Carlo seeds,
  paired across the two estimators" — now ACCURATE.

No caption edit needed. The high-dim panel CAN be paired (DGP and cohort
generation are identical for both estimators); the only obstacle was the stale
cache, now resolved.

## Verification

- Chapter PDF recompiled: `docs/ch10b_rl_for_ci.pdf`, 32 pages, 1,144,658 bytes,
  3 pdflatex passes + bibtex, exit 0, mtime 2026-05-20 02:06. Only warnings are
  the expected cross-chapter undefined references (`section:causal_rl`,
  `section:rl_algorithms`) that resolve in the full `main.tex` build.
- The open item flagged in `ch10b_rl_for_ci__dtr_qlearning_vs_murphy_polish_2026-05-20.md`
  (nick F1, stale dqn_hd cache) is now closed.

## Files touched

- `ch10b_rl_for_ci/sims/cache/dtr_qlearning_vs_murphy__dqn_hd.pkl` — re-run, mtime 2026-05-20 02:05
- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.png` — regenerated, mtime 2026-05-20 02:06
- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex` — regenerated, mtime 2026-05-20 02:06
- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt` — refreshed, shows the dqn_hd run
- `docs/ch10b_rl_for_ci.pdf` — recompiled, 32 pages, mtime 2026-05-20 02:06

No source `.py` or `.tex` edits were required — the source was already correct;
only the stale cache and its downstream artifacts needed refreshing.
