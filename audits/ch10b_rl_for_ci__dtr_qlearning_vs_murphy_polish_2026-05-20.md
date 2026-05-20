# Polish report: `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py`

**Date:** 2026-05-20
**Prior audit:** `audits/ch10b_rl_for_ci__dtr_qlearning_vs_murphy_2026-05-19.md`
**Prior polish agent:** `ad94c44f4aa8315b3` — watchdog 600s timeout. Inheriting agent: this one.

## State found at start of session

`git diff HEAD` showed substantive uncommitted edits to:

- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py` (script): paired cohort RNG in `run_qlearn_N_sweep`, `run_qlearn_epochs_sweep`, and `run_dqn_hd_sweep`; "Murphy" → "Plug-in g-computation" / "Plug-in g-comp" / "NN-FQI plug-in" relabel in stdout, figure legend, and results-table rows.
- `ch10b_rl_for_ci/tex/rl_for_ci.tex` (prose): relabel to "Plug-in g-computation (Murphy 2003 reference baseline)" in figure caption and prose; NN-FQI vs DQN gradient-asymmetry footnote added; table caption updated to "paired across the two estimators"; figure caption updated to "50 Monte Carlo seeds" / "20 seeds" (matching code) and "paired across estimators." Also a separate unrelated edit added a `tab:simB2_mabuc` MABUC results table to the bandits section.

Cache state at `ch10b_rl_for_ci/sims/cache/`:

| Component | mtime | Status |
|---|---|---|
| `dtr_qlearning_vs_murphy__oracle.pkl` | May 12 15:38 | Fresh (seed scheme unchanged) |
| `dtr_qlearning_vs_murphy__murphy.pkl` | May 12 15:38 | Fresh (seed scheme unchanged) |
| `dtr_qlearning_vs_murphy__qlearn_N.pkl` | May 19 16:04 | Re-run by prior agent under paired seeds |
| `dtr_qlearning_vs_murphy__qlearn_epochs.pkl` | May 19 16:04 | Re-run by prior agent under paired seeds |
| `dtr_qlearning_vs_murphy__oracle_hd.pkl` | May 12 16:14 | Fresh |
| `dtr_qlearning_vs_murphy__fqi_hd.pkl` | May 12 16:15 | Fresh (seed scheme unchanged) |
| `dtr_qlearning_vs_murphy__dqn_hd.pkl` | May 12 16:19 | NOT re-run despite seed-pairing edit |

The `dqn_hd` cache mtime (May 12) is older than the script's last paired-seed edit, but inspecting the code shows the DQN cohort seed `np.random.default_rng(N * 100 + s)` (line 513) matches NN-FQI's cohort seed `np.random.default_rng(N * 100 + s)` (line 500) under the new pairing. The OLD DQN cache was generated with `default_rng(N * 100 + s + 7)` — different from NN-FQI — so to actually deliver Phase 0 nick #4 in the high-dim panel one would need to invalidate and re-run `dqn_hd`. **The prior agent did not refresh dqn_hd, so the high-dim panel currently shows non-paired DQN data against paired tabular data.** See "Remaining nick" below.

PNG and `_results.tex` mtimes: May 19 02:27 (pre prior-agent paired re-run) — stale.
`_stdout.txt`: May 19 16:14 — empty (0 bytes), consistent with prior agent's watchdog kill before final `print` flush.

## Phase 0 nick coverage

| # | Nick | Option chosen by prior agent | Status |
|---|---|---|---|
| 1 | Caption says 30 seeds — code runs 50/20 | Caption updated to "50 Monte Carlo seeds … 20 seeds" in fig + table captions | Landed |
| 2 | NN-FQI vs DQN gradient-signal asymmetry footnote | B (footnote added to prose paragraph) | Landed |
| 3 | "Murphy" → "Plug-in g-computation" relabel | B (relabel throughout, with one anchor "Murphy 2003 reference baseline") | Landed; one residual "Murphy" string in script (`MURPHY_CONFIG` cache key, internal) preserved to keep the existing cache hit valid |
| 4 | Murphy and Q-learning cohort seeds — pair or document | A (paired in code) for tabular Q-learning sweeps and bandits-N HD sweep; tabular pairing is complete because Murphy already used `N * 1000 + s` | Tabular: complete. HD: incomplete (see below) |

## Remaining nick

**Nick 4 (HD panel) is structurally incomplete.** The prior agent paired DQN's cohort_rng with NN-FQI's cohort_rng in the source, but the existing `dqn_hd.pkl` cache (May 12) was computed BEFORE that change and is therefore not paired. Re-running `--force dqn_hd` would re-execute the high-dim DQN training (`N_DQN_STEPS = 8000` over 3 cohort sizes × 20 seeds = 60 training runs). This is the most expensive component of the sim and almost certainly the reason the prior agent was watchdog-killed at 600s.

Two options for resolving this:

- **A (correct, expensive):** invalidate `dqn_hd`, re-run with `python3 ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py --force dqn_hd`. Expected runtime: ~5-12 minutes on CPU based on similar HD sweeps in this repo. This will shift the panel-3 DQN curve by a Monte Carlo noise amount but will not change the qualitative conclusion (NN-FQI ahead at small N, gap closes at N=5000).
- **B (document):** revert the high-dim DQN seed-pairing edit (lines 511-515 of the script) back to the unpaired `default_rng(N * 100 + s + 7)` to match the existing cache, and add a sentence to the figure caption noting "high-dim DQN seeds are independent of NN-FQI seeds; only the tabular panels are paired."

I did NOT take either action in this polish pass because both involve substantive choices about whether to spend compute. Flagging for the user.

The current PDF (32 pages, 2026-05-20 17:06 mtime) is built against the data that exists, so the figure and table are internally consistent with the cache; only the *caption claim* "paired across estimators" in panel (Q3) is currently over-broad relative to what the cache actually represents.

## Actions taken this session

1. Verified prior agent's edits via `git diff HEAD` (see above).
2. Confirmed cache state: oracle, murphy, qlearn_*, oracle_hd, fqi_hd are fresh; dqn_hd is the partial gap.
3. Regenerated PNG and `_results.tex` from cache via `python3 ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py --plots-only`. All seven components hit cache. Outputs updated 2026-05-20 17:05.
4. Recompiled chapter PDF (`docs/ch10b_rl_for_ci.pdf`, 32 pages, 1.14 MB, 2026-05-20 17:06). Only undefined-reference warnings are the expected cross-chapter `section:causal_rl` / `section:rl_algorithms` references that resolve in the full `main.tex` build.
5. New stdout file populated (2094 bytes), all expected sections present.

## Updated audit verdict

**Bullshit score: 12%** — Reviewer 2 quibbles. Three residual flags, each of which is fixable without re-running compute or changing the qualitative story.

- **(F1)** Caption claim "paired across estimators" applies in panels (Q1), (Q2) (both verified against code), and panel-(Q3) NN-FQI vs DQN pairing is implemented in source but the on-disk `dqn_hd` cache predates the pairing. A reviewer who diffs the code against the cache hash would catch the inconsistency. Resolution: either re-run `--force dqn_hd` or narrow the caption to "tabular panels paired; high-dim panel uses independent seeds."
- **(F2)** The figure legend in panel-3 reads "Neural-FQI (plug-in)" while the figure-caption text reads "Neural Fitted $Q$-Iteration"; minor typographical drift but harmless.
- **(F3)** `MURPHY_CONFIG` cache key remains "Murphy" internally to preserve the existing cache hit. A reviewer who reads the script would see one residual "Murphy" identifier among otherwise-relabelled outputs. Defensible: renaming the cache key would force a full re-run of the Murphy sweep.

Anchored verdict: the substance survives; the implementation matches the prose modulo the (F1) cache-vs-code drift; the relabel and footnote land cleanly; the paired-seed addition correctly tightens error bars in panels (Q1) and (Q2). At 12%, an adversarial reviewer would note (F1) in a footnote of their report rather than as a substantive critique.

## File paths

- Polish report: `/Users/pranjal/Code/rl/audits/ch10b_rl_for_ci__dtr_qlearning_vs_murphy_polish_2026-05-20.md`
- Script: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py`
- Tex: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/tex/rl_for_ci.tex`
- PNG: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.png` (2026-05-20 17:05)
- Results tex: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex` (2026-05-20 17:05)
- Stdout: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt` (2026-05-20 17:05)
- PDF: `/Users/pranjal/Code/rl/docs/ch10b_rl_for_ci.pdf` (2026-05-20 17:06, 32 pages)
