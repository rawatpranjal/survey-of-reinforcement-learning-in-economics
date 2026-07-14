# Audit: ch09_rlhf/sims/job_search_preference_learning.py

**Date:** 2026-07-14
**Type:** DELTA (subject audited 2026-05-19 at 15%, polished 2026-05-20; promoted to
delta because the committed stdout's last commit (2026-03-23) predates the .py's
last commit (2026-05-19)).

**Delta summary.** The only change to the script since the artifact-producing commit
(`4cdbac6`, 2026-03-23) is `git diff 4cdbac6 HEAD` on this file: (1) swap
`os.environ['PYTHONUNBUFFERED']='1'` for `sys.stdout.reconfigure(line_buffering=True)`
and move `import sys` up; (2) a 10-line explanatory *comment* inside
`extract_reward_table` (lines 439-448). No RNG seed, hyperparameter, environment,
algorithm, or output-path change. The change is **behaviorally inert**: it alters
stdout flushing and adds a comment, nothing that can move a single number. The
committed `.tex` tables and `_stdout.txt` are byte-identical between `4cdbac6` and
HEAD (`git diff` empty). Because the delta cannot change computation, the published
March numbers are exactly what current code reproduces. The data-integrity worry that
triggered the promotion resolves in the affirmative: **published numbers reflect
current code.**

**Files read (end to end):**
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_preference_learning.py` (1195 lines)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_preference_learning_stdout.txt` (130 lines)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_results.tex`
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_diagnostics.tex`
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_horizon.tex`
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_env.png` (viewed)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_sample_complexity.png` (viewed)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_horizon.png` (viewed)
- `/Users/pranjal/Code/rl/ch09_rlhf/tex/rlhf.tex` (Section "Simulation Study: Preference Learning in Job Search", lines 143-197)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_rlhf.py` (sibling, output-collision check)
- `/Users/pranjal/Code/rl/scripts/run_all_sims.py` (registry order)
- `/Users/pranjal/Code/rl/audits/ch09_rlhf__job_search_preference_learning_2026-05-19.md`
- `/Users/pranjal/Code/rl/audits/ch09_rlhf__job_search_preference_learning_polish_2026-05-20.md`

---

## Step 3 — thesis statement (what this sim is evidence for)

(i) **Theoretical claim the chapter advances.** RLHF/preference learning in a dynamic
economic model is best cast as a two-stage estimator: learn a reward (a static
Bradley-Terry / discrete-choice estimation problem) then solve the MDP by dynamic
programming, which exploits the *known transition model*. Direct preference
optimization (DPO) collapses these two stages and forfeits the transition structure,
so it cannot propagate value into states the behavioral policy under-covers. Model
specification governs sample efficiency: a correct one-parameter structural model is
near-optimal at tiny K, a flexible neural reward needs more comparisons, and a
misspecified reward plateaus below optimal no matter how much data arrives.

(ii) **What this sim is used FOR.** A concrete McCall-style job-search MDP with
compensating differentials where the truth is known, used to demonstrate, on identical
per-seed data, the method ordering Correct structural > NN RLHF > DPO > Misspecified,
and to show the DPO plateau (~95%) and the misspecified plateau (~91%) as the
empirical face of the identification/welfare-aggregation argument.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS (carried forward; delta does not touch it)

Algorithm identity, environment fidelity, and theory-consistency were established in
the 2026-05-19 audit and are unaffected by the inert delta. Spot re-verified:

- Bradley-Terry reward MLE for the NN (`train_reward_net`, line 430:
  `-logsigmoid(r_w - r_l)` over discount-weighted segment sums) and structural
  (`fit_structural`, line 471) match the BT logit loss.
- DPO (`train_dpo`, line 537) is the tabular-softmax BT-on-log-ratios form with a
  uniform reference that cancels via the same-state pairing
  (`generate_comparisons_same_state`, line 371). Correct.
- Teacher uses the true reward only to *label* pairs (`generate_segment`, line 329);
  learners see states/actions/labels only. No leakage.
- Results are theory-consistent: DP MC 74.305±0.312 vs analytic VI 74.129 (within ~2
  SE, gap is the 200-step MC truncation); NN gap shrinks ~15x over a 200x K range
  (≈√K); misspecified plateaus at 91%; DPO plateaus at ~95%. All consistent with the
  chapter's argument.

The documented `extract_reward_table` action-averaging (state-only reward handed to
VI though the net is r(s,a)) is harmless because `TRUE_REWARD_VEC` is
action-independent (lines 144-152); the delta's new comment and the tex footnote now
disclose it. No correctness regression.

### (b) PRESENTATION / NUMBERS — PASS WITH DEFECTS

Every headline number in the prose traces to a committed artifact **except the
online/offline ablation footnote** (see Findings 1-2). Cross-check of traceable
numbers, all exact:

| Prose claim (rlhf.tex) | Artifact | Match |
|---|---|---|
| V*(s0)=74.13 | stdout 74.1290 / results.tex 74.13 | yes |
| accepts 25 / stays 25 of 56 | stdout "Accepts 25", "Stays at 25"; env.png counts = 25 / 25 | yes |
| ρ=-0.74 | stdout -0.740 | yes |
| Correct 99.9% at K=25 | stdout K=25 Correct 99.9% | yes |
| NN reaches 99.9% by K=5000 | stdout K=5000 NN 99.9% | yes |
| DPO plateaus ~95% by K=500 | stdout K=500 DPO 95.2% | yes |
| Misspecified plateaus 91% | stdout ~91.0-91.1% | yes |
| Diagnostics 100% / 96% agree | diagnostics.tex 100.0 / 96.4 | yes |
| DPO 57% agree, Vcorr 0.78 | diagnostics.tex 57.1 / 0.777 | yes |
| DPO mean amenity 3.4 vs misspec 3.0 | diagnostics.tex 3.38 / 3.00 | yes |
| **online 73.92±0.05, offline 73.99±0.03, p=0.09** | **absent from committed stdout (truncated)** | **NO artifact** |

Figures render correctly and match the tables (sample_complexity.png shows six method
series with the DPO hump/decline; horizon.png shows the L=1 dip then plateau; env.png
shows the diagonal accept/stay boundary with 25 accepts). The on-disk
sample_complexity.png is this script's 6-method version (no title), not the sibling's
5-method version, confirming correct provenance.

### (c) CHAPTER FIT — PASS

The sim directly demonstrates the Step-3 thesis: the four-method ordering, the DPO
plateau tied to policy coverage, and the misspecification plateau are exactly the
"computational version of the identification point" the chapter's closing paragraph
draws. The `job_search_diagnostics.tex` table and `job_search_horizon.png` figure are
produced only by this script and are wired into the subsection.

### (d) EFFICIENCY / STANDARDS — PARTIAL

- Seeds: 30 main, 20 ablation (both ≥10), SEs reported throughout. Good.
- Output path: `OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))` (line 87) —
  writes to its own `ch09_rlhf/sims/`. No ch07 misdirection. Good.
- Caching: script does not use `sim_cache.py` and writes no `.pkl`; it is a long
  (~10 min) uncached monolith. Pre-existing, not introduced by the delta.
- Stdout standard ("capture ALL console output"): **violated** — the committed stdout
  is truncated mid-Experiment-3 (Finding 1).
- Output-filename hygiene: **collision** with sibling `job_search_rlhf.py` on two
  chapter-included PNGs (Finding 3).

---

## 7-point checklist

1. **Algorithm identity** — PASS. BT reward MLE (NN + structural) and tabular DPO
   match their defining equations; verified term-by-term at lines 430, 471, 537-592.
2. **Environment / MDP fidelity** — PASS. 8x7=56 wz pairs, 112 states, α=0.6, b=28,
   layoff 0.05, γ=0.95, ρ=-0.74; all match rlhf.tex and stdout.
3. **Data integrity (MAX WEIGHT for delta)** — PASS on computation, FLAG on artifacts.
   The delta is inert (buffering + comment), so current code reproduces the published
   March numbers; committed `.tex`/`.png` are consistent with the (truncated) stdout
   on every overlapping value. BUT the stdout is a truncated replication log and one
   prose number (online/offline ablation) has no committed artifact (Findings 1-2).
4. **Comparison fairness** — PASS. All four preference methods consume the same
   per-seed cross-state batch (line 827); DPO gets the same-state variant at
   `rng_seed+500000` (line 831), justified by the LLM same-prompt setup. Any DPO
   informativeness edge works against the narrative, not for it.
5. **Theoretical sanity** — PASS. MC≈VI; NN rate ≈√K; misspecified/DPO plateaus land
   where theory predicts; no method beats the oracle.
6. **Information leakage** — PASS. True reward only labels pairs; learners never see it.
7. **Seed & reproducibility** — PASS. MASTER_SEED=42, np+torch seeded per seed, 30/20
   seeds, SEs everywhere. Determinism confirmed indirectly: the March-16 `.tex` run and
   the March-17 stdout run agree to 3 decimals on every overlapping figure.

---

## Prior-audit open-item disposition

Source audit (2026-05-19, 15%) and polish (2026-05-20, 15%):

1. **`extract_reward_table` action-averaging** — RESOLVED (documented, not eliminated).
   The delta added the promised code comment (lines 439-448) and the tex footnote
   (rlhf.tex line ~169). Harmless as established. No further action.

2. **Truncated `_stdout.txt`** — STILL OPEN. The committed file is 130 lines ending at
   "L = 15." (`wc -l` = 130; `tail` confirms). It is missing the rest of Experiment 3,
   the MAIN RESULTS TABLE, PARAMETER RECOVERY, VERIFICATION, and the ONLINE-VS-OFFLINE
   ablation (`rg` for those headers returns none). The polish pass explicitly
   *chose* to restore this file from HEAD rather than regenerate, and — regression in
   the audit trail — the polish doc describes it as "130-line, fully clean" / "130
   lines, complete" (polish audit lines 42, 84, 89). It is not complete; it is
   truncated. The 2026-05-19 audit correctly called it "visibly truncated
   mid-Experiment-3." So the defect persists and the polish note misstates its status.

3. **Tex wiring** — the polish "all four artifacts OK" table (polish lines 105-110) is
   correct as far as it goes, but it MISSED that two of those four
   (`job_search_env.png`, `job_search_sample_complexity.png`) are also written by the
   sibling `job_search_rlhf.py` (Finding 3). Newly surfaced here.

---

## Findings (severity-ordered)

**Finding 1 (medium) — Committed stdout is a truncated replication log.**
`job_search_preference_learning_stdout.txt` ends at line 130 ("L = 15.") and omits the
entire back half of the run: the completion of Experiment 3, the main results table,
parameter recovery, the verification block, and the online-vs-offline ablation.
Evidence: `wc -l` = 130; `rg -c "ONLINE VS OFFLINE|MAIN RESULTS TABLE|VERIFICATION|
PARAMETER RECOVERY"` returns 0. Violates the repo standard that each script's
`_stdout.txt` capture all console output. Known since 2026-05-19; the polish pass
deliberately left it and mislabeled it complete. Fix is a single clean re-run
(`python3 -u ...py > ...stdout.txt 2>&1`, ~10 min); with the delta's new line-buffering
even a future kill will truncate cleanly at the last completed line.

**Finding 2 (medium) — One published prose number has no committed artifact.**
rlhf.tex states the online/offline NN ablation as "online 73.92 ± 0.05, offline
73.99 ± 0.03 (p = 0.09)." These are produced only by the ablation block (lines
1160-1190), which is beyond the truncation point of the committed stdout, so they
appear in no saved artifact on disk. They are plausible (both near the K=1000 NN value
73.95) and regeneratable given determinism, but as shipped they violate "every
published number traces to a generated artifact." Fixing Finding 1 fixes this.

**Finding 3 (medium-low) — Sibling filename collision on two chapter-included PNGs.**
`job_search_rlhf.py` (a superseded 5-method version: DP, QL, NN, Correct, Misspecified;
no DPO) writes `job_search_env.png` (line 620) and `job_search_sample_complexity.png`
(line 979) — the same names this script writes (lines 717, 1125) and that the chapter
includes (rlhf.tex 152, 161). The sibling's sample-complexity figure lacks the DPO
series and carries a title, which would contradict the chapter caption "all six
methods." Currently benign only because `run_all_sims.py` runs `job_search_rlhf.py`
(registry line 76) before `job_search_preference_learning.py` (line 78), so this
script's 6-method figure overwrites and survives on disk (confirmed: the viewed
sample_complexity.png has six series and no title). A targeted re-run of the sibling,
or any registry reorder, would silently swap in the wrong figure. Recommend removing
`job_search_rlhf.py`/`job_search_dpo.py` from the registry (they appear superseded by
this script; rlhf.py's docstring still says "Chapter 8") or renaming their outputs.

**Finding 4 (cosmetic) — Figure legend not in rank order.**
`sample_complexity.png` legend order is NN RLHF, DPO, Correct, Misspecified; the
project convention is rank-by-performance (Correct > NN > DPO > Misspecified). Does not
affect any number. Note only.

---

**Bullshit score: 25%** — The delta is inert and the substance holds (correct BT/DPO/VI, no leakage, theory-consistent, 30 seeds with SEs), but a hostile reviewer writes real snark: the committed replication log is truncated to half a run, one prose number (the online/offline ablation) is backed by no on-disk artifact, and a superseded sibling script silently shares two of the chapter's figure filenames. Housekeeping/traceability, not method-identity; substance survives revision. Raised from the prior 15% because the untraceable footnote number and the filename collision are genuine defects neither prior pass caught, and the polish note misstated the truncated stdout as complete.
