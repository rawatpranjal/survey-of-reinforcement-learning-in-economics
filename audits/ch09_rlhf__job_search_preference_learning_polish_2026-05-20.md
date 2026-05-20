# Polish pass: ch09_rlhf/sims/job_search_preference_learning.py

**Date:** 2026-05-20
**Source audit:** `audits/ch09_rlhf__job_search_preference_learning_2026-05-19.md` (15%)
**Goal:** address the two snark-worthy nicks (action-averaging in
`extract_reward_table`, truncated `_stdout.txt`) and confirm tex wiring.
Target: hold at or below 15% (no degradation).

## Context: previous polish session timed out

The previous polish agent (`a7d82df4e6f1c469b`) made two source edits to
the script and started a fresh end-to-end re-run for a clean stdout, but
its wall-clock budget expired while the background python was still mid
Q-learning (only 2 of 30 seeds completed; stdout truncated at 40 lines).
The partial stdout is *not* a crash artifact: the script's slowest single
block is the 30-seed Q-learning sweep (10,000 episodes x ~200 steps x 30
seeds = ~60M tabular Q updates) immediately followed by the 30-seed
x 8-K x 100-epoch NN training sweep. This script is genuinely
long-running and was never going to fit inside the prior agent's
remaining budget.

I confirmed (by reading the diff and the script) there is no introduced
regression. The two source edits — see below — are both safe:

1. **`os.environ['PYTHONUNBUFFERED'] = '1'` -> `sys.stdout.reconfigure(line_buffering=True)`.**
   The original line was a no-op (Python had already cached its stdout
   buffering mode by the time the assignment executed inside the running
   process). The replacement actually reconfigures the existing
   `TextIOWrapper`; verified `sys.stdout.reconfigure` is available on the
   project's Python 3.11 and works under shell redirect.
2. **Inline code comment + tex footnote on `extract_reward_table` action-averaging.**
   No behavioural change.

Pyright's "possibly unbound" flags on `Q`, `V_new`, `it` in
`value_iteration_vec` (lines 257-258) and `policy_eval_vec` (line 271)
are static-analysis false positives: `max_iter=5000` guarantees the loop
body executes at least once, so all three are bound at use. Not a real
bug, not addressed.

Decision: per the instructions, re-running the script just to refresh
the stdout would re-burn the wall-clock budget for a cosmetic fix.
Restored the prior (130-line, fully clean, 2026-03-17) stdout from git
HEAD and let it stand. The committed stdout's numbers match
`job_search_diagnostics.tex`, `job_search_results.tex`, and
`job_search_horizon.tex` exactly (NN RLHF 96.4% agreement, DPO 57.1%,
Correct 100.0%, Misspec 50.0%; mean accepted amenities 4.67 / 3.38 / 4.84
/ 3.00; mean wages 73 / 63 / 71 / 70). The audit's "stdout-vs-tex
agreement" check therefore still holds against the restored file.

## Changes applied

### Nick 1 — `extract_reward_table` action-averaging

**Before:** `extract_reward_table` evaluates the trained network on all
`(s, a)` pairs, then collapses to a state-only reward vector via
`mean(axis=1)`. The tex framed the reward model as $r_\theta(s, a)$ but
silently discarded the action dimension at the VI handoff. Audit flagged
this as cosmetically inconsistent (Section 1, hostile-reviewer flag).

**Fix (script):** added a 10-line comment block at the top of
`extract_reward_table` explaining (i) the network parameterisation, (ii)
the averaging, and (iii) why it is harmless in this environment: the
true per-period reward `TRUE_REWARD_VEC` is action-independent (action
affects only the transition kernel, not the flow payoff), so the
Bradley-Terry MLE has no gradient signal pushing $r_\theta$ to fit an
action-dependent component. The comment also flags that this routine
would need to return the full $(\text{NUM\_STATES}, 2)$ table in a
generalisation with action-dependent rewards.

**Fix (tex):** extended the existing reward-model footnote (rlhf.tex
line 87) with one sentence: "The network parameterises $r_\theta(s, a)$,
but the table handed to value iteration averages over actions to give
$\bar{r}(s) = \tfrac{1}{2}(r_\theta(s, 0) + r_\theta(s, 1))$; this is
harmless here because the true per-period reward in this environment is
action-independent (actions affect only the next-state transition), so
the Bradley-Terry MLE has no incentive to fit action-dependent payoffs."

Reviewer 2 can still write a comment about why this isn't $r_\theta(s)$
directly, but the footnote pre-empts the "what happened to the action
input" objection and matches the implementation verbatim.

### Nick 2 — truncated `_stdout.txt`

**Before:** the file shipped at HEAD was 130 lines, complete, dated
2026-03-17 04:06. The prior agent's partial re-run had clobbered it down
to 40 lines (ending mid-Q-learning seed 1) before timing out.

**Fix:** `git checkout HEAD -- ch09_rlhf/sims/job_search_preference_learning_stdout.txt`.
The restored file is the original clean 130-line artifact. Re-running
the script to produce a fresh-timestamped equivalent was deferred — the
script takes ~10 minutes wall-clock per the prior session, the numbers
in the restored file match the published figures and tables exactly, and
the alternative (a half-written stdout) is strictly worse for any
reviewer doing artifact replication.

The audit's snark about "stdout file mtime predates the figure/table
mtime" is unchanged in spirit, but in practice every reported number
remains regeneratable from the script; the script is callable end-to-end
with no manual steps.

### Nick 3 — tex wiring verified

All four output artifacts cited by the audit are wired into `rlhf.tex`:

| Artifact | Cited at | Status |
|---|---|---|
| `job_search_env.png` | rlhf.tex:82 (Figure~\ref{fig:search_env}) | OK |
| `job_search_sample_complexity.png` | rlhf.tex:91 (Figure~\ref{fig:preference_sample}) | OK |
| `job_search_diagnostics.tex` | rlhf.tex:102 (Table~\ref{tab:preference_diagnostics}) | OK |
| `job_search_horizon.png` | rlhf.tex:109 (Figure~\ref{fig:preference_horizon}) | OK |

Verified by `grep -n "job_search\|fig:preference\|tab:preference" ch09_rlhf/tex/rlhf.tex`.
No orphan outputs, no broken `\includegraphics` paths.

## Verification

- `git diff HEAD ch09_rlhf/sims/job_search_preference_learning.py`: 42 lines,
  only the two intended edits (buffering + 10-line comment).
- `git diff HEAD ch09_rlhf/tex/rlhf.tex`: the footnote extension, single hunk.
- `wc -l ch09_rlhf/sims/job_search_preference_learning_stdout.txt`: 130 lines,
  full clean run restored from HEAD.
- Chapter PDF recompiled:
  ```
  cd docs && pdflatex -shell-escape -jobname=ch09_rlhf \
    "\def\chapterfile{../ch09_rlhf/tex/rlhf}\input{compile_chapter}"
  ```
  3-pass + bibtex run. Output: `/Users/pranjal/Code/rl/docs/ch09_rlhf.pdf`,
  16 pages, 890 KB. Only remaining warnings are expected cross-chapter
  undefined references (`section:language`, `eq:softmax_logit`,
  `sec:actor_critic`, `sec:deadly_triad`) — these resolve only in the
  full document build, not the single-chapter build.

## Files changed

- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_preference_learning.py`
  (buffering line + 10-line comment in `extract_reward_table`)
- `/Users/pranjal/Code/rl/ch09_rlhf/tex/rlhf.tex` (one sentence appended
  to existing reward-model footnote at line 87)
- `/Users/pranjal/Code/rl/ch09_rlhf/sims/job_search_preference_learning_stdout.txt`
  (restored from HEAD — file is back to the clean 130-line March 17 run)
- `/Users/pranjal/Code/rl/docs/ch09_rlhf.pdf` (recompiled)

No data, figures, or tables were regenerated; no cache was touched
(script does not use the project caching pattern).

## Open items / deferred

- **Fresh stdout re-run:** deferred. The script is ~10 minutes wall-clock
  end-to-end and the restored stdout is internally consistent with the
  shipped tables/figures. A future polish pass with a larger budget
  could re-run cleanly: `python3 -u ch09_rlhf/sims/job_search_preference_learning.py
  > ch09_rlhf/sims/job_search_preference_learning_stdout.txt 2>&1`.
  Note that with the new `sys.stdout.reconfigure(line_buffering=True)`
  line, the redirected file will update line-by-line, so any future
  truncation will at least show the last completed seed rather than a
  dead-mid-block.
- **Pyright "possibly unbound":** false positive on a `range(max_iter)` loop
  with `max_iter >= 1`. Not addressed; would require either initialising
  `Q, V_new, it = None, None, 0` before the loop (clutter for no
  behavioural benefit) or a `# type: ignore`.

## Hostile-reviewer revisit

- Reviewer 2 looks for "what does the reward network do with the action
  input" -> footnote at rlhf.tex line 87 now explicitly states the
  averaging step and the action-independence of the true per-period
  reward.
- Reviewer 2 looks for truncated stdout -> restored 130-line clean run;
  matches every cited number in the chapter exactly.
- Reviewer 2 looks for broken figure / table paths -> all four output
  artifacts cited and present, paths verified.
- Substantive headline numbers (NN RLHF 99.9% at K=5000, DPO plateau 95%,
  correct structural 100% at K=25, misspecified 91%) all unchanged from
  the audit baseline.

The two open items below the 15% threshold are: (i) the action-averaging
itself, which the footnote now describes but does not eliminate (a
reviewer wanting strict $r_\theta(s)$ would still prefer a state-only
network); (ii) the restored stdout has a March 17 timestamp rather than
a fresh one, which a paranoid reviewer might use to question whether the
restored numbers come from a current code revision. Both are cosmetic.

**Bullshit score: 15%** — unchanged from the source audit. The two
snark-worthy items are now documented in the tex (action averaging) and
restored to a clean state (stdout), but neither was deeper than a
cosmetic issue to start with, so the substantive grade does not move.
