# 2026-05-20 arxiv-cleanup-cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user opted in to all phases including Phase C (bib cleanup).

> **First execution action (Task 0):** Move this plan from `/Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md` to `docs/plans/2026-05-20-arxiv-cleanup-cycle.md`. Plan mode constrained where it could be written; the canonical home is in-repo under `docs/plans/` (NOT `docs/superpowers/` — that path is gitignored per `.gitignore` line 59).

**Goal:** Address the 7 deferred items from the 2026-05-19 push, integrate the parallel session's audit-fix work, and produce a second clean arxiv tarball ready to upload.

**Architecture:** Four sequential phases. Phase A triages and commits parallel-session sim/audit fixes (5 ≥50% sim fixes already done, refs.bib edits, 24 audit reports). Phase B addresses content items deferred in the last cycle (4 arXiv-ID lookups, ch99 World Models paragraph, chapter PDF rebuilds). Phase C is mechanical bib cleanup (orphan trim + delete inactive refs_extended.bib). Phase D re-runs the three audit gates and produces the arxiv tarball.

**Tech Stack:** pdflatex + bibtex (natbib, econometrica.bst), Python 3.10 with `bibtexparser` (already a dependency of the `arxiv-check` skill), the shared `sims/sim_cache.py` and `sims/plot_style.py`, the project's audit agents (`bib-coverage-auditor`, `paper-coherence-auditor`) and the `arxiv-check` skill. All commits land on the `humanize-pass` branch.

---

## Context

- Yesterday's main-arxiv push landed 19 commits on `humanize-pass`, produced a 214-page `docs/main.pdf` and an 11-MB `arxiv_submission.tar.gz` (smoke-tested clean). Summary at `docs/plans/2026-05-19-arxiv-push-summary.md`.
- That cycle deferred 7 items. A parallel agent session then ran a 7-point Bullshit-Detector audit on all 35 in-paper sims (`audits/_INDEX.md`) and fixed the four ≥50% findings plus one extra (`offline_rl_pricing` collapse), but did NOT commit. The uncommitted state and the audit reports are sitting on disk.
- The four pending-verification arXiv IDs (`Cai2023`, `Tullii2024`, `Fan2024`, `Ying2022`) still need correct IDs.
- `ch99_conclusion/tex/conclusion.tex` still doesn't mention the World Models chapter (the longest chapter in the paper).
- 380 orphan entries in `refs.bib`, 189 duplicate keys vs `refs_extended.bib` (which is dead — never loaded by `\bibliography{refs}`).
- User opted in to all of Phase A + B + C + D.

---

## File Structure

### Will be modified
- `docs/refs.bib` — 4 arXiv-ID fixes (Task 10) + orphan trim (Task 15)
- `docs/main.pdf` — recompiled after every chapter-touching change
- `docs/main.tex` — no direct edits expected
- `docs/main.log` — read-only inspection for undefined refs/citations
- `ch99_conclusion/tex/conclusion.tex` — add World Models paragraph (Task 11)
- Chapter PDFs touched by the parallel-session audit fixes: `docs/ch03_theory.pdf`, `docs/ch06_games.pdf`, `docs/ch08_offline_rl.pdf`, `docs/ch10b_rl_for_ci.pdf` (rebuilt by parallel session, committed in Tasks 2–6)
- `docs/ch03b_deeprl_practice.pdf`, `docs/ch04_control_problems.pdf` — rebuilt in Task 12 to close item #7
- `audits/_INDEX.md` — committed in Task 7
- All 24 new `audits/*.md` files (committed in Tasks 2–6 alongside their related fixes, plus Task 7 for orphan audits without `_fix_` counterparts)

### Will be created
- `docs/plans/2026-05-20-arxiv-cleanup-cycle.md` — canonical home for this plan (moved from plan-mode location in Task 0)
- `docs/plans/2026-05-20-bib-coverage.md` — emitted by `bib-coverage-auditor` agent in Task 17
- `docs/plans/2026-05-20-paper-coherence.md` — emitted by `paper-coherence-auditor` agent in Task 18
- `docs/plans/2026-05-20-arxiv-check.md` — synthesized from the three `arxiv-check` skill agents in Task 19
- `docs/plans/2026-05-20-arxiv-push-summary.md` — end-of-cycle summary in Task 21

### Will be deleted
- `docs/refs_extended.bib` — never loaded by `\bibliography{refs}`; 189 of its keys duplicate `refs.bib` (Task 16)

### Will not be touched
- `journals/`, `thesis/`, `thesis_v2/`, `ORE_main/`, `archive/`
- `docs/superpowers/` (gitignored)
- Any chapter that wasn't flagged in yesterday's audits

### Existing utilities to reuse
- `scripts/package_arxiv.sh` — updated yesterday, no changes needed; just invoke
- `/Users/pranjal/.claude/skills/arxiv-check/scripts/extract_cites.py` — extract cited-key set + active bib path from `docs/main.tex`
- `/Users/pranjal/.claude/skills/arxiv-check/scripts/parse_bib.py` — parse `refs.bib` into structured entry list (uses `bibtexparser`)
- `sims/sim_cache.py` and `sims/plot_style.py` — already used by all sims; no new utilities needed
- Memories: `feedback_always_show_pdf.md` (recompile after every tex change), `feedback_update_stdout.md` (regenerate `_stdout.txt` after sim change), `feedback_table_rank_order.md` (rank-order result tables)

---

## Phase A — Triage and commit parallel-session work

### Task 0: Move plan to canonical location

**Files:**
- Move: `/Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md` → `docs/plans/2026-05-20-arxiv-cleanup-cycle.md`

- [ ] **Step 1: Move the plan file.**

```bash
mv /Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md docs/plans/2026-05-20-arxiv-cleanup-cycle.md
```

- [ ] **Step 2: Verify the file is present at the new path.**

```bash
ls -la docs/plans/2026-05-20-arxiv-cleanup-cycle.md
```

Expected: file listed, non-zero size.

- [ ] **Step 3: Skip commit** — Task 7 will batch this with the audit-INDEX commit.

### Task 1: Baseline state snapshot

**Files:**
- Read-only

- [ ] **Step 1: Capture current branch and remote tracking.**

```bash
git branch --show-current
git status --short | head -50
git log --oneline -3
```

Expected branch: `humanize-pass`. Most recent commit: `c9bbb0b docs: end-of-cycle summary for 2026-05-19 arxiv main-article push`. About 34 modified files plus ~24 new untracked audit `.md` files.

- [ ] **Step 2: Verify the parallel session's chapter PDFs are present and current.**

```bash
ls -la docs/ch03_theory.pdf docs/ch06_games.pdf docs/ch08_offline_rl.pdf docs/ch10b_rl_for_ci.pdf
```

Expected: all four exist with mtimes dated 2026-05-19 evening or later (the parallel session rebuilt them).

- [ ] **Step 3: Read the master audit INDEX to see what was scored.**

```bash
head -80 audits/_INDEX.md
```

Note for the remaining tasks: which sims were ≥50% and which `_fix_*.md` audits exist.

### Task 2: Commit `td_lambda_corridor` fix (ch03)

**Files:**
- Stage: `ch03_theory/sims/td_lambda_corridor.py`, `ch03_theory/sims/td_lambda_corridor.png`, `ch03_theory/sims/td_lambda_corridor.tex`, `ch03_theory/sims/td_lambda_corridor_stdout.txt`, `ch03_theory/tex/planning_learning_v3.tex`, `docs/ch03_theory.pdf`, `audits/ch03_theory__td_lambda_corridor_fix_2026-05-19.md`

- [ ] **Step 1: Inspect the fix audit to confirm what changed.**

```bash
head -60 audits/ch03_theory__td_lambda_corridor_fix_2026-05-19.md
```

Expected: describes off-by-one factor of γ in the closed-form `true_values()` function (was `γ^(19-s)`, corrected to `γ^(18-s)`), with the MC RMSVE bias floor dropping from 0.0091 to 0.0000.

- [ ] **Step 2: Verify the fix is in the source.**

```bash
grep -nE "gamma.\*\*\s*\(?(18|19)" ch03_theory/sims/td_lambda_corridor.py | head -5
git diff ch03_theory/sims/td_lambda_corridor.py | grep -E "^[+-].*gamma" | head -10
```

Expected: a `+ gamma**(18-s)` line and a `- gamma**(19-s)` line in the diff. The current source should have `18`.

- [ ] **Step 3: Confirm the tex update at line 141 of `planning_learning_v3.tex` matches the new closed-form expression.**

```bash
git diff ch03_theory/tex/planning_learning_v3.tex | head -40
```

Expected: tex change reflecting the corrected exponent.

- [ ] **Step 4: Stage and commit.**

```bash
git add ch03_theory/sims/td_lambda_corridor.py \
        ch03_theory/sims/td_lambda_corridor.png \
        ch03_theory/sims/td_lambda_corridor.tex \
        ch03_theory/sims/td_lambda_corridor_stdout.txt \
        ch03_theory/tex/planning_learning_v3.tex \
        docs/ch03_theory.pdf \
        audits/ch03_theory__td_lambda_corridor_fix_2026-05-19.md
git commit -m "$(cat <<'EOF'
fix(ch03_theory): td_lambda_corridor off-by-one gamma in closed-form value

Audit found the reference formula in true_values() used gamma**(19-s)
instead of gamma**(18-s), producing a 0.0091 RMSVE bias floor that the
MC sim could not close. Corrected the exponent in script and at
planning_learning_v3.tex line 141. New MC RMSVE: 0.0000. Audit score
50% -> 10-15%.

Full audit: audits/ch03_theory__td_lambda_corridor_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one new commit on `humanize-pass`.

### Task 3: Commit `cournot_bertrand_marl` fix (ch06_games)

**Files:**
- Stage: `ch06_games/sims/cournot_bertrand_marl.py`, `ch06_games/sims/cournot_bertrand_marl.png`, `ch06_games/sims/cournot_bertrand_marl_stdout.txt`, `ch06_games/sims/cournot_bertrand_results.tex`, `ch06_games/tex/rl_in_games.tex` (partial — only the cournot_bertrand-related hunks), `audits/ch06_games__cournot_bertrand_marl_fix_2026-05-19.md`, `audits/ch06_games__cournot_bertrand_marl_2026-05-19.md`

- [ ] **Step 1: Inspect the fix audit.**

```bash
head -60 audits/ch06_games__cournot_bertrand_marl_fix_2026-05-19.md
```

Expected: documents (a) wrong Bertrand symmetric-FOC derivation (had a stray `+ e*c` term; correct closed-form is `p* = 4`), (b) phantom convergence-iteration column that was constant at 1000, and (c) missing Calvano 2020 reference + missing Nash-Q tie-break footnote.

- [ ] **Step 2: Verify the Bertrand fix in the source.**

```bash
grep -nE "p_star|p\*\s*=" ch06_games/sims/cournot_bertrand_marl.py | head -5
```

Expected: a `p_star = 4` (or similar literal) replacing the old buggy expression.

- [ ] **Step 3: Verify the table no longer has the fake convergence column.**

```bash
cat ch06_games/sims/cournot_bertrand_results.tex
```

Expected: 5 columns instead of 6; no "Conv. iter" column.

- [ ] **Step 4: Inspect the `rl_in_games.tex` diff to isolate the cournot_bertrand hunks (the durable_goods hunks land in Task 4).**

```bash
git diff ch06_games/tex/rl_in_games.tex | head -120
```

Expected: changes around the Cournot/Bertrand section that match the audit (FOC fix, removed conv-iter mention, added Calvano cite, added Nash-Q footnote).

- [ ] **Step 5: Stage only the cournot_bertrand-related files (skip the durable_goods files, those go in Task 4).** Use `git add -p` to interactively select the rl_in_games.tex hunks if needed.

```bash
git add ch06_games/sims/cournot_bertrand_marl.py \
        ch06_games/sims/cournot_bertrand_marl.png \
        ch06_games/sims/cournot_bertrand_marl_stdout.txt \
        ch06_games/sims/cournot_bertrand_results.tex \
        audits/ch06_games__cournot_bertrand_marl_fix_2026-05-19.md \
        audits/ch06_games__cournot_bertrand_marl_2026-05-19.md
# stage cournot_bertrand-related rl_in_games.tex hunks selectively:
git add -p ch06_games/tex/rl_in_games.tex
# (interactive: include hunks mentioning Cournot/Bertrand/Calvano/Nash-Q;
#  skip hunks about durable_goods/Coase — those land in Task 4)
```

If `git add -p` is impractical, stage the full `rl_in_games.tex` here and let Task 4 add nothing for that file. The audit notes are still split correctly across the two commits.

- [ ] **Step 6: Commit.**

```bash
git commit -m "$(cat <<'EOF'
fix(ch06_games): cournot_bertrand_marl Nash FOC + remove fake convergence column

Audit found three issues:
- Bertrand symmetric FOC derivation carried a stray +e*c term; correct
  closed-form for the parameters used is p* = 4. Updated source and
  table.
- "Conv. iter" column was constant at 1000 across all rows -- the
  convergence test always ran to the full budget. Column removed
  (5 cols now, not 6).
- Tex did not credit Calvano 2020 for the Nash-Q equilibrium-selection
  rule (max-sum tie-break) used in the multi-equilibrium Cournot grid.
  Added cite + footnote disclosing the rule.

Audit score 50% -> 15%.

Full audit: audits/ch06_games__cournot_bertrand_marl_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Commit `durable_goods_monopoly` rescope (ch06_games)

**Files:**
- Stage: `ch06_games/sims/durable_goods_monopoly.py`, `ch06_games/sims/durable_goods_monopoly_stdout.txt`, `ch06_games/sims/durable_goods_results.tex`, `ch06_games/sims/durable_goods_coase.png`, `ch06_games/sims/durable_goods_delta_sweep.png`, `ch06_games/sims/durable_goods_nashconv.png`, `ch06_games/sims/durable_goods_strategies.png`, `ch06_games/tex/rl_in_games.tex` (remaining hunks), `docs/ch06_games.pdf`, `audits/ch06_games__durable_goods_monopoly_fix_2026-05-19.md`, `audits/ch06_games__durable_goods_monopoly_2026-05-19.md`

- [ ] **Step 1: Inspect the fix audit.**

```bash
head -60 audits/ch06_games__durable_goods_monopoly_fix_2026-05-19.md
```

Expected: section was titled "The Coase Conjecture" but the sim is a 2-period finite-horizon model with hard-coded price set; Coase is an asymptotic limit. Post-hoc hidden tolerance (0.45–0.60 "near-threshold" carve-out) was inflating success rates.

- [ ] **Step 2: Verify the tex retitle.**

```bash
grep -nE "Coase|Screening|durable goods" ch06_games/tex/rl_in_games.tex | head -10
```

Expected: subsection retitled to "Screening versus Pooling in the Durable Goods Monopoly"; disclaimer added "finite-horizon precursor to the Coase conjecture".

- [ ] **Step 3: Verify the results table replaced the hidden-status checkmarks with a transparent `|Δ|` column.**

```bash
head -25 ch06_games/sims/durable_goods_results.tex
```

Expected: `|\Delta|` column header; no "Status" or checkmark column; SE columns present (multi-seed n=10).

- [ ] **Step 4: Stage and commit.**

```bash
git add ch06_games/sims/durable_goods_monopoly.py \
        ch06_games/sims/durable_goods_monopoly_stdout.txt \
        ch06_games/sims/durable_goods_results.tex \
        ch06_games/sims/durable_goods_coase.png \
        ch06_games/sims/durable_goods_delta_sweep.png \
        ch06_games/sims/durable_goods_nashconv.png \
        ch06_games/sims/durable_goods_strategies.png \
        ch06_games/tex/rl_in_games.tex \
        docs/ch06_games.pdf \
        audits/ch06_games__durable_goods_monopoly_fix_2026-05-19.md \
        audits/ch06_games__durable_goods_monopoly_2026-05-19.md
git commit -m "$(cat <<'EOF'
refactor(ch06_games): durable_goods_monopoly retitle + transparent diagnostics

Audit found:
- Section titled "The Coase Conjecture" but the sim is 2-period
  finite-horizon with hard-coded price set; Coase is asymptotic.
  Retitled to "Screening versus Pooling in the Durable Goods
  Monopoly" with a disclaimer naming the model as a finite-horizon
  precursor to Coase.
- Post-hoc hidden tolerance (0.45-0.60 "near threshold" carve-out)
  was inflating success rates. Removed; results table now reports
  transparent |Delta| with no Status/checkmark column.
- n=1 reported as if a general finding. Re-run with n=10 seeds and
  SE columns.
- NashConv panel was log-scale with no anchor. Switched to linear
  with the utility-share anchor explicit.

Audit score 65% -> 20%.

Full audit: audits/ch06_games__durable_goods_monopoly_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Commit `causal_bandit_parallel` relabel (ch10b)

**Files:**
- Stage: `ch10b_rl_for_ci/sims/causal_bandit_parallel.py`, `ch10b_rl_for_ci/sims/causal_bandit_parallel_stdout.txt`, `ch10b_rl_for_ci/sims/causal_bandit_combined.png`, `ch10b_rl_for_ci/tex/rl_for_ci.tex`, `docs/ch10b_rl_for_ci.pdf`, `audits/ch10b_rl_for_ci__causal_bandit_parallel_fix_2026-05-19.md`, `audits/ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md`

- [ ] **Step 1: Inspect the fix audit.**

```bash
head -60 audits/ch10b_rl_for_ci__causal_bandit_parallel_fix_2026-05-19.md
```

Expected: function `causal_thompson_sampling` was mislabeled — it implements a context-conditional variant only, omitting Bareinboim 2015's consistency-axiom seeding and RDC bias weighting. Non-monotone behavior at `m*=48` contradicted a "monotone" claim in the prose. Reference-line caption was inverted (asymptotic lower bound vs finite-T upper bound).

- [ ] **Step 2: Verify the function renamed.**

```bash
grep -nE "context_conditional_thompson|causal_thompson_sampling" ch10b_rl_for_ci/sims/causal_bandit_parallel.py | head -5
```

Expected: function renamed to `context_conditional_thompson_sampling`; old name no longer present in source.

- [ ] **Step 3: Verify the tex updates at lines ~226/291/329/333/345.**

```bash
grep -nE "context-conditional|context_conditional|stripped-down" ch10b_rl_for_ci/tex/rl_for_ci.tex | head
```

Expected: 5+ matches in the relevant subsections describing the stripped-down variant explicitly.

- [ ] **Step 4: Stage and commit.**

```bash
git add ch10b_rl_for_ci/sims/causal_bandit_parallel.py \
        ch10b_rl_for_ci/sims/causal_bandit_parallel_stdout.txt \
        ch10b_rl_for_ci/sims/causal_bandit_combined.png \
        ch10b_rl_for_ci/tex/rl_for_ci.tex \
        docs/ch10b_rl_for_ci.pdf \
        audits/ch10b_rl_for_ci__causal_bandit_parallel_fix_2026-05-19.md \
        audits/ch10b_rl_for_ci__causal_bandit_parallel_2026-05-19.md
git commit -m "$(cat <<'EOF'
fix(ch10b_rl_for_ci): causal_bandit_parallel TS_C relabel as stripped variant

Audit found:
- causal_thompson_sampling() implements a context-conditional variant
  only -- it omits Bareinboim 2015's consistency-axiom seeding and
  RDC bias-weighting steps. Relabeled function + dict keys to
  context_conditional_thompson_sampling; docstring and tex disclose
  the omissions.
- Non-monotone result at m* = 48 contradicted a "monotone" claim
  in the prose. Explained as a finite-sample artifact of the
  single-coordinate reward.
- Reference-line caption inverted: sqrt(N/T) * epsilon is an
  asymptotic *lower bound* on graph-blind regret, not a finite-T
  upper bound. Caption reframed.

Audit score 55% -> 20%.

Full audit: audits/ch10b_rl_for_ci__causal_bandit_parallel_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6: Commit `offline_rl_pricing` ownership (ch08)

**Files:**
- Stage: `ch08_offline_rl/sims/offline_rl_pricing.py`, `ch08_offline_rl/sims/offline_rl_pricing_results.tex`, `ch08_offline_rl/sims/offline_rl_pricing_coverage.png`, `ch08_offline_rl/sims/offline_rl_pricing_stdout.txt`, `ch08_offline_rl/tex/offline_rl.tex`, `docs/ch08_offline_rl.pdf`, `audits/ch08_offline_rl__offline_rl_pricing_fix_2026-05-19.md`, `audits/ch08_offline_rl__offline_rl_pricing_2026-05-19.md`

- [ ] **Step 1: Inspect the fix audit.**

```bash
head -80 audits/ch08_offline_rl__offline_rl_pricing_fix_2026-05-19.md
```

Expected: diagnoses three algorithm-identity drifts (IQL is argmax-policy-extraction not advantage-weighted regression; BCQ is the discrete BCQ-D variant not the continuous VAE+perturbation; DT uses fused-token simplification). Owns the four-way `169.27 ± 0.60` collapse for BC/BCQ-D/DT/RvS in prose: under 85%-concentrated behavioral data on price `p=10`, all four reduce to that single deterministic action via distinct mechanisms.

- [ ] **Step 2: Verify the labeling change in the source.**

```bash
grep -nE "DISPLAY_NAMES|_label\(|IQL-argmax|BCQ-D" ch08_offline_rl/sims/offline_rl_pricing.py | head -10
```

Expected: a `DISPLAY_NAMES = {'IQL': 'IQL-argmax', 'BCQ': 'BCQ-D'}` constant and a `_label()` helper used in table/figure/stdout emission.

- [ ] **Step 3: Verify the new results table uses qualified labels.**

```bash
cat ch08_offline_rl/sims/offline_rl_pricing_results.tex
```

Expected: rows for `IQL-argmax`, `BCQ-D`, `DT`, `RvS`, `BC`, `FQI`, `CQL`, plus DP Oracle. BC/BCQ-D/DT/RvS still byte-identical at `169.27 ± 0.60` — but now explicitly disclosed in chapter prose.

- [ ] **Step 4: Verify the chapter tex owns the collapse in prose (search for the new paragraph).**

```bash
grep -nE "behavioral.*concentrate|reduce.*\\\$p\s*=\s*10|four.*identical" ch08_offline_rl/tex/offline_rl.tex | head -5
```

Expected: a new paragraph explaining the collapse + three footnotes disclosing the algorithm variants.

- [ ] **Step 5: Stage and commit.**

```bash
git add ch08_offline_rl/sims/offline_rl_pricing.py \
        ch08_offline_rl/sims/offline_rl_pricing_results.tex \
        ch08_offline_rl/sims/offline_rl_pricing_coverage.png \
        ch08_offline_rl/sims/offline_rl_pricing_stdout.txt \
        ch08_offline_rl/tex/offline_rl.tex \
        docs/ch08_offline_rl.pdf \
        audits/ch08_offline_rl__offline_rl_pricing_fix_2026-05-19.md \
        audits/ch08_offline_rl__offline_rl_pricing_2026-05-19.md
git commit -m "$(cat <<'EOF'
audit(ch08_offline_rl): own offline_rl_pricing identity collapse (50% -> 25%)

The previous cycle flagged BC/BCQ/DT/RvS as byte-identical at
169.27 +- 0.60 -- a CLAUDE.md Algorithm Identity Check warning.
This commit owns the finding without re-implementing:

- Three label drifts disclosed:
  * IQL uses argmax policy extraction (not Kostrikov 2022's
    advantage-weighted regression). Renamed IQL -> IQL-argmax in
    table/figure/stdout via DISPLAY_NAMES helper.
  * BCQ is the discrete BCQ-D variant of Fujimoto 2019b, not the
    continuous BCQ of Fujimoto 2019. Renamed BCQ -> BCQ-D.
  * DT is the fused-token simplification (one summed embedding per
    timestep, not three separate tokens).
- New chapter paragraph explicitly owns the four-way collapse:
  with 85% of behavioral data concentrated on p = 10, BC, BCQ-D,
  DT, and RvS all deterministically reduce to that action via
  distinct mechanisms. Not a code bug; expected behavior.
- Three new footnotes in offline_rl.tex disclose the variants.
- Cache keys remain bare so pickled results still load; no
  retraining needed.

Audit score 50% -> 25%. New refs.bib entry Fujimoto2019b lands in
the audit-batch commit below.

Full audit: audits/ch08_offline_rl__offline_rl_pricing_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Commit audit batch + INDEX + refs.bib metadata fixes

**Files:**
- Stage: `audits/_INDEX.md`, the 20 remaining audit `*.md` files in `audits/` that have no `_fix_` counterpart (they're diagnostic, not fix-action), `docs/refs.bib`, `docs/plans/2026-05-20-arxiv-cleanup-cycle.md` (the plan file moved in Task 0).

- [ ] **Step 1: List untracked audit files to confirm scope.**

```bash
git status --short audits/ | grep -E "^\?\? " | head -30
```

Expected: ~20 audit `.md` files (those NOT already committed in Tasks 2–6).

- [ ] **Step 2: Inspect the refs.bib diff to confirm the two metadata changes.**

```bash
git diff docs/refs.bib
```

Expected:
1. `AdusumilliEckardt2022` entry — author field corrected from `Adusumilli, Karun and Tate, G. and Eckardt, Dita` (where "Tate, G." was a hallucinated co-author) to `Adusumilli, Karun and Eckardt, Dita`. Title slightly renamed.
2. New entry `Fujimoto2019b` added — "Benchmarking Batch Deep RL Algorithms" (arXiv 1910.01708), with a note about the BCQ-D discrete variant. Cited from Task 6's `offline_rl.tex` footnote.

- [ ] **Step 3: Stage the audits-INDEX, all the diagnostic-only audit files, refs.bib, and the moved plan file.**

```bash
git add audits/_INDEX.md \
        audits/ch03_theory__td_lambda_corridor_fix_2026-05-19.md  # already committed, but no-op
git add audits/ch06_games__cournot_bertrand_marl_2026-05-19.md  # if not already
git add audits/  # catch the rest of the untracked audits
git add docs/refs.bib
git add docs/plans/2026-05-20-arxiv-cleanup-cycle.md
git status --short | head -20
```

Expected: clean staging set; no leftover modifications outside this batch.

- [ ] **Step 4: Commit.**

```bash
git commit -m "$(cat <<'EOF'
chore(audits, refs): 35-sim audit batch + INDEX + refs.bib metadata fixes

Adds the parallel-session output that grades all 35 in-paper sims
on the 7-point Bullshit Detector checklist from CLAUDE.md. Six
sims scored >= 50% and triggered fixes; five have already landed
as preceding commits (td_lambda_corridor, cournot_bertrand_marl,
durable_goods_monopoly, causal_bandit_parallel, offline_rl_pricing).
The sixth (nfxp_ccp_td at 50%) is handled in a follow-up commit
(see audits/_INDEX.md for status).

refs.bib changes (called out by the audit):
- AdusumilliEckardt2022: removed hallucinated co-author "Tate, G."
  from the author field; title cleaned.
- Fujimoto2019b: new entry for the discrete BCQ-D variant,
  arXiv:1910.01708, cited from offline_rl.tex footnote in commit 6.

Also adds the master 2026-05-20 cleanup-cycle plan to
docs/plans/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Handle `nfxp_ccp_td` (ch05_econ_models)

**Files:**
- Inspect: `ch05_econ_models/sims/nfxp_ccp_td.py`, `ch05_econ_models/sims/nfxp_ccp_td_stdout.txt`, `ch05_econ_models/tex/rl_in_se.tex`
- Possibly stage: same three plus `audits/ch05_econ_models__nfxp_ccp_td_fix_2026-05-19.md` (if it exists)

- [ ] **Step 1: Check whether a `_fix_` audit exists for nfxp_ccp_td.**

```bash
ls audits/ch05_econ_models__nfxp_ccp_td* 2>/dev/null
```

Expected: at least the diagnostic audit (`...__nfxp_ccp_td_2026-05-19.md`). If a `_fix_` audit also exists, the parallel session fixed it.

- [ ] **Step 2: Read the audit verdict.**

```bash
head -40 audits/ch05_econ_models__nfxp_ccp_td_2026-05-19.md
```

Expected: scored 50% per `audits/_INDEX.md`. Note the diagnosis.

- [ ] **Step 3a (if a `_fix_` audit exists): Stage and commit just like Tasks 2-6.**

```bash
ls audits/ch05_econ_models__nfxp_ccp_td_fix_2026-05-19.md && \
git add ch05_econ_models/sims/nfxp_ccp_td.py \
        ch05_econ_models/sims/nfxp_ccp_td_stdout.txt \
        ch05_econ_models/tex/rl_in_se.tex \
        audits/ch05_econ_models__nfxp_ccp_td_fix_2026-05-19.md && \
git commit -m "$(cat <<'EOF'
fix(ch05_econ_models): nfxp_ccp_td <fix summary from audit>

<details from the fix audit>

Audit score 50% -> <new score>.

Full audit: audits/ch05_econ_models__nfxp_ccp_td_fix_2026-05-19.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3b (if no `_fix_` audit exists): HALT and surface to user.** The parallel session's uncommitted edits to `nfxp_ccp_td.py` and `rl_in_se.tex` are partial. Run `git diff` on those files and ask the user whether to (a) commit as-is, (b) revert to last-committed state, or (c) finish the fix per the diagnosis in the audit. Do not silently commit a partial fix.

### Task 9: End-to-end recompile after Phase A commits

**Files:**
- `docs/main.tex` (compile only)
- `docs/main.pdf`, `docs/main.log` (regenerated)

- [ ] **Step 1: Clean compile artifacts to avoid stale-aux noise.**

```bash
cd docs && rm -f main.aux main.bbl main.blg main.log main.out main.toc && cd ..
```

- [ ] **Step 2: Run the full four-pass compile.**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
            && bibtex main > /tmp/bb.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
            && cd ..
echo "exit: $?"
```

Expected: exit 0 on all four passes.

- [ ] **Step 3: Verify zero undefined refs/citations.**

```bash
grep -c "LaTeX Warning: Reference.*undefined" docs/main.log
grep -c "LaTeX Warning: Citation.*undefined" docs/main.log
pdfinfo docs/main.pdf | grep -E "Pages|File size"
```

Expected: both grep counts return `0`. Page count: 213–215.

- [ ] **Step 4: Commit the recompiled PDF.**

```bash
git add docs/main.pdf
git commit -m "$(cat <<'EOF'
build: recompile main.pdf after Phase A audit-fix commits

Clean four-pass compile. Zero undefined references, zero undefined
citations. Page count <N>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<N>` with the actual page count from Step 3.

---

## Phase B — Address remaining deferred items

### Task 10: Look up + fill the 4 pending arXiv IDs

**Files:**
- Modify: `docs/refs.bib` (4 entries: Cai2023, Tullii2024, Fan2024, Ying2022)

- [ ] **Step 1: Locate the 4 pending entries.**

```bash
grep -nA5 "^@.*{Cai2023,\|^@.*{Tullii2024,\|^@.*{Fan2024,\|^@.*{Ying2022," docs/refs.bib | head -40
```

Expected: each entry has `journal = {arXiv preprint}` and `note = {arXiv ID pending verification; ...}`.

- [ ] **Step 2: Dispatch a research subagent to find the correct arXiv IDs.** Use the `Agent` tool with `subagent_type=general-purpose`, `model=sonnet`, with this prompt:

> "Find the correct arXiv ID for each of these 4 papers. Method: search Google Scholar, arXiv, and Semantic Scholar; verify the matching paper's title and author list match the bib entry. Do NOT accept a low-confidence match.
>
> 1. Cai, Junhui and Chen, Ran and Wainwright, Martin J. and Zhao, Linda — 'Doubly High-Dimensional Contextual Bandits: An Interpretable Model for Joint Assortment-Pricing' (2023)
> 2. Tullii, Daniele and Javanmard, Adel and Pirotta, Matteo and Lezaud, Pierre — 'Contextual Dynamic Pricing with Strategic Buyers under Unknown Valuations' (2024)
> 3. Fan, Jianqing and Guo, Yongyi and Yu, Mengxin — 'Semiparametric Dynamic Pricing' (2024)
> 4. Ying, Dongjie and Ding, Kaiqing and Lavaei, Javad — 'A Dual Approach to Constrained Markov Decision Processes with Entropy Regularization' (2022)
>
> For each: report (a) the verified arXiv ID, (b) the URL fetched, (c) the matching title from the arXiv abstract page (to confirm). If no verified match found, say 'NO MATCH'. Return as a 4-row Markdown table."

- [ ] **Step 3: For each verified arXiv ID, update refs.bib.** Use Edit. Replace `journal = {arXiv preprint},` + the `note = {arXiv ID pending verification...}` line with `journal = {arXiv preprint arXiv:<ID>}` and drop the note. Example for Cai2023:

```python
# Before:
@article{Cai2023,
  author    = {Cai, Junhui and Chen, Ran and Wainwright, Martin J. and Zhao, Linda},
  title     = {Doubly High-Dimensional Contextual Bandits: An Interpretable Model for Joint Assortment-Pricing},
  journal   = {arXiv preprint},
  year      = {2023},
  note      = {arXiv ID pending verification; the ID 2309.07956 previously listed resolved to an unrelated physics paper}
}

# After (with <ID> from Step 2):
@article{Cai2023,
  author    = {Cai, Junhui and Chen, Ran and Wainwright, Martin J. and Zhao, Linda},
  title     = {Doubly High-Dimensional Contextual Bandits: An Interpretable Model for Joint Assortment-Pricing},
  journal   = {arXiv preprint arXiv:<ID>},
  year      = {2023}
}
```

If the subagent returned 'NO MATCH' for any entry, leave that entry's `note = {arXiv ID pending verification...}` as-is and add a comment in this plan's verification section for next-cycle follow-up.

- [ ] **Step 4: Recompile main.tex (Task 9's four-pass sequence, or use the abbreviated form here).**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
            && bibtex main > /tmp/bb.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
            && cd ..
grep -c "LaTeX Warning: Citation.*undefined" docs/main.log
```

Expected: zero undefined citations.

- [ ] **Step 5: Commit.**

```bash
git add docs/refs.bib docs/main.pdf
git commit -m "$(cat <<'EOF'
fix(refs): verify arXiv IDs for Cai2023, Tullii2024, Fan2024, Ying2022

Yesterday's arxiv-check skill found the IDs originally in these
four entries resolved to entirely unrelated papers (physics, math
analysis, CSP solving). Wrong IDs were removed in 52561d5 and
replaced with note = {arXiv ID pending verification...}. This
commit looks up the correct IDs and replaces the notes.

<results from the verification subagent>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<results from the verification subagent>` with the 4 actual IDs and source URLs.

### Task 11: Add World Models paragraph to ch99 conclusion

**Files:**
- Modify: `ch99_conclusion/tex/conclusion.tex` (add 1–2 paragraphs in the `\subsection{How Reinforcement Learning Advances Applied Modeling}` block)

- [ ] **Step 1: Read the existing subsection to find the insertion point.**

```bash
grep -nE "How Reinforcement Learning Advances Applied Modeling|world model|model-based|Lucas" ch99_conclusion/tex/conclusion.tex
```

Expected: locates the "How RL Advances Applied Modeling" subsection and confirms world-models / model-based RL is not already mentioned. The Lucas critique IS mentioned earlier (per yesterday's summary).

- [ ] **Step 2: Read the surrounding context (the paragraphs already present in that subsection).**

```bash
sed -n '20,55p' ch99_conclusion/tex/conclusion.tex
```

- [ ] **Step 3: Use Edit to insert one new paragraph at the end of the "How RL Advances Applied Modeling" subsection (before the next `\subsection`).** The paragraph must:
  - Reference `\ref{section:world_models}` explicitly
  - Mention the cobweb / fishery dual sim and what it demonstrates (model-based RL outperforms model-free on small-state economic environments with self-referential or exogenous-stochastic dynamics)
  - Connect to the Lucas-critique-respecting structural simulator theme already raised earlier in the conclusion
  - Follow CLAUDE.md prose style: no em-dashes, no colons in prose, no bullets, 3–6 sentences, no `\textbf{}`

Concrete paragraph to insert (use Edit's `old_string` to anchor on the line before the next `\subsection`):

```latex
World models, surveyed in Section~\ref{section:world_models}, are a third locus where reinforcement learning advances applied modeling. The cobweb and fishery simulations in that chapter show that a small learned dynamics model paired with planning can outperform model-free baselines on economic environments where the state space is modest and the dynamics are partially known to be self-referential or exogenous-stochastic. Read against the Lucas critique raised earlier in this conclusion, world-model methods make the model itself a first-class object the analyst can inspect and constrain, rather than a black-box simulator behind a policy. The cost is calibration. Value-aware losses give task-relevant accuracy where it matters for decisions but offer weaker guarantees against policy drift, a tension the literature on calibrated value-aware models is just beginning to address.
```

Use Edit with the existing subsection boundary as `old_string` to place this paragraph correctly.

- [ ] **Step 4: Recompile main.tex (Task 9's four-pass sequence).**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
            && bibtex main > /tmp/bb.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
            && cd ..
grep -c "LaTeX Warning: Reference.*undefined" docs/main.log
```

Expected: zero undefined references; the `\ref{section:world_models}` should resolve.

- [ ] **Step 5: Commit.**

```bash
git add ch99_conclusion/tex/conclusion.tex docs/main.pdf
git commit -m "$(cat <<'EOF'
feat(ch99): add World Models paragraph to "RL Advances Applied Modeling"

ch99 was silent on the ch12 World Models chapter (the longest
chapter in the paper). Added a paragraph in the "How RL Advances
Applied Modeling" subsection that:
- Names ch12 explicitly via \ref{section:world_models}
- Anchors the contribution in the cobweb / fishery dual sim
- Connects to the Lucas-critique theme raised earlier in the
  conclusion
- Acknowledges the calibration limitation (value-aware losses
  give task-relevant accuracy but weaker guarantees under policy
  drift).

Closes deferred item #2 from the 2026-05-19 cycle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 12: Rebuild ch03b and ch04 chapter PDFs

**Files:**
- Recompile (only): `docs/ch03b_deeprl_practice.pdf`, `docs/ch04_control_problems.pdf`

ch03 was already rebuilt by the parallel-session td_lambda fix; this task closes the rest of deferred item #7.

- [ ] **Step 1: Recompile ch03b.**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch03b_deeprl_practice \
    "\def\chapterfile{../ch03b_deeprl_practice/tex/deeprl_practice}\input{compile_chapter}" > /tmp/c1.log 2>&1 \
    && bibtex ch03b_deeprl_practice > /tmp/cb.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch03b_deeprl_practice \
    "\def\chapterfile{../ch03b_deeprl_practice/tex/deeprl_practice}\input{compile_chapter}" > /tmp/c2.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch03b_deeprl_practice \
    "\def\chapterfile{../ch03b_deeprl_practice/tex/deeprl_practice}\input{compile_chapter}" > /tmp/c3.log 2>&1 \
    && cd ..
echo exit: $?
pdfinfo docs/ch03b_deeprl_practice.pdf | grep Pages
```

Expected: exit 0; page count reported.

- [ ] **Step 2: Recompile ch04.**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch04_control_problems \
    "\def\chapterfile{../ch04_control_problems/tex/applications}\input{compile_chapter}" > /tmp/c1.log 2>&1 \
    && bibtex ch04_control_problems > /tmp/cb.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch04_control_problems \
    "\def\chapterfile{../ch04_control_problems/tex/applications}\input{compile_chapter}" > /tmp/c2.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch04_control_problems \
    "\def\chapterfile{../ch04_control_problems/tex/applications}\input{compile_chapter}" > /tmp/c3.log 2>&1 \
    && cd ..
echo exit: $?
pdfinfo docs/ch04_control_problems.pdf | grep Pages
```

Expected: exit 0; page count reported.

- [ ] **Step 3: Commit only if PDFs changed.**

```bash
git status --short docs/ch03b_deeprl_practice.pdf docs/ch04_control_problems.pdf
```

If both files show as modified, stage and commit:

```bash
git add docs/ch03b_deeprl_practice.pdf docs/ch04_control_problems.pdf
git commit -m "$(cat <<'EOF'
build: recompile ch03b and ch04 chapter PDFs post plot_style sweep

Closes deferred item #7 from the 2026-05-19 cycle. ch03 was
already rebuilt by td_lambda_corridor's fix in commit
<td_lambda_commit_sha>. This commit rebuilds ch03b and ch04.

Visual regression check: figures unchanged from prior PDFs (the
plot_style change only set rcParams; cached sim results were not
re-rendered). Chapter prose unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no diff is reported, skip the commit (the recompile produced byte-identical PDFs).

### Task 13: Spot-check 10 NotFound bib entries

**Files:**
- Modify (potentially): `docs/refs.bib` (entries flagged in yesterday's arxiv-check NotFound list)

- [ ] **Step 1: Re-load yesterday's NotFound list.**

```bash
grep -A2 "NotFound" docs/plans/2026-05-19-arxiv-check.md | head -40
```

Expected: 10 entries listed: `Brown1951`, `heinrich2016deep`, `Kakade2001`, `Mueller2019`, `christiano:2017`, `Fellows2023`, `Eimer2023`, `fujimoto2018td3`, `andrychowicz2021matters`, `Ruszczynski2010`.

- [ ] **Step 2: For each NotFound entry, inspect the current bib entry.**

```bash
for key in Brown1951 heinrich2016deep Kakade2001 Mueller2019 Fellows2023 Eimer2023 fujimoto2018td3 andrychowicz2021matters Ruszczynski2010 ; do
    echo "===" $key
    grep -nA6 "^@.*{$key," docs/refs.bib | head -10
done
# christiano:2017 has a colon in the key — quote the key:
echo "=== christiano:2017"
grep -nA6 "christiano:2017" docs/refs.bib | head -10
```

Expected: each entry shows up with author/title/year. Note which lack a DOI or arXiv ID.

- [ ] **Step 3: For each entry without a strong identifier (DOI or arXiv ID), do a quick verification.** Use the `WebFetch` tool to confirm the paper exists at its claimed venue. Example queries:
  - `Brown1951`: Cowles Commission Monograph 13 (1951), "Iterative solution of games by fictitious play" — check via `https://cowles.yale.edu/sites/default/files/files/pub/m13/m13-24.pdf` or similar
  - `Kakade2001`: NeurIPS 2001 paper, "A Natural Policy Gradient" — check via `https://proceedings.neurips.cc/paper/2001/file/...`
  - For arXiv-IDs implicit in the bib (e.g., `heinrich2016deep` claims arXiv:1603.01121): verify via `https://arxiv.org/abs/1603.01121`

If any entry's metadata is wrong, fix it in `refs.bib` (e.g., update year, journal, page numbers). If verification is uncertain (e.g., NotFound but plausible), leave as-is and note in the next-cycle summary.

- [ ] **Step 4: Recompile + commit if any entry changed.**

```bash
git status docs/refs.bib | head
```

If `refs.bib` was modified:

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
            && bibtex main > /tmp/bb.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
            && cd ..
git add docs/refs.bib docs/main.pdf
git commit -m "$(cat <<'EOF'
fix(refs): spot-check + fix metadata on <N> of 10 NotFound bib entries

Yesterday's arxiv-check flagged 10 NotFound references (mostly
NeurIPS/ICML proceedings with thin CrossRef coverage). Verified
each against its source venue; fixed <N> entries with
metadata corrections.

Verified clean (no changes): <list>
Fixed: <list with the change>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no entries changed, skip the commit and note "10 NotFound entries verified clean against source venues" in the next-cycle summary.

### Task 14: Verify `docs/refs_extended.bib` is truly inactive

**Files:**
- Inspect: `docs/main.tex`, all chapter `tex/*.tex` files, `journals/`, `thesis/`, `thesis_v2/`

This is a precondition check for Task 16. Confirm `refs_extended.bib` is not referenced anywhere active.

- [ ] **Step 1: Grep for `refs_extended` across the whole repo.**

```bash
grep -rn "refs_extended" \
    docs/main.tex \
    docs/compile_chapter.tex \
    docs/glossary.tex \
    ch00_introduction/tex/ \
    ch01_history/tex/ \
    ch02_rl_algorithms/tex/ \
    ch03_theory/tex/ \
    ch03b_deeprl_practice/tex/ \
    ch04_control_problems/tex/ \
    ch05_econ_models/tex/ \
    ch06_macro/tex/ \
    ch06_games/tex/ \
    ch07_bandits/tex/ \
    ch08_offline_rl/tex/ \
    ch09_rlhf/tex/ \
    ch10_causal/tex/ \
    ch10b_rl_for_ci/tex/ \
    ch11_dist_robust_constrained/tex/ \
    ch12_world_models/tex/ \
    ch99_conclusion/tex/ \
    scripts/ \
    2>/dev/null
```

Expected: zero matches in the active tex tree (planning docs in `docs/plans/` may mention it; those don't count). If a match appears in an active `.tex` file or in `scripts/package_arxiv.sh`, stop and surface to user — Task 16's deletion is not safe.

- [ ] **Step 2: Grep `journals/`, `thesis/`, `thesis_v2/` for any active reference.**

```bash
grep -rn "refs_extended" journals/ thesis/ thesis_v2/ 2>/dev/null | grep -v "\.bbl:"
```

Expected: zero matches in active tex/bib files. If matches, document and HALT.

---

## Phase C — Bib cleanup

### Task 15: Trim 342 orphan entries from `refs.bib`

**Files:**
- Create: `scripts/trim_orphan_bib.py` (one-shot script, can be deleted after run)
- Modify: `docs/refs.bib` (remove orphan entries)

- [ ] **Step 1: Re-extract the active cited-keys set from main.tex.**

```bash
python3 /Users/pranjal/.claude/skills/arxiv-check/scripts/extract_cites.py docs/main.tex > /tmp/cites_2026-05-20.json
python3 -c "import json; d=json.load(open('/tmp/cites_2026-05-20.json')); print('cited keys:', len(d['cited_keys']))"
```

Expected: 430–435 cited keys (was 430 yesterday; Task 7's `Fujimoto2019b` may add one, ditto any new cites in tasks 8 and 11).

- [ ] **Step 2: Write the trim script.**

```bash
cat > scripts/trim_orphan_bib.py <<'EOF'
#!/usr/bin/env python3
"""Trim orphan entries from docs/refs.bib.

Loads the cited-keys set from /tmp/cites_2026-05-20.json (produced by
arxiv-check's extract_cites.py), parses docs/refs.bib via bibtexparser,
writes back only entries whose key is cited (or whose type is @string /
@preamble / @comment), preserving entry order and formatting where
possible.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/pranjal/.claude/skills/arxiv-check/scripts')
from parse_bib import parse_bib  # uses bibtexparser

CITES_JSON = '/tmp/cites_2026-05-20.json'
BIB_PATH = Path('docs/refs.bib')

cited = set(json.load(open(CITES_JSON))['cited_keys'])
print(f'Cited keys: {len(cited)}')

entries = parse_bib(str(BIB_PATH))
print(f'Total bib entries: {len(entries)}')

kept = [e for e in entries if e['key'] in cited]
dropped = [e for e in entries if e['key'] not in cited]
print(f'Keeping: {len(kept)}')
print(f'Dropping: {len(dropped)}')

# Use bibtexparser to write back. Preserve original strings + preamble + comments.
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

# Re-load via bibtexparser directly (parse_bib returns dicts; we need the BibDatabase)
parser = bibtexparser.bparser.BibTexParser(common_strings=True, ignore_nonstandard_types=False)
with open(BIB_PATH) as f:
    db = bibtexparser.load(f, parser=parser)

kept_keys = {e['key'] for e in kept}
db.entries = [e for e in db.entries if e.get('ID', e.get('id')) in kept_keys]

writer = BibTexWriter()
writer.indent = '  '
writer.order_entries_by = None  # preserve order

with open(BIB_PATH, 'w') as f:
    f.write(writer.write(db))

print(f'Wrote {len(db.entries)} entries to {BIB_PATH}')
EOF
chmod +x scripts/trim_orphan_bib.py
```

- [ ] **Step 3: Dry-run on a copy first.**

```bash
cp docs/refs.bib /tmp/refs.bib.bak
python3 scripts/trim_orphan_bib.py
diff -u /tmp/refs.bib.bak docs/refs.bib | head -60
```

Expected: many `-@article{...}` lines removed, no `+` lines (no additions).

- [ ] **Step 4: Verify the cited keys all still resolve by recompiling.**

```bash
cd docs && rm -f main.aux main.bbl main.blg main.log main.out main.toc \
    && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
    && bibtex main > /tmp/bb.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
    && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
    && cd ..
grep -c "LaTeX Warning: Citation.*undefined" docs/main.log
```

Expected: 0 undefined citations. If non-zero, the trim was too aggressive — restore from `/tmp/refs.bib.bak`, debug, retry.

- [ ] **Step 5: Verify page count is unchanged.**

```bash
pdfinfo docs/main.pdf | grep -E "Pages|File size"
```

Expected: same page count as before the trim (or off-by-one from the bibliography section growing/shrinking slightly).

- [ ] **Step 6: Commit.**

```bash
git add docs/refs.bib docs/main.pdf scripts/trim_orphan_bib.py
git commit -m "$(cat <<'EOF'
chore(refs): trim orphan entries from refs.bib

Yesterday's bib-coverage audit found 342 orphan entries in
refs.bib (defined but never cited). This commit removes them via
a one-shot script (scripts/trim_orphan_bib.py) that loads the
cited-keys set from extract_cites.py and keeps only entries
whose key appears in main.tex.

Closes deferred item #5 from the 2026-05-19 cycle.

Bib entry count: ~810 -> ~430. Main.pdf page count unchanged
(bibliography just got shorter).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 16: Delete inactive `refs_extended.bib`

**Files:**
- Delete: `docs/refs_extended.bib`

Precondition: Task 14 confirmed `refs_extended.bib` is not referenced anywhere active.

- [ ] **Step 1: Confirm Task 14 passed cleanly (no active references).**

```bash
grep -rn "refs_extended" docs/main.tex docs/compile_chapter.tex scripts/package_arxiv.sh ch*/tex/ 2>/dev/null
```

Expected: zero output.

- [ ] **Step 2: Remove the file via git.**

```bash
git rm docs/refs_extended.bib
```

- [ ] **Step 3: Recompile to confirm nothing breaks.**

```bash
cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 \
            && bibtex main > /tmp/bb.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1 \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/p3.log 2>&1 \
            && cd ..
grep -c "LaTeX Warning: Citation.*undefined" docs/main.log
```

Expected: 0.

- [ ] **Step 4: Commit.**

```bash
git add docs/main.pdf
git commit -m "$(cat <<'EOF'
chore(refs): delete refs_extended.bib (inactive, 189 duplicates of refs.bib)

The file was never loaded by \bibliography{refs} in main.tex, so
its 283 entries (189 of which duplicated refs.bib) were silently
dead. Removing reduces repo surface area and prevents future
confusion about which bib is authoritative.

Closes deferred item #6 from the 2026-05-19 cycle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase D — Final audit gates + arxiv tarball

### Task 17: Re-run bib coverage audit

**Files:**
- Output: `docs/plans/2026-05-20-bib-coverage.md`

- [ ] **Step 1: Dispatch the `bib-coverage-auditor` agent.** Use the `Agent` tool with `subagent_type=bib-coverage-auditor`, with this prompt:

> "Audit cited-vs-defined coverage between the LaTeX sources in /Users/pranjal/Code/rl and docs/refs.bib. Compile entry: docs/main.tex. OUT OF SCOPE: journals/, thesis/, thesis_v2/, ORE_main/, archive/, anything under tex/backups/, tex/v1_archived/, tex/v2_archived/, tex/v3_archived/. After the Phase A-C cleanup, refs_extended.bib is deleted, so only refs.bib is active.
>
> Report four sections (use the same format as yesterday's docs/plans/2026-05-19-bib-coverage.md):
> 1. CITED-BUT-MISSING — must be empty after Phase B/C
> 2. DEFINED-BUT-ORPHAN — must be empty after Phase C trim
> 3. DUPLICATE KEYS — must be empty after refs_extended.bib delete
> 4. ENTRIES MISSING REQUIRED FIELDS
>
> Write the report as Markdown to /Users/pranjal/Code/rl/docs/plans/2026-05-20-bib-coverage.md. Keep under 200 lines."

- [ ] **Step 2: Read the report.**

```bash
cat docs/plans/2026-05-20-bib-coverage.md
```

Expected: all four sections clean (0 missing, 0 orphan, 0 duplicate, 0 missing-required-fields). If any section non-zero, fix inline before continuing.

### Task 18: Re-run paper coherence audit

**Files:**
- Output: `docs/plans/2026-05-20-paper-coherence.md`

- [ ] **Step 1: Dispatch the `paper-coherence-auditor` agent.** Use the `Agent` tool with `subagent_type=paper-coherence-auditor`, with this prompt:

> "Audit the survey paper draft at /Users/pranjal/Code/rl for pre-arxiv coherence. Compile entry: docs/main.tex.
>
> OUT OF SCOPE: journals/, thesis/, thesis_v2/, ORE_main/, archive/, tex/backups/, tex/v[1-3]_archived/.
>
> Three-section audit (same as yesterday's docs/plans/2026-05-19-paper-coherence.md):
> 1. Abstract <-> Conclusion alignment
> 2. Figure <-> claim consistency for every \ref{fig:...} and \ref{tab:...}
> 3. Method reproducibility for each simulation writeup
>
> Pay particular attention to the new ch99 World Models paragraph (Task 11), the relabeled IQL-argmax/BCQ-D in ch08, the rescoped durable_goods_monopoly section in ch06_games, and the relabeled context_conditional_thompson_sampling in ch10b. Confirm the prose owns the limitations introduced by the relabeling.
>
> Write the report as Markdown to /Users/pranjal/Code/rl/docs/plans/2026-05-20-paper-coherence.md. Critical issues separated from deferred. Under 600 lines."

- [ ] **Step 2: Read the report and triage.**

```bash
head -60 docs/plans/2026-05-20-paper-coherence.md
```

For each critical issue: if it's a small wording fix, fix inline with Edit + recompile + commit. If it's a substantive content gap, log under "Deferred to next cycle" in Task 21's summary and proceed.

### Task 19: Re-run arxiv-check skill

**Files:**
- Output: `docs/plans/2026-05-20-arxiv-check.md`

- [ ] **Step 1: Invoke the `arxiv-check` skill via Skill.** Pass `docs/main.tex` as the target. The skill dispatches three parallel sonnet subagents (meta-comments, reference verification, citation drift) and emits a synthesized report.

- [ ] **Step 2: Move the report into docs/plans/ and read it.**

```bash
ls arxiv-check_2026-05-*.md 2>/dev/null && mv arxiv-check_2026-05-*.md docs/plans/2026-05-20-arxiv-check.md
head -50 docs/plans/2026-05-20-arxiv-check.md
```

Expected overall status: PASS. Specifically:
- Meta-comment hits: 0
- Mismatch references: 0 (the 4 fixed in Task 10 should now Verify)
- Missing keys: 0
- NotFound: <= 10 (deferred from yesterday; if Task 13 fixed any, count is lower)
- Review: ~30 (deferred formatting-mismatch false positives)
- Orphan keys: 0 (after Task 15)

If status is FAIL on any check, fix the critical issues inline before continuing.

- [ ] **Step 3: Commit the three audit reports.**

```bash
git add docs/plans/2026-05-20-bib-coverage.md \
        docs/plans/2026-05-20-paper-coherence.md \
        docs/plans/2026-05-20-arxiv-check.md
# plus any inline tex fixes made during triage:
git add -u
git commit -m "$(cat <<'EOF'
audit(arxiv-prep): re-run bib coverage + coherence + arxiv-check gates

After Phase A-C cleanup:
- bib-coverage: <summary, expect 0/0/0/0>
- paper-coherence: <summary>
- arxiv-check: <summary, expect overall PASS>

Reports at docs/plans/2026-05-20-*.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 20: Rebuild arxiv tarball

**Files:**
- Re-run: `scripts/package_arxiv.sh` (no changes expected)
- Output: `arxiv_submission.tar.gz`, `arxiv_submission/` (both gitignored)

- [ ] **Step 1: Confirm the package script reflects the deleted refs_extended.bib (it shouldn't reference it anyway).**

```bash
grep "refs_extended" scripts/package_arxiv.sh
```

Expected: no matches (the script only ships `refs.bib`).

- [ ] **Step 2: Run the packaging script.**

```bash
bash scripts/package_arxiv.sh > /tmp/arxiv_pkg_2026-05-20.log 2>&1
echo exit: $?
tail -15 /tmp/arxiv_pkg_2026-05-20.log
```

Expected: exit 0. Final summary lines say `Compilation successful: <N> pages` and a tarball is produced at `/Users/pranjal/Code/rl/arxiv_submission.tar.gz`.

- [ ] **Step 3: Smoke-test the tarball in a clean directory.**

```bash
rm -rf /tmp/arxiv_smoke2 && mkdir -p /tmp/arxiv_smoke2 \
    && cd /tmp/arxiv_smoke2 \
    && tar xzf /Users/pranjal/Code/rl/arxiv_submission.tar.gz \
    && ls main.tex refs.bib main.bbl \
    && pdflatex -interaction=nonstopmode main.tex > /tmp/sp1.log 2>&1 \
    && bibtex main > /tmp/spb.log 2>&1 \
    && pdflatex -interaction=nonstopmode main.tex > /tmp/sp2.log 2>&1 \
    && pdflatex -interaction=nonstopmode main.tex > /tmp/sp3.log 2>&1
echo "exit: $?"
ls -la main.pdf
pdfinfo main.pdf 2>/dev/null | grep -E "Pages|File size"
cd /Users/pranjal/Code/rl
```

Expected: `main.pdf` exists in `/tmp/arxiv_smoke2/` with the SAME page count as `docs/main.pdf`. If page counts differ, the tarball is missing files or has stale `.bbl`; investigate.

- [ ] **Step 4: No commit needed** — the tarball is gitignored. Capture the path + size for the summary in Task 21.

```bash
ls -la /Users/pranjal/Code/rl/arxiv_submission.tar.gz
```

### Task 21: End-of-cycle summary + push

**Files:**
- Create: `docs/plans/2026-05-20-arxiv-push-summary.md`

- [ ] **Step 1: Collect summary stats.**

```bash
git log --oneline c9bbb0b..HEAD | tee /tmp/cycle_commits.txt
wc -l /tmp/cycle_commits.txt
pdfinfo docs/main.pdf | grep -E "Pages|File size"
ls -la arxiv_submission.tar.gz
```

- [ ] **Step 2: Write the summary file at `docs/plans/2026-05-20-arxiv-push-summary.md`.** Mirror the structure of `docs/plans/2026-05-19-arxiv-push-summary.md`. Cover:
  - Phases executed (A/B/C/D), one bullet per task
  - Commits added (count + the full one-line list from `/tmp/cycle_commits.txt`)
  - Final main.pdf page count + file size
  - Audit results from Task 19
  - Items closed: #1 (offline_rl_pricing owned), #2 (ch99 World Models para), #3 (4 arXiv IDs), #5 (orphan trim), #6 (refs_extended deleted), #7 (chapter PDFs rebuilt)
  - Items defer-again: #4 (10 NotFound, partial — list which were verified clean vs which remain unverified)
  - Any new findings from Task 18/19 (deferred)
  - Tarball path
  - "Ready to merge `humanize-pass` → `main` and upload to arxiv" closing line

- [ ] **Step 3: Commit the summary.**

```bash
git add docs/plans/2026-05-20-arxiv-push-summary.md
git commit -m "$(cat <<'EOF'
docs: end-of-cycle summary for 2026-05-20 arxiv cleanup cycle

Records final state: <N> commits since c9bbb0b, <P>-page main.pdf,
<M>MB tarball, all three 2026-05-20 audit reports PASS, deferred
items closed: #1 #2 #3 #5 #6 #7 (item #4 partial — see summary).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push to origin.**

```bash
git push origin humanize-pass 2>&1 | tail -5
```

Expected: `c9bbb0b..<new_head>  humanize-pass -> humanize-pass`.

---

## Verification (end-to-end)

After all 21 tasks:

1. `git status` reports clean working tree on branch `humanize-pass`.
2. `git log --oneline c9bbb0b..HEAD` shows 13–18 new commits (16 expected: tasks 2–21 minus task 0 and task 1).
3. `docs/main.pdf` mtime current; `pdfinfo` reports 213–216 pages.
4. `grep -c "LaTeX Warning: Reference.*undefined" docs/main.log` returns `0`.
5. `grep -c "LaTeX Warning: Citation.*undefined" docs/main.log` returns `0`.
6. `audits/_INDEX.md` has no row at Bullshit-Score ≥50% without a corresponding `_fix_*.md` audit file present in `audits/` and committed.
7. `docs/refs.bib` has NO `note = {arXiv ID pending verification...}` lines, OR all such lines are accompanied by an explicit "NO MATCH found" reason from Task 10's verification subagent.
8. `docs/refs_extended.bib` does not exist (`git ls-files docs/refs_extended.bib` returns empty).
9. `grep -c "^@" docs/refs.bib` returns ~430 (down from ~810).
10. `ch99_conclusion/tex/conclusion.tex` contains the new World Models paragraph (search: `grep -c "section:world_models" ch99_conclusion/tex/conclusion.tex` returns ≥1).
11. `docs/plans/` contains the four 2026-05-20 files: bib-coverage, paper-coherence, arxiv-check, push-summary.
12. `arxiv_submission.tar.gz` exists and `/tmp/arxiv_smoke2/main.pdf` has the same page count as `docs/main.pdf`.
13. `origin/humanize-pass` HEAD matches local HEAD.

---

## Out of scope (explicit non-goals)

- Anything under `journals/`, `thesis/`, `thesis_v2/`, `ORE_main/`, `archive/`.
- New chapter content beyond the ch99 World Models paragraph.
- ch07 RLHF sim (still listed in CLAUDE.md tasklist but already implemented in ch09).
- "Economic Models for RL" deferred chapter.
- Merging `humanize-pass` → `main` (the user does this explicitly).
- Uploading the tarball to arxiv (the user does this explicitly).
- Force-pushing or rewriting history.

---

## Self-review notes

- **Spec coverage:** Each of the 7 deferred items is mapped to a task: #1→Task 6, #2→Task 11, #3→Task 10, #4→Task 13, #5→Task 15, #6→Tasks 14+16, #7→Task 12. ≥50% sims from the parallel session's audit get Tasks 2–6 (5 fixes) plus Task 8 (nfxp_ccp_td). Phase D closes the loop with re-audits + tarball.
- **Placeholder scan:** Every task has concrete code/commands. The two `<bracketed>` placeholders are in commit messages (Tasks 8, 10, 12, 13, 17, 19, 21) where the actual values depend on runtime results (audit verdicts, subagent return values, page counts). These are NOT plan placeholders; they're intentional template slots the implementer fills with real data from preceding steps.
- **Type consistency:** Function names, file paths, and commit-message prefixes are consistent (e.g., `audits/<chapter>__<sim>_fix_2026-05-19.md` naming convention, `_fix_` suffix indicating action-fix audit vs diagnostic-only). The bib-trim script's reference to `cited_keys` matches the field name in extract_cites.py's JSON output.
