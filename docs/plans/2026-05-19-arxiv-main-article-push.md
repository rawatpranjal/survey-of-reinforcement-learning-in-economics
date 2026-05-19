# Main arxiv-article push: finish three new content streams + polish sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) to execute this plan task-by-task with a fresh subagent per task. Steps use checkbox (`- [ ]`) syntax for tracking. The user wants autonomous execution over an extended period and has explicitly accepted long runtimes.

> **First execution action (Task 0.0):** Move this plan from `/Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md` to `docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md` (the canonical location for plans in this repo). Plan mode constrained where it could be written, but the canonical home is in-repo.

**Goal:** Finish the open work on the main arxiv-article version of the survey: commit the humanize-pass branch, close the small loose ends in the three new content streams (ch06_macro / ch10_causal / ch10b_rl_for_ci / ch12_world_models), standardize stragglers in ch02–ch05 to current sim conventions, then run a final cross-cutting audit and recompile `docs/main.pdf` for arxiv push.

**Architecture:** Five phases. Phase 0 is mechanical branch hygiene (commits, baseline recompile). Phase 1 closes loose ends in the three recently-added content streams (sims and tex). Phase 2 is a mechanical sweep over 9 non-conforming sims in ch02–ch05 to bring them onto the `sims.plot_style` + `sims.sim_cache` standards from CLAUDE.md. Phase 3 dispatches the project's audit agents (`bib-coverage-auditor`, `paper-coherence-auditor`, the `arxiv-check` skill) for a pre-submission sweep. Phase 4 produces the final clean `docs/main.pdf` and arxiv tarball. The plan is explicitly NOT working on `journals/`.

**Tech Stack:** Python 3.10 + numpy / scipy / matplotlib / torch (existing sims), pdflatex + bibtex with natbib + econometrica.bst, the project's shared `sims/sim_cache.py` (`compute_or_load`, `add_component_args`, `parse_force_set`, `load_results`, `save_results`, `add_cache_args`) and `sims/plot_style.py` (`apply_style`, `COLORS`, `ALGO_COLORS`, `DOMAIN_COLORS`, `FIG_SINGLE`, `FIG_DOUBLE`, `BENCH_STYLE`, `CMAP_SEQ`). All commits land on the `humanize-pass` branch unless a Tier C item promotes a new branch.

---

## Context

The survey paper at `/Users/pranjal/Code/rl` is an arxiv-bound monograph (Pranjal Rawat / Georgetown). The current branch `humanize-pass` carries (a) humanize prose edits on ch02/ch08/ch10, (b) a folder rename `ch12_forecasting_rl/` → `ch12_world_models/`, and (c) a sim move `counterfactual_ope.*` from ch12 into ch10. None of this is committed yet.

Recent content additions across April–May 2026:
- **ch06_macro** — Restructured 2026-05-15 into a method-classified six-section spine. 1,321-line tex with 10 substantive subsections. Two main sims (`rbc_dp_vs_drl.py`, `lq_mfg.py`) plus a supplementary grid sweep. Compiled PDF current.
- **ch10_causal + ch10b_rl_for_ci** — Causal split landed 2026-05-12. Both chapters compile, both have substantive prose (325 / 361 lines), and all six sims have outputs.
- **ch12_world_models** — Rebuilt as "World Models and Model-Based Reinforcement Learning" with three core sims (Sutton blocking maze / cobweb / fishery) plus an auxiliary AR1 decision-focused sim. PDF current, 39 pages.

A three-agent parallel Explore audit (2026-05-19) and follow-up read of `ch09_rlhf/tex/rlhf.tex` confirms most CLAUDE.md "Tasklist" items are stale:
- ch09 RLHF tex deepening (DPO derivation, Bradley-Terry worked example, link to discrete choice): **already done** at lines 5–96 of `ch09_rlhf/tex/rlhf.tex`.
- ch09 RLHF/DPO simulation: **already done** — `job_search_dpo.py`, `job_search_rlhf.py`, `gridworld_rlhf.py`, `rlhf_dpo_pipeline.py`, `nfxp_vs_rlhf.py`, `preference_learning.py` all exist with figures, tables, and stdouts.
- ch02 tex consolidation between `planning_learning.tex` and `..._alt.tex`: **moot** — only `rl_algorithms.tex` exists in `ch02_rl_algorithms/tex/`; the variants moved to `ch03_theory/tex/` during the split.
- ch99 conclusion expansion: **already done** — `ch99_conclusion/tex/conclusion.tex` is a 4-subsection ~3,000-word section covering structure-improves-RL, RL-advances-applied-modeling, open challenges, and a closing paragraph.

Real remaining open work (the basis for this plan):
1. Commit the staged branch state.
2. Verify end-to-end `docs/main.tex` compile is clean after the ch12 rename.
3. Add one missing `.tex` results table to `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py`.
4. Refresh the `ch12_world_models/sims/dyna_maze.py` cache (CHAPTER_NOTES.md flags it as potentially stale).
5. Decide what to do with `ch12_world_models/sims/decision_focused_ar1.py` (orphan — not referenced in the chapter tex).
6. Bring 9 sims in ch02–ch05 onto the `sims.plot_style` standard required by CLAUDE.md.
7. Run the three audit gates (`bib-coverage-auditor`, `paper-coherence-auditor`, `arxiv-check`).
8. Final clean `docs/main.pdf` compile + arxiv package.

The intended outcome is a clean main-branch-mergeable `humanize-pass` with `docs/main.pdf` arxiv-ready.

---

## File Structure (where the work happens)

### Will be modified
- `ch10_causal/tex/causal_rl.tex` — already path-corrected for counterfactual_ope move; verify only.
- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py` — add `generate_outputs()` table emission.
- `ch10b_rl_for_ci/tex/rl_for_ci.tex` — add `\input` of new table.
- `ch12_world_models/sims/dyna_maze.py` — re-run with `--force` to refresh cache after agent-code updates.
- `ch12_world_models/sims/dyna_maze_stdout.txt` — regenerated.
- `ch12_world_models/tex/s03_dyna_q.tex` — update numerical claims if they drift from the new stdout.
- `ch12_world_models/CHAPTER_NOTES.md` — drop the "cache may be stale" open thread once refreshed.
- 9 sims in `ch02_rl_algorithms/sims/`, `ch03_theory/sims/`, `ch03b_deeprl_practice/sims/`, `ch04_control_problems/sims/` (enumerated in Phase 2 below) — refactor imports + colors.
- `docs/main.pdf`, plus the per-chapter PDFs touched (`ch10b_rl_for_ci.pdf`, `ch12_world_models.pdf`, any chapter whose sim figures changed).

### Will be created
- `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex` — new results table.
- `docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md` — canonical home for this plan (moved from plan-mode location at Task 0.0).
- `docs/superpowers/audits/2026-05-19-arxiv-check.md` — emitted by the `arxiv-check` skill in Phase 3.
- `docs/superpowers/audits/2026-05-19-bib-coverage.md` — emitted by the `bib-coverage-auditor` agent.
- `docs/superpowers/audits/2026-05-19-paper-coherence.md` — emitted by the `paper-coherence-auditor` agent.

### Will be moved (already staged, just needs commit)
- `ch12_forecasting_rl/` → `ch12_world_models/` (entire subtree).
- `ch12_forecasting_rl/sims/counterfactual_ope.{py,png,stdout.txt,table.tex}` → `ch10_causal/sims/counterfactual_ope.{py,png,stdout.txt,table.tex}`.

### Will not be touched (out of scope)
- Anything under `journals/`, `thesis/`, `thesis_v2/`, `ORE_main/`, `archive/`.
- ch09 RLHF tex/sims (already complete per audit).
- ch99 conclusion (already complete per audit).
- ch00/ch01/ch06b/ch07/ch08/ch11 — no flagged open items.

---

## Existing utilities to reuse (no re-implementation)

- `sims/sim_cache.py`: `compute_or_load`, `add_component_args`, `parse_force_set`, `load_results`, `save_results`, `add_cache_args`.
- `sims/plot_style.py`: `apply_style`, `COLORS`, `ALGO_COLORS`, `DOMAIN_COLORS`, `FIG_SINGLE`, `FIG_DOUBLE`, `FIG_TRIPLE`, `FIG_WIDE`, `FIG_SQUARE`, `BENCH_STYLE`, `CMAP_SEQ`.
- `docs/compile_chapter.tex` — per-chapter compile driver. Invoke per the CLAUDE.md "Key Commands" block.
- `scripts/run_all_sims.py` — supports `--chapter chXX`, `--script <name>`, `--plots-only`, `--list`.
- Agents available via `Agent`: `bib-coverage-auditor`, `paper-coherence-auditor`, `code-reviewer`, `latex-build-summarizer`.
- Skills available via `Skill`: `arxiv-check`, `humanizer`, `proofread`, `verification-before-completion`.

---

## Phase 0 — Branch hygiene + baseline (mechanical)

### Task 0.0: Move plan to canonical location

**Files:**
- Move: `/Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md` → `docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md`

- [ ] **Step 1: Create the in-repo plans directory if it does not exist.**
  ```bash
  mkdir -p docs/superpowers/plans docs/superpowers/audits
  ```

- [ ] **Step 2: Move the plan file.**
  ```bash
  mv /Users/pranjal/.claude/plans/no-dont-work-on-async-flamingo.md docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md
  ```

- [ ] **Step 3: Verify.**
  ```bash
  ls -la docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md
  ```
  Expected: file exists.

- [ ] **Step 4: Skip commit** (the next task will batch this with the staged humanize-pass changes).

### Task 0.1: Baseline state snapshot

**Files:**
- Read-only.

- [ ] **Step 1: Capture current `git status`.**
  ```bash
  git status --short > /tmp/arxiv_push_baseline_status.txt
  wc -l /tmp/arxiv_push_baseline_status.txt
  ```
  Expected: ~60 lines of `M` / `R` / `D` entries.

- [ ] **Step 2: Capture branch info.**
  ```bash
  git branch --show-current
  git log --oneline -5
  ```
  Expected branch: `humanize-pass`. Most recent commit: `9dbaab0 chore(docs): recompile main.pdf and update ch06_macro simulation output`.

- [ ] **Step 3: Confirm the four touched chapter PDFs all exist (sanity check before commit).**
  ```bash
  ls -la docs/ch06_macro.pdf docs/ch10_causal.pdf docs/ch10b_rl_for_ci.pdf docs/ch12_world_models.pdf
  ```
  Expected: all four exist, mtimes from the last few days.

### Task 0.2: Commit the staged humanize-pass + ch12 rename + counterfactual_ope move

**Files:** all entries from `git status` (per Task 0.1).

Per CLAUDE.md: NEVER use `git add -A` or `git add .`. Always stage specific files. Split into three commits for clean history.

- [ ] **Step 1: Stage commit 1 — humanize edits.**
  ```bash
  git add ch02_rl_algorithms/sims/algorithm_architectures.png \
          ch02_rl_algorithms/sims/algorithm_architectures.py \
          ch02_rl_algorithms/tex/rl_algorithms.tex \
          ch08_offline_rl/sims/offline_rl_pricing.py \
          ch08_offline_rl/sims/offline_rl_pricing_coverage.png \
          ch08_offline_rl/sims/offline_rl_pricing_results.tex \
          ch08_offline_rl/sims/offline_rl_pricing_stdout.txt \
          ch08_offline_rl/tex/offline_rl.tex \
          ch10_causal/tex/causal_rl.tex
  git status --short | grep -E '^[ M][ M]' | head
  ```
  Expected: the staged files appear with leading `M `.

- [ ] **Step 2: Commit 1.**
  ```bash
  git commit -m "$(cat <<'EOF'
  humanize(ch02, ch08, ch10): prose polish pass + offline_rl_pricing refresh

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: pre-commit hooks pass; one new commit on `humanize-pass`.

- [ ] **Step 3: Stage commit 2 — ch12 folder rename (all `R` entries pointing into `ch12_world_models/`).**
  ```bash
  git status --short | awk '$1 ~ /^R/ && $4 ~ /ch12_world_models/ { print $2, $4 }' | head
  ```
  This previews the renames. Stage them by re-running `git add` on the new paths (rename detection picks them up automatically):
  ```bash
  git add ch12_world_models/ ch12_forecasting_rl/ \
          docs/main.tex scripts/package_arxiv.sh 2>/dev/null || true
  git status --short | head -20
  ```
  Expected: only the rename entries (`R`) and the deleted `ch12_forecasting_rl/tex/05_forecaster_as_policy.tex` (`D`) remain staged.

- [ ] **Step 4: Commit 2.**
  ```bash
  git commit -m "$(cat <<'EOF'
  refactor(ch12): rename ch12_forecasting_rl -> ch12_world_models

  Reflects chapter retitle to "World Models and Model-Based RL". All
  \input paths in docs/main.tex and scripts/package_arxiv.sh updated.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: second commit on `humanize-pass`.

- [ ] **Step 5: Stage commit 3 — counterfactual_ope move ch12 → ch10.**
  ```bash
  git add ch10_causal/sims/counterfactual_ope.png \
          ch10_causal/sims/counterfactual_ope.py \
          ch10_causal/sims/counterfactual_ope_stdout.txt \
          ch10_causal/sims/counterfactual_ope_table.tex
  git status --short | head
  ```
  Expected: only the move entries (`R`) staged; working tree otherwise clean.

- [ ] **Step 6: Commit 3.**
  ```bash
  git commit -m "$(cat <<'EOF'
  refactor(ch10): move counterfactual_ope sim from ch12 into ch10_causal

  Counterfactual OPE belongs with the causal-inference-for-RL chapter
  (ch10), not the world-models chapter (ch12). Tex \input paths in
  ch10_causal/tex/causal_rl.tex already use the new ../ch10_causal/sims/
  prefix.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: third commit. `git status` reports clean.

- [ ] **Step 7: Stage commit 4 — move this plan into the repo.**
  ```bash
  git add docs/superpowers/plans/2026-05-19-arxiv-main-article-push.md
  git status --short
  ```
  Expected: just the new plan file staged.

- [ ] **Step 8: Commit 4.**
  ```bash
  git commit -m "$(cat <<'EOF'
  docs(plans): add 2026-05-19 arxiv-main-article-push plan

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 9: Verify clean tree.**
  ```bash
  git status
  git log --oneline -6
  ```
  Expected: `working tree clean`; four new commits visible.

### Task 0.3: Baseline end-to-end recompile

**Files:**
- `docs/main.tex` (compile only, no edits expected here).

- [ ] **Step 1: Clean compile artifacts to avoid stale-reference noise.**
  ```bash
  cd docs && rm -f main.aux main.bbl main.blg main.log main.out main.toc && cd ..
  ```

- [ ] **Step 2: Run the full four-pass compile.**
  ```bash
  cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && bibtex main \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && cd ..
  ```
  Expected: exit 0 on all four passes. If any pass fails, halt and dispatch the `latex-build-summarizer` agent on `docs/main.log` to extract only the Error/Warning lines.

- [ ] **Step 3: Summarize warnings.**
  ```bash
  grep -E "(LaTeX Warning: Reference|LaTeX Warning: Citation).*undefined" docs/main.log | sort -u | head -30
  grep -c "LaTeX Warning: Reference.*undefined" docs/main.log
  grep -c "LaTeX Warning: Citation.*undefined" docs/main.log
  ```
  Expected: 0 undefined references, 0 undefined citations after the third pass. If non-zero, capture the list — Phase 3 will resolve.

- [ ] **Step 4: Confirm PDF.**
  ```bash
  ls -la docs/main.pdf
  pdfinfo docs/main.pdf | grep Pages
  ```
  Expected: PDF mtime current, ~140 pages (full survey).

- [ ] **Step 5: Commit the recompiled PDF + any aux files git tracks.**
  ```bash
  git add docs/main.pdf docs/main.bbl 2>/dev/null
  git status --short
  git commit -m "$(cat <<'EOF'
  build: recompile main.pdf after humanize + ch12 rename

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  If `main.bbl` is gitignored, the add will silently skip it. Only `main.pdf` should commit.

---

## Phase 1 — Three-stream loose ends

### Task 1.1: Add results table to dtr_qlearning_vs_murphy sim

**Files:**
- Modify: `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py` (extend `generate_outputs()` only — never touch `compute_data()` per CLAUDE.md sim modularity rules)
- Create: `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex`
- Modify: `ch10b_rl_for_ci/tex/rl_for_ci.tex` (add `\input` of the new table)

- [ ] **Step 1: Read the current sim to understand what `data` dict already contains.**
  ```bash
  wc -l ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py
  grep -n "def compute_data\|def generate_outputs\|return\|data\[" \
       ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py | head -30
  ```
  Expected: identify the `data` keys (likely `qlearning_mean`, `qlearning_se`, `murphy_value`, `optimal_value`, sample-complexity series).

- [ ] **Step 2: Extend `generate_outputs(data)` to emit a rank-ordered LaTeX `tabular` per `feedback_table_rank_order.md` memory.**
  Add at the end of `generate_outputs(data)`:
  ```python
  rows = [
      ("Optimal (DP)",         data["optimal_value"],     None),
      ("Murphy backward rec.", data["murphy_value"],      data.get("murphy_se")),
      ("Q-learning",           data["qlearning_mean"],    data["qlearning_se"]),
  ]
  rows.sort(key=lambda r: -r[1])  # highest value first

  lines = [
      r"\begin{tabular}{lcc}",
      r"\toprule",
      r"Method & Policy value $V^\pi(s_0)$ & SE \\",
      r"\midrule",
  ]
  for name, mean, se in rows:
      se_str = f"{se:.3f}" if se is not None else "--"
      lines.append(f"{name} & {mean:.3f} & {se_str} \\\\")
  lines += [r"\bottomrule", r"\end{tabular}", ""]

  out_path = Path(__file__).parent / "dtr_qlearning_vs_murphy_results.tex"
  out_path.write_text("\n".join(lines))
  print(f"Wrote {out_path}")
  ```
  Adjust the keys to match what the existing `data` dict actually contains. If `data` lacks one of these fields, do not invent values — read the sim source to find the right key.

- [ ] **Step 3: Regenerate from cache without re-running the heavy compute.**
  ```bash
  python3 ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py --plots-only \
      > ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt 2>&1
  ```
  Expected: exit 0; new `.tex` file printed.

- [ ] **Step 4: Verify file exists and parses.**
  ```bash
  ls -la ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex
  cat ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex
  ```
  Expected: tabular block, rank-ordered by policy value.

- [ ] **Step 5: Locate the right spot in `rl_for_ci.tex` to `\input` the table.**
  ```bash
  grep -n "dtr_qlearning_vs_murphy\|Dynamic Treatment\|backward recursion" ch10b_rl_for_ci/tex/rl_for_ci.tex
  ```
  Expected: an existing figure or result paragraph in the DTR section.

- [ ] **Step 6: Insert `\input{../ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex}` inside a `table` float in `rl_for_ci.tex`, near the existing DTR figure, with caption "Q-learning vs. Murphy backward recursion on the synthetic DTR (rank-ordered by policy value)."** Use the Edit tool with the exact surrounding context to avoid match ambiguity.

- [ ] **Step 7: Recompile the chapter PDF.**
  ```bash
  cd docs && pdflatex -shell-escape -jobname=ch10b_rl_for_ci \
      "\def\chapterfile{../ch10b_rl_for_ci/tex/rl_for_ci}\input{compile_chapter}" \
      && bibtex ch10b_rl_for_ci \
      && pdflatex -shell-escape -jobname=ch10b_rl_for_ci \
      "\def\chapterfile{../ch10b_rl_for_ci/tex/rl_for_ci}\input{compile_chapter}" \
      && pdflatex -shell-escape -jobname=ch10b_rl_for_ci \
      "\def\chapterfile{../ch10b_rl_for_ci/tex/rl_for_ci}\input{compile_chapter}" \
      && cd ..
  ```
  Expected: clean compile, `docs/ch10b_rl_for_ci.pdf` regenerated. New table visible on visual inspection (open the PDF).

- [ ] **Step 8: Commit.**
  ```bash
  git add ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py \
          ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex \
          ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_stdout.txt \
          ch10b_rl_for_ci/tex/rl_for_ci.tex \
          docs/ch10b_rl_for_ci.pdf
  git commit -m "$(cat <<'EOF'
  feat(ch10b): emit dtr_qlearning_vs_murphy results table

  Rank-ordered tabular of policy value V^pi(s_0) for Optimal (DP),
  Murphy backward recursion, and Q-learning. \input'd in the DTR
  results paragraph of rl_for_ci.tex.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 1.2: Refresh ch12 dyna_maze cache

**Files:**
- Re-run: `ch12_world_models/sims/dyna_maze.py`
- Regenerate: `ch12_world_models/sims/dyna_maze_stdout.txt`, `dyna_maze.png`, `dyna_maze_results.tex` (the existing names — verify)
- Possibly modify: `ch12_world_models/tex/s03_dyna_q.tex` if headline numbers drift
- Modify: `ch12_world_models/CHAPTER_NOTES.md` (drop stale-cache open thread)

Per memory `feedback_update_stdout.md`: always update `_stdout.txt` and recompile after a sim change.

- [ ] **Step 1: Capture current headline numbers from the tex before re-running.**
  ```bash
  grep -E "Dyna-Q|Q-Learn|Schmidhuber" ch12_world_models/tex/s03_dyna_q.tex | head -20
  grep -E "Dyna-Q|Q-Learn|Schmidhuber" ch12_world_models/sims/dyna_maze_stdout.txt | head -20
  ```
  Record both for comparison after re-run.

- [ ] **Step 2: Identify cache invalidation scope.**
  ```bash
  ls ch12_world_models/sims/cache/dyna_maze_* 2>/dev/null
  head -5 ch12_world_models/sims/dyna_maze.py | grep -i "ALGO_REGISTRY\|train_" || \
       grep -n "ALGO_REGISTRY\|def train_" ch12_world_models/sims/dyna_maze.py
  ```
  Expected: per-algo cache files for Dyna-Q, Dyna-Q+, Schmidhuber, Q-Learn (plus shared.pkl).

- [ ] **Step 3: Force re-run only the agents whose code changed (Dyna-Q+ with untried-action registration; Schmidhuber with reward head — per CHAPTER_NOTES.md).**
  ```bash
  python3 ch12_world_models/sims/dyna_maze.py \
      --force "Dyna-Q+,Schmidhuber" \
      > ch12_world_models/sims/dyna_maze_stdout.txt 2>&1
  echo "exit: $?"
  tail -40 ch12_world_models/sims/dyna_maze_stdout.txt
  ```
  Expected: exit 0; tail shows the new mean/SE numbers per agent.

  If the sim script uses a different forcing flag name, fall back to `--force shared` to invalidate all caches and re-run from scratch (slower but always correct).

- [ ] **Step 4: Diff new headline numbers against the tex.**
  ```bash
  grep -E "Dyna-Q|Q-Learn|Schmidhuber" ch12_world_models/sims/dyna_maze_stdout.txt | head -20
  ```
  Compare against the numbers captured in Step 1. If numbers drifted by more than the printed SE, update `ch12_world_models/tex/s03_dyna_q.tex` with the new values using the Edit tool, preserving rank order per `feedback_table_rank_order.md`.

- [ ] **Step 5: Recompile ch12 PDF.**
  ```bash
  cd docs && pdflatex -shell-escape -jobname=ch12_world_models \
      "\def\chapterfile{../ch12_world_models/tex/world_models}\input{compile_chapter}" \
      && bibtex ch12_world_models \
      && pdflatex -shell-escape -jobname=ch12_world_models \
      "\def\chapterfile{../ch12_world_models/tex/world_models}\input{compile_chapter}" \
      && pdflatex -shell-escape -jobname=ch12_world_models \
      "\def\chapterfile{../ch12_world_models/tex/world_models}\input{compile_chapter}" \
      && cd ..
  ```
  Expected: clean compile.

- [ ] **Step 6: Update `CHAPTER_NOTES.md`.** Remove the "cache may be stale" open-thread line (typically near the end under "Open threads"). Use Edit tool with the exact original line as `old_string`.

- [ ] **Step 7: Commit.**
  ```bash
  git add ch12_world_models/sims/dyna_maze.py \
          ch12_world_models/sims/dyna_maze_stdout.txt \
          ch12_world_models/sims/dyna_maze.png \
          ch12_world_models/sims/dyna_maze_results.tex \
          ch12_world_models/tex/s03_dyna_q.tex \
          ch12_world_models/CHAPTER_NOTES.md \
          docs/ch12_world_models.pdf
  # Add any cache files git tracks (likely gitignored; ignore failure):
  git add ch12_world_models/sims/cache/ 2>/dev/null || true
  git status --short
  git commit -m "$(cat <<'EOF'
  refresh(ch12): re-run dyna_maze with updated Dyna-Q+ / Schmidhuber agents

  Cache invalidation after agent code changes (Dyna-Q+ untried-action
  registration; Schmidhuber reward head). Updates tex headline numbers
  if they drifted from cached run.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 1.3: Decide fate of decision_focused_ar1.py

**Files:**
- Inspect: `ch12_world_models/sims/decision_focused_ar1.py`, `..._stdout.txt`, `.png`, `_table.tex`
- Either modify `ch12_world_models/tex/world_models.tex` (or a section file) to integrate the sim, or move the sim files into `archive/`.

This is the only task in the plan that requires a judgment call. Default rule: if the sim demonstrates a topic NOT already covered by the three core sims (dyna_maze, cobweb, fishery), wire it in as an additional simulation; otherwise archive it.

- [ ] **Step 1: Read the script header and stdout to determine scope.**
  ```bash
  head -40 ch12_world_models/sims/decision_focused_ar1.py
  head -30 ch12_world_models/sims/decision_focused_ar1_stdout.txt
  ```
  Expected: header describes "decision-focused learning on AR(1)" — a topic about end-to-end optimization of forecast-driven decisions.

- [ ] **Step 2: Check for any latent reference in the chapter tex.**
  ```bash
  grep -rn "decision_focused\|decision-focused\|AR(1)\|AR1" ch12_world_models/tex/ docs/main.tex
  ```
  Expected: no references — confirms orphan status.

- [ ] **Step 3: Decision branch.**
  - **If the sim covers decision-focused learning (DFL) and there is no chapter coverage of DFL elsewhere:** add a short subsection §11 to `ch12_world_models/tex/` titled "Decision-Focused Learning" with one paragraph of motivation, one paragraph of results referencing the sim figure and table. Skip if any other chapter (e.g. ch10b) covers DFL.
  - **Otherwise:** move the orphan files to `archive/ch12_decision_focused_ar1/`.
  
  Default action absent contrary evidence: **archive**. Reasoning: this plan optimizes for arxiv push speed; new content additions are out of scope unless required. The user can revisit DFL coverage in a future cycle.

- [ ] **Step 4 (archive branch): Move files.**
  ```bash
  mkdir -p archive/ch12_decision_focused_ar1
  git mv ch12_world_models/sims/decision_focused_ar1.py \
         archive/ch12_decision_focused_ar1/
  git mv ch12_world_models/sims/decision_focused_ar1_stdout.txt \
         archive/ch12_decision_focused_ar1/
  git mv ch12_world_models/sims/decision_focused_ar1.png \
         archive/ch12_decision_focused_ar1/
  git mv ch12_world_models/sims/decision_focused_ar1_table.tex \
         archive/ch12_decision_focused_ar1/
  ```

- [ ] **Step 5 (archive branch): Document in CHAPTER_NOTES.md.** Add a one-line "Archived: decision_focused_ar1 (2026-05-19) — orphan, not referenced in world_models.tex" at the bottom under an "Archived sims" heading.

- [ ] **Step 6: Commit.**
  ```bash
  git add archive/ch12_decision_focused_ar1/ ch12_world_models/CHAPTER_NOTES.md
  git commit -m "$(cat <<'EOF'
  archive(ch12): move orphan decision_focused_ar1 sim out of ch12

  Sim produced figure/table/stdout but was never referenced in
  world_models.tex. Preserved under archive/ in case it becomes
  relevant for a future DFL section.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 1.4: Verify ch10_causal counterfactual_ope tex refs are correct

This is a check, not a change. The Phase 0 audit reported the paths are already correct (`../ch10_causal/sims/`). Confirm.

- [ ] **Step 1: Verify refs.**
  ```bash
  grep -n "counterfactual_ope" ch10_causal/tex/causal_rl.tex
  ```
  Expected output lines (313 and 318 per audit):
  ```
  ../ch10_causal/sims/counterfactual_ope_table.tex
  ../ch10_causal/sims/counterfactual_ope.png
  ```

- [ ] **Step 2: Confirm files exist at those paths.**
  ```bash
  ls -la ch10_causal/sims/counterfactual_ope_table.tex \
         ch10_causal/sims/counterfactual_ope.png
  ```
  Expected: both files exist.

- [ ] **Step 3: Recompile ch10 chapter as a sanity check.**
  ```bash
  cd docs && pdflatex -shell-escape -jobname=ch10_causal \
      "\def\chapterfile{../ch10_causal/tex/causal_rl}\input{compile_chapter}" \
      && bibtex ch10_causal \
      && pdflatex -shell-escape -jobname=ch10_causal \
      "\def\chapterfile{../ch10_causal/tex/causal_rl}\input{compile_chapter}" \
      && pdflatex -shell-escape -jobname=ch10_causal \
      "\def\chapterfile{../ch10_causal/tex/causal_rl}\input{compile_chapter}" \
      && cd ..
  ```
  Expected: clean compile.

- [ ] **Step 4: Commit refreshed ch10 PDF.**
  ```bash
  git add docs/ch10_causal.pdf
  git diff --cached --stat
  git commit -m "$(cat <<'EOF'
  build: recompile ch10_causal PDF after counterfactual_ope move

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  If the PDF didn't change after Phase 0's earlier recompile, this commit will be empty — skip in that case.

---

## Phase 2 — Sim standardization sweep (ch02–ch05)

Per CLAUDE.md "Color Standards": every script must import `from sims.plot_style import apply_style, COLORS, ALGO_COLORS` and call `apply_style()`. A grep audit found **9 sims** in ch02–ch05 that do not.

**Non-conforming sims (verified via `grep -L "^from sims.plot_style"`):**
- `ch03_theory/sims/info_geometry_npg.py`
- `ch03_theory/sims/gridworld_study.py`
- `ch03_theory/sims/gridworld_algorithms.py`
- `ch03_theory/sims/mm_surrogate_trpo.py`
- `ch03_theory/sims/ssp_gridworld_20x20.py`
- `ch03_theory/sims/theory_validation.py`
- `ch03_theory/sims/lqr_convergence.py`
- `ch03b_deeprl_practice/sims/bellman_vs_return.py`
- `ch04_control_problems/sims/econ_benchmark.py`

For each: minimal change — add the import block at the top of the file and call `apply_style()` once before any plotting code. Do NOT rewrite plotting calls unless a script hardcodes hex colors that conflict with the palette.

Reference standard from CLAUDE.md "Color Standards" section:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS
apply_style()
```

### Task 2.1–2.9: Standardize one sim per task

The nine tasks follow an identical template. They are listed individually so the autorun loop can checkpoint between them.

**Template (Task 2.k for k = 1..9):**

**Files:**
- Modify: `<sim_path>` (one of the nine listed above)

- [ ] **Step 1: Read the top of the file to find the imports block.**
  ```bash
  head -30 <sim_path>
  ```

- [ ] **Step 2: Add the plot_style import block immediately after the existing imports.**
  Use Edit with `old_string` being the last existing import line and `new_string` being that line plus the four-line plot_style block. If `apply_style()` was already called elsewhere in the script for some reason, do not add a second call.

- [ ] **Step 3: Search for hardcoded hex colors or `'C0'`/`'C1'`/... shorthand that conflict with the palette.**
  ```bash
  grep -nE "#[0-9a-fA-F]{6}|color=['\"]C[0-9]" <sim_path>
  ```
  If matches exist, replace with `COLORS['blue']`, `COLORS['red']`, `ALGO_COLORS['Q-Learning']`, etc. as appropriate. If `plot_style.py` lacks a needed key, use `COLORS['black']` as fallback rather than reintroducing a hex literal.

- [ ] **Step 4: Re-run the sim from cache (no compute) to verify plots regenerate with the palette.**
  ```bash
  python3 <sim_path> --plots-only > <sim_path_no_ext>_stdout.txt 2>&1 || \
       python3 <sim_path> > <sim_path_no_ext>_stdout.txt 2>&1
  echo exit: $?
  ```
  Some older sims may not support `--plots-only`; the fallback re-runs everything. Either is acceptable here.

- [ ] **Step 5: Visually inspect the regenerated PNG (open the file). If colors look broken, revert and skip — the sim isn't conformant enough for an in-place upgrade. Note that sim in CHAPTER_NOTES.md (or in `docs/superpowers/audits/2026-05-19-plot-style-skipped.md`) for a future cleanup cycle.**

- [ ] **Step 6: Recompile the affected chapter PDF only if a figure changed.**
  ```bash
  cd docs && pdflatex -shell-escape -jobname=<chXX> \
      "\def\chapterfile{<chapter_tex_path>}\input{compile_chapter}" && cd ..
  ```

- [ ] **Step 7: Commit.**
  ```bash
  git add <sim_path> <sim_path_no_ext>_stdout.txt <sim_path_no_ext>.png 2>/dev/null
  git commit -m "$(cat <<'EOF'
  style(<chXX>): adopt plot_style.apply_style in <sim_name>

  Brings the script onto the centralized palette required by
  CLAUDE.md "Color Standards".

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

**Concrete task list (auto-loop iterates):**
- Task 2.1: `ch03_theory/sims/info_geometry_npg.py`
- Task 2.2: `ch03_theory/sims/gridworld_study.py`
- Task 2.3: `ch03_theory/sims/gridworld_algorithms.py`
- Task 2.4: `ch03_theory/sims/mm_surrogate_trpo.py`
- Task 2.5: `ch03_theory/sims/ssp_gridworld_20x20.py`
- Task 2.6: `ch03_theory/sims/theory_validation.py`
- Task 2.7: `ch03_theory/sims/lqr_convergence.py`
- Task 2.8: `ch03b_deeprl_practice/sims/bellman_vs_return.py`
- Task 2.9: `ch04_control_problems/sims/econ_benchmark.py`

### Task 2.10: Recompile affected chapter PDFs in a batch

After all nine standardization commits land:

- [ ] **Step 1: Recompile ch03_theory, ch03b_deeprl_practice, ch04_control_problems PDFs.**
  ```bash
  for ch in ch03_theory ch03b_deeprl_practice ch04_control_problems; do
      tex_main=$(ls $ch/tex/*.tex | head -1)  # main tex file per chapter
      cd docs && pdflatex -shell-escape -jobname=$ch \
          "\def\chapterfile{../$tex_main}\input{compile_chapter}" \
          && bibtex $ch \
          && pdflatex -shell-escape -jobname=$ch \
          "\def\chapterfile{../$tex_main}\input{compile_chapter}" \
          && pdflatex -shell-escape -jobname=$ch \
          "\def\chapterfile{../$tex_main}\input{compile_chapter}" \
          && cd ..
  done
  ```
  Note: the `tex_main` autodetect picks the first `.tex` file alphabetically. Verify for each chapter that this is the correct main file before relying on it; if not, hardcode the path.

- [ ] **Step 2: Commit the regenerated PDFs.**
  ```bash
  git add docs/ch03_theory.pdf docs/ch03b_deeprl_practice.pdf docs/ch04_control_problems.pdf
  git status --short
  git commit -m "$(cat <<'EOF'
  build: recompile ch03/ch03b/ch04 PDFs after plot_style sweep

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Phase 3 — Pre-arxiv audit gates

### Task 3.1: Bib coverage audit

**Files:**
- Read-only.
- Output: `docs/superpowers/audits/2026-05-19-bib-coverage.md`

- [ ] **Step 1: Dispatch the `bib-coverage-auditor` agent.**
  Use the Agent tool with `subagent_type=bib-coverage-auditor` and this prompt:
  ```
  Audit cited-vs-defined coverage between the LaTeX sources in
  /Users/pranjal/Code/rl and the bib file at docs/refs.bib (plus
  docs/refs_extended.bib if it exists). Tex sources: docs/main.tex
  is the master; it \inputs chapter files in ch00_*/tex/, ch01_*/tex/,
  ..., ch99_*/tex/. The journals/ folder is OUT OF SCOPE — skip it
  entirely.

  Report:
  1. Citations in tex with no entry in either bib file (CITED-BUT-MISSING)
  2. Bib entries with no citation anywhere in tex (DEFINED-BUT-ORPHAN)
  3. Duplicate keys across the two bib files
  4. Entries missing required fields (year for any; author for article/inproceedings)

  Write the report as Markdown to
  /Users/pranjal/Code/rl/docs/superpowers/audits/2026-05-19-bib-coverage.md
  ```

- [ ] **Step 2: Read the report.**
  ```bash
  ls -la docs/superpowers/audits/2026-05-19-bib-coverage.md
  wc -l docs/superpowers/audits/2026-05-19-bib-coverage.md
  ```

- [ ] **Step 3: Triage CITED-BUT-MISSING items.** For each missing key, either: (a) add a verified entry to `docs/refs.bib`, OR (b) replace the cite in tex with a verified key from the bib. Do NOT invent bib entries — use the `zotero-bib` skill's `lookup` subcommand to fetch real BibTeX from Zotero or CrossRef. If a citation cannot be sourced, remove it from the tex and replace with the proper attribution in prose. This step may produce 0 or many additional commits depending on what the audit finds.

- [ ] **Step 4: Triage DEFINED-BUT-ORPHAN items.** Orphans are not blockers for arxiv (bibtex omits them from the final .bbl), but flag them in the audit report for future cleanup. No action this cycle.

- [ ] **Step 5: Commit audit report + any bib fixes.**
  ```bash
  git add docs/superpowers/audits/2026-05-19-bib-coverage.md docs/refs.bib 2>/dev/null
  git diff --cached --stat
  git commit -m "$(cat <<'EOF'
  audit(refs): bib coverage report + missing-key fixes

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  If no fixes were needed, commit just the report.

### Task 3.2: Paper coherence audit

**Files:**
- Read-only across the survey.
- Output: `docs/superpowers/audits/2026-05-19-paper-coherence.md`

- [ ] **Step 1: Dispatch the `paper-coherence-auditor` agent** with prompt:
  ```
  Audit the survey paper draft at /Users/pranjal/Code/rl for
  pre-arxiv coherence. Compile entry is docs/main.tex. Out of scope:
  journals/, thesis/, thesis_v2/, ORE_main/, archive/.

  Three-section audit:
  1. Abstract <-> Conclusion: do the claims in ch00 introduction
     match the closing claims in ch99 conclusion?
  2. Figure <-> claim: for each \ref{fig:...} in chapter prose,
     confirm the cited figure actually shows what the prose claims.
     Flag any figure cited in a way that contradicts its caption.
  3. Method reproducibility: for each simulation writeup, confirm
     (state space, action space, reward, hyperparameters, seeds,
     episode count) are all stated. Flag any sim where a reader
     could not replicate.

  Write the report as Markdown to
  /Users/pranjal/Code/rl/docs/superpowers/audits/2026-05-19-paper-coherence.md
  with specific line and figure references.
  ```

- [ ] **Step 2: Read the report and triage findings.** For each issue: if it's a small wording fix or missing hyperparameter mention, fix inline with Edit. If it requires substantive re-writing, log a follow-up task in the audit report under "Deferred to next cycle" — do not block the arxiv push for non-critical findings.

- [ ] **Step 3: Commit fixes + report.**
  ```bash
  git add docs/superpowers/audits/2026-05-19-paper-coherence.md \
          $(git diff --name-only)
  git commit -m "$(cat <<'EOF'
  audit(coherence): pre-arxiv coherence report + inline fixes

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 3.3: ArXiv-readiness scan

**Files:**
- Output: `arxiv-check_2026-05-19.md` (emitted by the skill at its default location)

- [ ] **Step 1: Invoke the `arxiv-check` skill** via the Skill tool with skill name `arxiv-check`. The skill runs three parallel subagents (meta-comment / placeholder residue, reference verification against CrossRef + Semantic Scholar + arXiv, .tex/.bib citation drift). Pass it the entry point `docs/main.tex`.

- [ ] **Step 2: Read the emitted report and triage.** Critical findings (LLM meta-comments, hallucinated references, placeholder text) MUST be fixed before arxiv push. Non-critical findings (style nits) can be deferred.

- [ ] **Step 3: Move the report into the audits directory and commit.**
  ```bash
  mv arxiv-check_2026-05-19.md docs/superpowers/audits/ 2>/dev/null
  git add docs/superpowers/audits/arxiv-check_2026-05-19.md \
          $(git diff --name-only)
  git commit -m "$(cat <<'EOF'
  audit(arxiv): pre-submission arxiv-check report + critical fixes

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Phase 4 — Final compile + arxiv package

### Task 4.1: Full clean recompile of main.tex

- [ ] **Step 1: Wipe compile artifacts.**
  ```bash
  cd docs && rm -f main.aux main.bbl main.blg main.log main.out main.toc \
                   $(ls *.aux *.bbl *.blg 2>/dev/null | grep -v main) && cd ..
  ```

- [ ] **Step 2: Run the full four-pass compile.**
  ```bash
  cd docs && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && bibtex main \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && pdflatex -shell-escape -interaction=nonstopmode main.tex \
            && cd ..
  echo exit: $?
  ```

- [ ] **Step 3: Confirm zero undefined warnings.**
  ```bash
  grep -E "(LaTeX Warning: Reference|LaTeX Warning: Citation).*undefined" \
       docs/main.log | wc -l
  ```
  Expected: 0. If non-zero, fix iteratively (use `latex-build-summarizer` agent if the list is long).

- [ ] **Step 4: Capture page count + size.**
  ```bash
  pdfinfo docs/main.pdf | grep -E "Pages|File size"
  ```

- [ ] **Step 5: Commit final PDF.**
  ```bash
  git add docs/main.pdf
  git commit -m "$(cat <<'EOF'
  build: final main.pdf for arxiv submission

  Clean four-pass compile, zero undefined references/citations.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 4.2: Build the arxiv package

- [ ] **Step 1: Check the packaging script exists and is current.**
  ```bash
  ls -la scripts/package_arxiv.sh
  head -20 scripts/package_arxiv.sh
  ```
  Expected: script present, references `ch12_world_models/` (post-rename) not `ch12_forecasting_rl/`.

- [ ] **Step 2: Run it.**
  ```bash
  bash scripts/package_arxiv.sh > /tmp/arxiv_package_log.txt 2>&1
  echo exit: $?
  tail -30 /tmp/arxiv_package_log.txt
  ls -la arxiv_submission*/ arxiv_submission*.tar.gz 2>/dev/null | head
  ```
  Expected: a `.tar.gz` produced. If the script fails, dispatch a `code-reviewer` agent on `scripts/package_arxiv.sh` to identify the breakage; fix and re-run.

- [ ] **Step 3: Smoke-test the arxiv tarball.**
  ```bash
  mkdir -p /tmp/arxiv_smoke && cd /tmp/arxiv_smoke && \
      tar xzf $(ls -t /Users/pranjal/Code/rl/arxiv_submission*.tar.gz | head -1) && \
      ls main.tex refs.bib && \
      pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/smoke_pass1.log 2>&1 && \
      bibtex main > /tmp/smoke_bibtex.log 2>&1 && \
      pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/smoke_pass2.log 2>&1 && \
      pdflatex -shell-escape -interaction=nonstopmode main.tex > /tmp/smoke_pass3.log 2>&1 && \
      ls -la main.pdf && \
      cd /Users/pranjal/Code/rl
  ```
  Expected: `main.pdf` exists in `/tmp/arxiv_smoke/` and is reasonably similar in page count to `docs/main.pdf`.

- [ ] **Step 4: Commit packaging artifacts that git tracks (if any).**
  ```bash
  git status --short
  # The .tar.gz and arxiv_submission/ folder are likely gitignored.
  # If not, decide per-file whether to commit; arxiv tarballs are
  # generally NOT committed.
  ```
  Likely no commit needed here.

### Task 4.3: Final summary report

- [ ] **Step 1: Generate the end-of-cycle report.** Write a short Markdown summary to `docs/superpowers/audits/2026-05-19-arxiv-push-summary.md` covering:
  - Commits added on this branch (count + one-line list from `git log --oneline main..HEAD`)
  - Final main.pdf page count and file size
  - Audit results: counts of fixes from bib-coverage / paper-coherence / arxiv-check
  - Any deferred items (orphan bib entries, non-conforming sims that couldn't be auto-fixed, deferred coherence findings)
  - The arxiv tarball path (if produced)

- [ ] **Step 2: Commit the summary.**
  ```bash
  git add docs/superpowers/audits/2026-05-19-arxiv-push-summary.md
  git commit -m "$(cat <<'EOF'
  docs: end-of-cycle summary for arxiv main-article push

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 3: Final report to user.** Print to console (or write a top-level message) the summary plus "Ready to merge `humanize-pass` → `main` and arxiv-push. Tarball at `<path>`. Plan complete." Stop. Do not push to remote without explicit user confirmation per CLAUDE.md "Executing actions with care".

---

## Verification (end-to-end)

After the full plan runs:

- `git status` reports a clean working tree on branch `humanize-pass`.
- `git log --oneline main..HEAD` shows ~20–30 small commits with sensible messages.
- `docs/main.pdf` mtime is current; `pdfinfo` reports ~140 pages.
- `grep -c "LaTeX Warning: Reference.*undefined" docs/main.log` returns `0`.
- `grep -c "LaTeX Warning: Citation.*undefined" docs/main.log` returns `0`.
- `ls docs/superpowers/audits/` contains the four 2026-05-19 reports.
- `ls ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex` succeeds.
- `ch12_world_models/sims/dyna_maze_stdout.txt` mtime is post-cache-refresh.
- No sim in `ch02_rl_algorithms/sims/`, `ch03_theory/sims/`, `ch03b_deeprl_practice/sims/`, or `ch04_control_problems/sims/` is missing the `from sims.plot_style import apply_style` import. Verify:
  ```bash
  grep -L "^from sims.plot_style" ch02_rl_algorithms/sims/*.py ch03_theory/sims/*.py \
       ch03b_deeprl_practice/sims/*.py ch04_control_problems/sims/*.py 2>/dev/null
  ```
  Expected: empty output, or only files that were explicitly flagged "skip" with a written reason in the audit summary.
- Arxiv smoke compile in `/tmp/arxiv_smoke/` produced a `main.pdf` of similar page count to `docs/main.pdf`.

---

## Out of scope (explicit non-goals)

These were considered and dropped from this cycle. Listed here so the autorun loop does NOT add them:

- Any work under `journals/` — the user explicitly excluded this scope. Tracked separately at `~/.claude/plans/create-a-new-folder-abstract-ocean.md`.
- Any new chapter content (e.g. ch99 expansion — already substantively done; ch09 RLHF deepening — already done; ch07 RLHF sim — already done in `ch09_rlhf/sims/`).
- ch02 tex consolidation between `planning_learning.tex` and `..._alt.tex` — moot, the variants no longer exist in ch02.
- "Economic Models for RL" deferred chapter — placement decision deferred again; no folder created.
- ORE_main sister survey — read-only reference; do not edit.
- thesis/ and thesis_v2/ — separate submission targets; out of scope for the arxiv push.
- Pushing to `origin` or merging to `main` — never automatic. Final user action.

---

## Execution notes for the autorun loop

- The plan groups commits per task, not per step. The loop should checkpoint between tasks (review the diff with the user, optionally), but it MAY proceed task-to-task autonomously since the user explicitly requested "autorun over time (will take a long time and i dont care)."
- If any task fails verification, the loop should HALT and surface the failure rather than push past it. Examples: compile fails, sim returns non-zero exit, audit agent reports a Bullshit-score >= 50% on any sim per the CLAUDE.md Simulation Audit framework.
- Per CLAUDE.md "Working Behavior": never declare victory. Report what happened factually at each task boundary.
- Per memory `feedback_always_show_pdf.md`: every tex-touching task ends with a recompile and a path to the regenerated PDF.
- Per memory `feedback_update_stdout.md`: every sim-touching task regenerates `_stdout.txt`.
- Per memory `feedback_table_rank_order.md`: any new result table is rank-ordered by performance.
- Per memory `feedback_one_sim_at_a_time.md`: Phase 2 standardization processes one sim per task; do not batch.

## Self-review notes

Coverage check vs the open items identified during exploration:
- Commit staged work ✓ (Tasks 0.2, 0.3)
- Baseline main.tex compile ✓ (Task 0.3)
- Add dtr_qlearning_vs_murphy results table ✓ (Task 1.1)
- Refresh dyna_maze cache ✓ (Task 1.2)
- Decide decision_focused_ar1 fate ✓ (Task 1.3, archive default)
- Verify counterfactual_ope refs ✓ (Task 1.4)
- Standardize 9 ch02–ch05 sims ✓ (Tasks 2.1–2.9 + 2.10 recompile)
- Bib audit ✓ (Task 3.1)
- Coherence audit ✓ (Task 3.2)
- ArXiv-check ✓ (Task 3.3)
- Final compile + package ✓ (Tasks 4.1, 4.2, 4.3)

Placeholder scan: no "TBD", "TODO later", "add appropriate error handling" markers. All code snippets are concrete. All file paths are exact.

Type consistency: function names, file paths, and data-dict keys referenced in later tasks match what earlier tasks created.
