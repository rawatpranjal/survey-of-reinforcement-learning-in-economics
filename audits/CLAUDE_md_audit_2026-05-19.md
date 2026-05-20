**Bloat score: 55%** — The Tasklist subsection is stale across the board: item 1 directs new work into `ch07_rlhf/` (a folder that does not exist; the RLHF chapter is `ch09_rlhf/` and already holds the requested DPO scripts), and item 4 names two `ch02` tex variants that no longer exist.

# CLAUDE.md Audit — /Users/pranjal/Code/rl/CLAUDE.md

Audit date: 2026-05-19. Target: 37,321 bytes, 446 lines. Project-level file; loads once per session whenever cwd is `~/Code/rl`.

## 1. Bloat score

See first line above. The score sits at 55% rather than 25-50% because the Tasklist directs the agent to act on a folder that no longer exists (`ch07_rlhf/sims/`) and to consolidate two files that no longer exist (`planning_learning.tex`, `planning_learning_alt.tex`). Per the audit rules, a directive pointing at a missing path floors the score at 50%; the additional stale tasklist items and the dead `refs_extended.bib` symlink claim push it to 55%.

## 2. Section table

| Section title | Line range | Bytes | Verdict | One-line reason |
|---|---|---|---|---|
| What This Is | 3–7 | 312 | LOAD-BEARING | States project identity (arXiv survey, PhD thesis component, advisor) not derivable from any file. |
| Sister Survey (ORE_main) | 9–19 | 717 | LOAD-BEARING | Defines the do-not-replicate boundary against `ORE_main/`, which is a live sibling repo. |
| Master Plan | 21–166 | ~9,400 | LOAD-BEARING | Holds the canonical chapter map; but the `Tasklist` subsection (142–166) is stale — see drift flags. |
| Writing Style | 168–227 | ~5,600 | LOAD-BEARING | Repo-specific prose constraints (no em dashes, no `\textbf{}`, notation conventions) that override default behavior. |
| Simulation Standards | 229–400 | ~13,300 | LOAD-BEARING | Repo-specific sim contract (caching API, audit checklist, color palette); the largest section but each rule is enforceable and repo-specific. |
| Working Behavior | 402–413 | 1,737 | NICE-TO-HAVE | "Read papers thoroughly", "verify before reporting", "show, don't tell" largely restate generic agent discipline; only the journal-drafting bullet (413) and the no-victory-declaration norm add repo-specific value. |
| File Tracking | 415–421 | 1,069 | LOAD-BEARING | Maps `changelog.md`, `bloat.md`, `journal_target.md`, the restructure plan, and the journal plan; all live links. |
| Key Commands | 423–446 | 1,300 | LOAD-BEARING | Repo-specific build invocations (`compile_chapter` jobname trick, `run_all_sims.py` flags); not derivable without reading multiple scripts. |

No section is fully DEAD, so the score is not floored by a dead section; it is floored by the drift directives inside `Master Plan`.

## 3. Redundancy map

- "Read papers thoroughly" / verify-against-papers: stated twice. `CLAUDE.md:68` — "Before implementing any algorithm or running benchmarks, verify formulations against reference papers in the chapter's `papers/` directory" — and `CLAUDE.md:406` — "When asked to 'read a paper' ... perform a deep read of the actual content." Same instruction, two sections (Master Plan / Working Behavior).
- "tables first, prose second" for simulation results: stated twice. `CLAUDE.md:196` — "For simulation results and experiments: tables first, prose second" — and `CLAUDE.md:351` — "**Copious tables.** Print parameter sweeps, results grids, and validation metrics in tabular format." Prose-style section and Stdout-format section overlap.
- "No subjective commentary / state facts not judgments": stated three times. `CLAUDE.md:199` — "State facts objectively. Do not give comments or opinions" — `CLAUDE.md:350` — "**No opinions, only facts.**" — and `CLAUDE.md:409` — "**State facts, not judgments.**" Three sections (Prose Style, Stdout Output Format, Working Behavior) carry the same rule.
- Journal-drafting "load the plan first" directive: stated twice verbatim in intent. `CLAUDE.md:150` (Tasklist item 3) — "Draft journal-version content per the plan at `/Users/pranjal/.claude/plans/create-a-new-folder-abstract-ocean.md`" — and `CLAUDE.md:413` (Working Behavior) — "load the plan at `/Users/pranjal/.claude/plans/create-a-new-folder-abstract-ocean.md` first." Plus a third reference at `CLAUDE.md:420` (File Tracking).
- "Always compile after editing .tex": stated twice. `CLAUDE.md:446` — "Always compile after modifying any `.tex` file and show the PDF output path" — and the same intent embedded in the chapter-compile command block at `CLAUDE.md:429-431`.

## 4. Drift flags

- `CLAUDE.md:146` — "Output figures and a comparison table saved to `ch07_rlhf/sims/`." BROKEN. No `ch07_rlhf/` folder exists; `ls` confirms only `ch07_bandits/` and `ch09_rlhf/`. The RLHF chapter is `ch09_rlhf/`, and `ch09_rlhf/sims/` already contains `job_search_dpo.py`, `rlhf_dpo_pipeline.py`, `preference_learning.py`, and ~10 more RLHF/DPO scripts. Tasklist item 1 directs the agent to create work that already exists, in a folder that does not.
- `CLAUDE.md:148` — "Expand Chapter 8 (Conclusion) from its current 6-line stub." STALE/WRONG. Chapter 8 is `ch08_offline_rl/` (Offline RL), not the conclusion; the conclusion is `ch99_conclusion/`, which already contains a `notes_for_conclusion.md`, `tex/`, and `sims/`. The chapter numbering in the tasklist predates the renaming documented elsewhere in the same file.
- `CLAUDE.md:154` — "Consolidate the two tex variants in Chapter 2 (`planning_learning.tex` and `planning_learning_alt.tex`)." BROKEN. `ls ch02_rl_algorithms/tex/` returns only `rl_algorithms.tex`; `find ch02_rl_algorithms -name 'planning_learning*'` returns nothing. Both named files are gone; the consolidation already happened.
- `CLAUDE.md:156` — "Deepen the RLHF chapter tex (Chapter 7)." STALE. The RLHF chapter is Chapter 9 (`ch09_rlhf/`) per the file's own table at line 40. The tasklist calls it Chapter 7.
- `CLAUDE.md:162` — "Standardize existing simulation notebooks in Chapters 2 through 5." STALE relative to `CLAUDE.md:232` which states the script convention is settled and "Existing notebooks in Chapters 2 through 5 remain as-is." The tasklist item and the Simulation Standards section give opposite directives on the same notebooks.
- `CLAUDE.md:138` — "**Shared assets (`journals/shared/`):** symlinks to `docs/refs.bib`, `docs/refs_extended.bib`, ..." BROKEN. `docs/refs_extended.bib` does not exist (`find docs journals -name 'refs_extended*'` returns nothing; git log shows commit `7df99c8 chore(refs): delete refs_extended.bib`). `ls -la journals/shared/` confirms no `refs_extended.bib` symlink remains. The file lists a deleted asset as a current symlink.
- `CLAUDE.md:419` — "Detailed restructure plan: `docs/plans/2026-01-27-repo-restructure.md`." LIVE but STALE: the file exists, last modified 2026-01-27 (113 days old). It is described as the active step-by-step task list; the chapter structure it planned has since been renamed multiple times (ch11→ch99, etc.), so it no longer matches the repo.
- `CLAUDE.md:51` — "An empty `ch03a_bm/` folder also exists from a prior reshuffle." PARTLY DRIFTED. `ch03a_bm/` is not empty; it contains a `sims/` subdirectory with 14 child entries. The "empty" claim is false.

## 5. Cross-reference health

- `/Users/pranjal/Code/rl/ORE_main` — LIVE (directory exists, sibling repo).
- `docs/main.tex` — LIVE (source of truth for chapter compile order; line 170 ch03a `\input` is commented out as stated).
- `docs/refs.bib` — LIVE.
- `docs/refs_extended.bib` — BROKEN (deleted in commit `7df99c8`; still referenced at line 138).
- `docs/econometrica.bst`, `docs/figs/`, `docs/glossary.tex`, `docs/compile_chapter.tex` — LIVE.
- `sims/plot_style.py`, `sims/sim_cache.py` — LIVE (shared modules referenced in Simulation Standards).
- `scripts/run_all_sims.py` — LIVE (Key Commands block).
- `scripts/package_arxiv.sh` — LIVE (referenced at line 55; git log `ff8572f` confirms active maintenance).
- `journals/build.sh`, `journals/shared/` — LIVE.
- `changelog.md` — LIVE (modified 2026-05-16, current).
- `bloat.md` — STALE (exists, last modified 2026-03-15, 65 days old; referenced as the live bloat guide at line 203 — under the 90-day threshold but aging).
- `journal_target.md` — STALE (exists, last modified 2026-05-07; current).
- `docs/plans/2026-01-27-repo-restructure.md` — STALE (exists, last modified 2026-01-27, 113 days old; >90-day threshold, content predates the chapter renames).
- `ORE_main/NOTATION_REVIEW.md` — STALE (exists, last modified 2026-01-27, 113 days old; >90-day threshold).
- `/Users/pranjal/.claude/plans/create-a-new-folder-abstract-ocean.md` — STALE (exists, last modified 2026-05-07; current, just under threshold).
- `/Users/pranjal/.claude/plans/yes-re-tireing-pass-first-zesty-shore.md` — STALE (exists, last modified 2026-05-11; current).
- `docs/superpowers/plans/2026-05-15-macro-rl-chapter-restructure.md` — LIVE (exists, recent).
- `thesis/`, `thesis_v2/` — LIVE (both directories exist).
- `archive/ch05_rl_as_behaviour/` — LIVE (exists, matches the "Archived to" claim at line 49).
- `https://github.com/rawatpranjal/aitools` — not referenced in this project CLAUDE.md (lives in the global file); not in scope.

## 6. Per-turn cost estimate

The file is 37,321 bytes, which at char/4 is roughly 9,330 tokens. As a project-level CLAUDE.md it loads once per session, not once per turn, so the cost is ~9,330 tokens per session for any session whose cwd is `~/Code/rl`. Across a typical multi-turn working session the per-turn amortized cost is negligible; the real cost is the one-time ~9,330-token load and the risk that the stale Tasklist misroutes early work before the agent reads the chapter table. No section is fully DEAD, so no whole-section deletion recovers tokens; the recoverable savings come from trimming the stale Tasklist (lines 142–166, ~2,000 bytes / ~500 tokens of which items 1, 4, 5, 7 are drift), the dead `refs_extended.bib` clause, and the three-way redundancy on "state facts not judgments." Tokens recovered if all DEAD sections deleted: 0 (0% of total) — no section is fully DEAD; the bloat is concentrated in stale directives within otherwise load-bearing sections, which this audit does not delete wholesale.

## 7. Cross-cutting recommendations

- Rewrite or delete the Tasklist subsection (lines 142–166). Items 1, 4, 5 reference chapter numbers and folders (`ch07_rlhf/`, `planning_learning.tex`) that the file's own chapter table contradicts; item 1's deliverable already exists in `ch09_rlhf/sims/`.
- Remove the `docs/refs_extended.bib` clause from line 138; the file was deleted in commit `7df99c8` and the symlink no longer exists in `journals/shared/`.
- Fix line 51: `ch03a_bm/` is not empty (it has a populated `sims/` subdirectory). Either describe it accurately or drop the mention.
- Collapse the three restatements of "state facts, not judgments" (lines 199, 350, 409) into one canonical location; same for the duplicated "verify against papers" (lines 68, 406) and "tables first" (lines 196, 351).
- Re-anchor or date-stamp the stale plan references: `docs/plans/2026-01-27-repo-restructure.md` and `ORE_main/NOTATION_REVIEW.md` are both >90 days old and predate the heavy chapter renaming; mark them historical or refresh them.

---

**Bloat score: 55%** — The Tasklist subsection is stale across the board: item 1 directs new work into `ch07_rlhf/` (a folder that does not exist; the RLHF chapter is `ch09_rlhf/` and already holds the requested DPO scripts), and item 4 names two `ch02` tex variants that no longer exist.
