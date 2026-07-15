# Paper-wide de-strain sweep

Status of the Phase 3 plain-construction pass (`docs/bloat.md`, rules P1-P6) across the
survey. `CLAUDE.md` declares plain construction a repo-wide prose rule, so this file records
which files have had the pass and what was deliberately not applied.

## Coverage

| Date | Scope | Result |
|---|---|---|
| 2026-07-14 | `ch12_world_models` | Done in commits `680d64f`, `ea25624`, `2a63fc9`. The P1-P6 rules were authored in `2a63fc9` itself, so no chapter predates them. |
| 2026-07-14 | 19 remaining prose files | Survey ("find") only. 67 clauses inventoried; no edits applied. |
| 2026-07-15 | 19 remaining prose files | Applied. 58 of 67 rows landed, 9 rejected on verification. |

## Method

The find/apply split matters and should be kept for any future pass.

1. **Find.** Reader agents flag strained clauses and propose a verbatim OLD -> NEW per row.
2. **Apply mechanically, never by agent.** A script matches OLD exactly and rejects any row
   that does not appear exactly once. Whitespace is normalised to a `\s+` run because the
   sources hard-wrap and a quoted clause often straddles a newline; every non-whitespace
   character must match. Rows are applied longest-OLD-first, since one row's OLD can be a
   substring of another's. An agent applying these rows can paraphrase, and that is exactly
   how the ch12 circular definition landed.
3. **Check invariants mechanically.** Multiset diff of citation keys, ref/label targets,
   numeric literals, math spans, and environments across removed vs added lines. Any
   asymmetry must be explained.
4. **Check meaning with a fresh agent, per file, never sampled.** One proposition per agent,
   handed the diff and not a paraphrase, not told what to conclude. This is the step the
   ch12 pass skipped, and it is the only step that catches the defects that matter. The
   invariant check passed ch12's circular definition without complaint.

Scripts used, kept for reuse: `apply_destrain.py`, `check_invariants.py`, `revert_failed.py`
(session scratchpad, 2026-07-15).

## Outcome, 2026-07-15

67 rows parsed, all 67 matched disk exactly, 67 applied, then 9 reverted after a fresh
per-file verifier read every diff. Net 58 rows across 19 files. Full build clean at 269pp
with zero undefined references or citations.

The 13 percent defect rate in the inventory is the headline finding. Reader agents proposing
plainer wording reliably produce rewrites that read better and assert something different.

## Rows rejected on verification (do not re-apply)

| File | Proposed rewrite | Why rejected |
|---|---|---|
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:250` | "the MABUC instance is" -> "MABUC is" | Category error. MABUC is a problem setting, not an algorithm. Line 341 runs three algorithms on it (vanilla TS, CCTS, TS_C); equating the setting with one of them makes that comparison and Table `tab:simB2_mabuc` incoherent. |
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:25` | "A parallel thread examines" -> "Related work examines" | Deletes the only occurrence of "thread", which is the antecedent for line 27. Also recategorises chapter content as prior literature, contradicting the next sentence. |
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:27` | "these two threads" -> "both" | Leaves a bare pronoun whose nearest antecedent is the wrong pair. Coupled to the row above. |
| `appA_preliminaries/tex/preliminaries.tex:325` | "are the workhorse of" -> "drive" | Grammar break. The sentence is a correlative on the noun ("the workhorse of X, and of Y"); the verb form leaves "and of ..." with no head noun, so the `\ref{sec:lcpo}` clause asserts nothing. |
| `ch04_control_problems/tex/applications.tex:115` | "a scale that defeats human specialists" -> "a scale too large for human specialists to manage" | Changes outperformed into infeasible, contradicting lines 123 and 140, where the specialists' historical actions are the pre-training demonstration data and the comparison baseline. |
| `ch03_theory/tex/planning_learning_v3.tex:695` | "Two questions remain important" -> "This subsection addresses two further questions" | False scope claim. The subsubsection it heads answers only the first question; the second is answered in the next sibling subsubsection. |
| `ch03_theory/tex/planning_learning_v3.tex:12` | "are the workhorses of" -> "are the two fundamental algorithms of" | Adds an exhaustiveness claim the old text did not make. The LP formulation is a third fundamental method, and Puterman (1994), cited in this file's own header, treats it as one. |
| `ch08_offline_rl/tex/offline_rl.tex:133` | "The methodological move is the same," -> "The method is the same:" | Contradicts the two preceding footnote sentences, which enumerate how the methods differ. "Method" denotes a concrete named algorithm everywhere else in the chapter. Also introduces a banned prose colon. |
| `ch06_macro/tex/macro_rl.tex:1266` | "target" -> "address" | "Target" is a term of art for the algorithm-aims-at-equilibrium-concept relation, retained at lines 135, 1257, 1274. "Address" weakens aims-at to deals-with and breaks the parallel. |

Each rejected site keeps its original wording, which is the text that shipped in 269pp of
reviewed PDF. Where a row was rejected the strain it flagged is still present; a better
rewrite is welcome but must clear the same verifier.

## Applied with a correction

`ch06_macro/tex/macro_rl.tex` 1266-1267. The lenses -> roles rename landed (lines 27, 1240,
1266) and now agrees with the subsubsection heading at 1222, "Four roles, one toolbox". Line
1267 was not in the inventory but said "papers within a lens", whose antecedent the rename
removed; it now reads "within a role". The chapter's only remaining "lens" hits are a LaTeX
comment at line 2 and line 222, which is a different sense (the behavioural lens).

## Known, not addressed

`ch12_world_models` defines "world model" twice: `s01_intro.tex:4` ("learned forecasters of
state transitions and rewards") and `s04_deep_mbrl.tex:6` ("A world model is an estimated
model of next states and rewards", inside a nine-term glossary). Pre-existing, introduced in
`9238bab`, untouched by any de-strain commit. A Rule 1 redundancy (one canonical location per
idea) rather than a strain defect.
