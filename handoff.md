# Handoff — 2026-05-29 — humanize-pass

## Where we left off
Surgical de-AI ("humanizer") pass across all 17 chapters: done, accepted, committed (e79f828), pushed. ~112 prose edits + 5 caption de-bolds, each chapter independently check-PASSed. Edits swapped live into the chapter .tex; main.pdf rebuilt 230pp (was 232, removed closers/epigrams reflowed). Branch now 78 commits ahead of origin/main, not merged.

## Active streams (clustered)
```
[de-AI pass]  17 chapters DONE 2026-05-29, committed+pushed            — CLOSED
[submit D]    D1 merge humanize-pass -> main, D2 full compile,
              D3 arxiv tarball, D4 submit                              — READY
[re-audit]    B1.6 fishery: Opus re-audit on fishery_paradigms.py <25% — PENDING
[journals E]  deferred until D ships                                   — POST-ARXIV
```

## Decisions made this session
- De-AI = SURGICAL ONLY (egregious tells: colon-drumrolls, negative parallelism, aphoristic closers, em dashes incl Unicode). Wholesale + section-by-section rewrites both tried on ch07, REJECTED by user (prose too technical). Cap ~2 edits/section, 0 if clean.
- Em-dash parenthetical pairs -> parentheses, not commas (commas garden-path). Table N/A "---" cells + caption N/A symbols preserved.
- ch09 only: stripped 5 bold caption lead-ins (\textbf banned by CLAUDE.md; other 16 chapters had none).
- Two-pass per chapter: Opus edit + independent Opus check (executor != verifier). Originals saved to each chapter's tex/backups/2026-05-29-* (committed).
- Full before/after record: docs/humanizer_edits_report.md (committed).
- Pushed branch only; did NOT merge to main (user's call; gates Phase D).

## Open questions
- Merge humanize-pass -> main (Phase D1): now carries the de-AI edits too; still the gating step for arxiv.
- B1.6 fishery re-audit still pending from the prior (2026-05-22) session.
- search/ left untracked (pre-existing, not gitignored).

## Landmines / gotchas
- compile_chapter.tex lacks `placeins` -> \FloatBarrier undefined when compiling any chapter STANDALONE (harmless, PDF still builds; main.tex loads placeins so the full build is clean).
- main.pdf full build ~3 min (3 passes + bibtex); now 230pp.
- ch12_world_models/sims/cache/ fishery caches still fresh — do NOT delete unless re-running.
- This session's backups under each chapter's tex/backups/2026-05-29-*_original.tex (committed; full originals recoverable there or via git pre-e79f828).

## Suggested next move
Phase D1: merge humanize-pass -> main (now includes the de-AI pass), then D2 full compile + page-count check, D3 arxiv tarball. Optional sidecar: Opus re-audit on fishery_paradigms.py to confirm B1.6 <25%.
