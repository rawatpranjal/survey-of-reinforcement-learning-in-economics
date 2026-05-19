# Journal Submissions

Workspace for journal-targeted versions of the RL-for-Economics survey. Each top-level subfolder is one *manuscript version*; per-journal cover letters and correspondence live inside `submissions/<journal>/`. Source files only — compile with `./build.sh <version>` against the symlinked shared assets in `shared/`.

The strategy and venue analysis are in `../journal_target.md`; this index just maps versions to journals.

## Versions × journals

| Version | Source chapters | Length | Primary venue | Backups | Status |
|---|---|---|---|---|---|
| `csur_full_50pp` | all (condensed) | ~50 pp | ACM Computing Surveys | — | not started |
| `fntml_monograph_100pp` | all | ~100 pp | Foundations and Trends in ML | — | not started |
| `se_with_rl` | ch05_econ_models | ~40–60 pp | FnT in Econometrics | JES | not started |
| `causal_rl` | ch10_causal | ~40–60 pp | Statistical Science | JES | not started |
| `bandits_economics` | ch07_bandits | ~25–35 pp / longer for FnT | JEP | FnT-Micro, JES | not started |
| `offline_rl_rlhf` | ch08_offline_rl + ch09_rlhf | ~25–35 pp | JEP | FnT-Micro | not started |
| `games_collusion` | ch06_games | ~25–35 pp | JEP | JES, ACM CSUR | not started |
| `csreview_short_30pp` | all (further cut) | ~30 pp | Computer Science Review | — | not started |
| `aimag_practitioner_15pp` | all (heavy cut) | ~10–15 pp | AI Magazine | — | not started |

## Sequencing (from `journal_target.md`)

**Stage 1 — primaries (parallel within ~30 days).**
1. ACM CSUR (`csur_full_50pp/`). Submit the full survey condensed to ~50 pp; original taxonomy as the explicit contribution. Address co-EiCs My T. Thai (Florida) and Hanghang Tong (UIUC). The 138-page master becomes a supplementary technical report on arXiv.
2. FnT-ML (`fntml_monograph_100pp/`). Two-step: send `abstract.tex` first to the publisher (Now/Emerald, Tibshirani EiC); on preliminary acceptance, submit the ~100-page monograph framed as an ML ↔ statistics ↔ economics bridge.

**Stage 2 — chapter-aligned carve-outs (60–90 days).**
3. `causal_rl/` → Statistical Science (Lutz Dümbgen, EiC 2026–28). Backup JES.
4. `se_with_rl/` → FnT in Econometrics (William H. Greene, EiC). Avoid duplication with sister survey ORE_main.
5. `bandits_economics/` → JEP (Williams/Kling) for the broad-audience dynamic-pricing piece. Technical-monograph alt FnT-Micro (Viscusi).
6. `offline_rl_rlhf/` → JEP (RLHF + rationality angle). Technical alt FnT-Micro.
7. `games_collusion/` → JEP. Algorithmic-collusion topic anchors on Calvano et al. 2020 *AER*. Backups JES, CSUR.

**Stage 3 — fall-backs (only if Stage 1 rejects).**
8. `csreview_short_30pp/` → Computer Science Review (~30 pp). Invitation-leaning; email EiC Jan Kratochvíl before submitting (`submissions/cs_review/presubmit_email.md`).
9. `aimag_practitioner_15pp/` → AI Magazine (AAAI/Wiley). Practitioner-facing, OA, no APC.

## Layout convention

```
journals/
  README.md                  this file
  build.sh                   ./build.sh <version_dir> [entry_file]
  shared/                    symlinks: refs.bib, econometrica.bst, figs/, glossary.tex, compile_chapter.tex
  <version>/
    main.tex                 (or abstract.tex / outline.tex where structure differs)
    NOTES.md                 framing, length norms, EiC names, what to cut from master
    submissions/
      <journal>/
        cover_letter.tex
        notes.md             dates, status, response correspondence
        presubmit_email.md   only where the journal is invitation-leaning (CSR)
```

Skeletons are source-only: chapter content is `\input{../../chXX_topic/tex/...}` directly from the master draft. Fork to a local copy only when journal-tailored prose actually diverges.

## Master draft (for reference, not modified)

- Master: `../docs/main.tex` (~138 pp, full arXiv survey)
- arXiv submission bundle: `../arxiv_submission/` (self-contained snapshot, no symlinks)
- Thesis variants: `../thesis/` (121 pp, all chapters), `../thesis_v2/` (86 pp, economist-facing dissertation cut)
