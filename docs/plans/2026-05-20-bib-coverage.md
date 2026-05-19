# Bibliography Coverage Audit — 2026-05-20

**Scope:** `docs/main.tex` input chain (29 in-scope .tex files after sub-input expansion).
**Bibliography:** `docs/refs.bib` (433 entries; `refs_extended.bib` deleted).
**Excluded:** `journals/`, `thesis/`, `thesis_v2/`, `ORE_main/`, `archive/`,
`tex/backups/`, `.claude/worktrees/`.

**Methodology note:** Citation regex handles `\citep[pre][post]{key}` (natbib optional args);
commented-out lines (`%`) are excluded from the cited-key set.

**Post-audit correction (2026-05-20):** The 10 "orphan" entries reported below are
FALSE POSITIVES. The audit's regex missed `\citet{key}` forms in footnotes and
prose. Manual grep verification (controller, 2026-05-20) confirms each key has 1-2
active citations in the in-scope tex tree. The trim performed in commit f71f40c
(based on extract_cites.py output) was correct; no entries need to be restored.
Treat this section as a known auditor regex bug, not a real finding.

---

## Cited but missing (in .tex but not in .bib)

(none)

---

## Defined but orphan (in .bib but never cited)

| Key | Type | Defined in |
|-----|------|------------|
| Ajay2023 | @inproceedings | refs.bib:3530 |
| Badanidiyuru2013 | @article | refs.bib:450 |
| Eimer2023 | @inproceedings | refs.bib:2428 |
| farahmand2010 | @inproceedings | refs.bib:1407 |
| idaIshiharaItoEtAl2024energyrebate | @techreport | refs.bib:3719 |
| Janner2022diffuser | @inproceedings | refs.bib:3522 |
| luckett2020vlearning | @article | refs.bib:3709 |
| Mueller2019 | @inproceedings | refs.bib:722 |
| Patterson2024 | @inproceedings | refs.bib:2437 |
| Towers2024 | @inproceedings | refs.bib:2572 |

10 orphan entries. These survived the 810→433 trim. Candidates for deletion unless
they are placeholders for prose still being drafted (ch09 RLHF, ch12 world models).

Notable clusters:
- **ch08/ch09 offline/RLHF:** `Ajay2023`, `Janner2022diffuser` (Decision Transformer variants),
  `luckett2020vlearning` (V-learning for offline RL).
- **ch07 bandits:** `Badanidiyuru2013` (bandits with knapsacks).
- **ch11 robust/constrained:** `Eimer2023`, `Patterson2024`.
- **ch04 control / Gymnasium:** `Towers2024` (Gymnasium).
- **ch03 theory:** `farahmand2010` (concentrability; was cited in commented-out prose at
  planning_learning_v3.tex:309).
- **ch06 macro:** `Mueller2019`.
- **ch10b causal:** `idaIshiharaItoEtAl2024energyrebate`.

---

## Duplicate keys

(none)

---

## Entries missing required fields

(none)

---

## Summary

| Section | Count |
|---------|-------|
| Cited but missing | 0 |
| Defined but orphan | 10 |
| Duplicate keys | 0 |
| Missing required fields | 0 |

The post-cleanup bib is clean on the hard constraints (no missing, no duplicates, no
malformed entries). The 10 orphans are the only actionable items; none block compilation.
