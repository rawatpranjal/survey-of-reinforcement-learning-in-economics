# csreview_short_30pp — Computer Science Review

## Target

- **Journal:** Computer Science Review
- **Publisher:** Elsevier (Ireland)
- **Editor-in-Chief (2025– ):** Jan Kratochvíl, Charles University, Department of Applied Mathematics, Prague (single EiC)
- **Length:** ~30 pp / ~20 000 words (preferred); minimum 20 typeset pages for the author honorarium
- **Author payment:** EUR 400 per accepted article meeting the 20-pp minimum
- **OA / fees:** Submission free; OA option USD 4,420 APC

## Critical caveat

The journal is **invitation-leaning**. The Guide for Authors says:
> *"Authors should provide a PDF or PS copy of their manuscript to the Editor who invited the author to write the survey."*
> *"At least one author is expected to have at least three papers on the subject of the survey published in high impact factor journals or highly ranked conferences and listed in the bibliographic references of your submission."*

**Do not cold-submit.** Email Kratochvíl first with a fit pitch before investing time. Template is in `submissions/cs_review/presubmit_email.md`.

The 138-page master draft is roughly 4× the journal's preferred length, so substantial cutting is required regardless.

## Cuts plan

TODO. Roughly half the `csur_full_50pp` version. Suggested:
1. Keep: Intro, single Background section, Bandits, Games, Offline+RLHF, Causal, Discussion.
2. Drop: History, separate RL Algorithms / Theory / Empirics sections (fold into Background), Control, Structural Estimation, Robust/Constrained.
3. Strip simulation tables aggressively.

## Cover letter

`submissions/cs_review/cover_letter.tex` — only after the pre-submission email gets a positive response. Lead with the author qualifications (advisor John Rust, prior survey work) since the Guide for Authors explicitly weighs this.
