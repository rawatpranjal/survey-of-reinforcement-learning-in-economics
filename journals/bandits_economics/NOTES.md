# bandits_economics — Economic bandits & dynamic pricing

## Source chapter

`ch07_bandits/tex/dynamic_pricing.tex` and surrounding chapter material (knowledge ladder, structural pricing).

## Targets

### Primary: Journal of Economic Perspectives (broad-audience version)
- **Co-Editors:** Heidi Williams (Stanford), Jeffrey Kling (CBO); managing editor Timothy Taylor
- **Length:** ~25–35 pp; general-readership prose, minimal formal apparatus
- **Submission:** outline-first (~2–5 pp proposal); ~10–15% of articles originate from unsolicited proposals
- **OA / fees:** Free public access, no APC

### Technical alt: Foundations and Trends in Microeconomics
- **Editor-in-Chief:** W. Kip Viscusi, Vanderbilt
- **Length:** ~80–120 pp
- **Submission:** abstract-first (Now Publishers / Emerald)

### Backup: Journal of Economic Surveys
- See `../se_with_rl/NOTES.md`.

## Framing

Per `journal_target.md` Theme 3: "JEP is the right venue for showing economists how rationality axioms shape RLHF/LLM alignment for general readers; FnT-Micro for the formal monograph on demand systems imposed on bandits."

Pitch:
- For **JEP**: economic structure (WARP, McFadden, demand elasticities, incentive compatibility) changes how bandit algorithms behave in real markets — algorithms blind to structure leave money on the table or violate consumer rationality. Lead with concrete dynamic-pricing examples.
- For **FnT-Micro**: the same content with full formal apparatus — Bradley-Terry, structural demand identification, Slutsky restrictions on bandit policies, pricing under heterogeneous WTP.

## JEP outline-first sequencing

JEP wants a 2–5-pp proposal before a full draft. TODO: write `submissions/jep/proposal.md` first; only write `main.tex` after proposal is greenlit.

## Cuts / additions plan

TODO:
1. JEP version: cut all simulation tables to one-line summaries; tighten formal apparatus to a few key equations; add a real-market case study.
2. FnT-Micro version: expand to include (a) full Bradley-Terry derivation, (b) Slutsky restrictions on bandit policies, (c) chapter-length treatment of structural pricing.

## Cover letters / proposals

- `submissions/jep/cover_letter.tex` — broad-audience pitch; cite Calvano et al. 2020 AER as a precedent for econ-relevant algorithmic-pricing surveys at general-readership venues.
- `submissions/fnt_micro/cover_letter.tex` — address Viscusi; lead with the demand-systems-on-bandits framing.
- `submissions/jes/cover_letter.tex` — address editorial board; emphasize the dynamic-pricing application angle.
