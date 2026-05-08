# csur_full_50pp — ACM Computing Surveys

## Target

- **Journal:** ACM Computing Surveys (CSUR)
- **Publisher:** Association for Computing Machinery (ACM), United States
- **Co-Editors-in-Chief (1 July 2025 – 30 June 2028):** My T. Thai (University of Florida), Hanghang Tong (UIUC)
- **Submission portal:** https://mc.manuscriptcentral.com/csur (unsolicited; no invitation needed)
- **Length norm:** 30–50 pp typical, 100–300+ references. 138 pp would need to be condensed or split.
- **OA / fees:** ACM Open transition (since Jan 2026); subsidized APC ~$950 (ACM/SIG members) / $1,450 (non-members) at non-Open institutions.
- **2024 IF:** 28; ranked #1 in "Computer Science, Theory & Methods".

## Framing requirement

CSUR demands an *original taxonomy or analytical framework*, not a chronological literature catalog. The cover letter must lead with the organizing structure (DP → modern RL → applied carve-outs → practical limitations) as the explicit contribution.

The journal's own description welcomes "Contributions which bridge existing and emerging technologies (such as machine learning) with a variety of science and engineering domains in a novel and interesting way" — this exact framing fits the RL-↔-economics bridge.

## Cuts plan (138 → ~50 pp)

TODO. Suggested order of compression:
1. Drop the History section (`ch01_history`) entirely or fold a 1-paragraph summary into the Introduction.
2. Compress the RL Algorithms / Theory / Empirics blocks (ch02, ch03, ch03b) into a single ~6-pp "Background" section.
3. Keep Control (ch04), Bandits (ch07), Games (ch06), Offline+RLHF (ch08+ch09), Causal (ch10) as the applied carve-outs — these are the contribution.
4. Drop Robust/Constrained (ch11) entirely or compress to a footnote.
5. Drop the Glossary appendix.
6. Strip simulation tables to one line per result, push details to supplementary technical report on arXiv.

## Cover letter

`submissions/csur/cover_letter.tex`. Address Thai and Tong jointly. Pitch the taxonomy framing, point to arXiv supplementary report, name the bridge framing.
