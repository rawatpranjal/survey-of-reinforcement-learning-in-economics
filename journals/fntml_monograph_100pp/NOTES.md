# fntml_monograph_100pp — Foundations and Trends in Machine Learning

## Target

- **Journal:** Foundations and Trends in Machine Learning (FnT-ML)
- **Publisher:** Now Publishers, distributed by Emerald
- **Editor-in-Chief (2022– ):** Ryan Tibshirani, UC Berkeley
- **Submission policy:** Abstract-first. Original research papers rejected. Quote: *"In the first instance, send an abstract for initial review to the publisher. After this initial submission, a preliminary acceptance may follow. The full draft paper will be subject to a reviewing process to ensure quality standards and balance before being finally accepted."*
- **Length norm:** ±100 pp — purpose-built for monograph-length surveys. 138-page draft fits this format with minor compression.
- **OA / fees:** Subscription model; no APC. Final monograph distributed as Now Publishers eBook.

## Framing requirement

Tibshirani's editorial scope (verbatim) explicitly welcomes monographs that *"bridge such problems and perspectives with those from related fields, including (but not limited to) statistics, economics, and optimization."* This is the exact framing for the proposal — emphasize the ML-↔-statistics-↔-economics bridge in both the abstract and the cover letter.

## Two-step process

1. Send `abstract.tex` (built standalone via `../build.sh fntml_monograph_100pp abstract.tex`) to the publisher first.
2. On preliminary acceptance, submit the full `main.tex` (~100 pp).

## Cuts plan (138 → ~100 pp)

TODO. Minimal compression — FnT-ML is the only target where the master draft's length is close to ideal. Suggested:
1. Trim simulation writeups to one paragraph + one table per chapter.
2. Compress the History section to ~3 pp.
3. Trim Robust/Constrained (ch11) by ~50%.

## Direct precedent

Moerland, Broekens, Plaat, & Jonker, "Model-based Reinforcement Learning: A Survey," *Foundations and Trends in Machine Learning* 16, no. 1 (2023): 1–118 (DOI: 10.1561/2200000086) — same length, same scope structure. Cite in cover letter.

## Cover letter

`submissions/fnt_ml/cover_letter.tex`. Address Tibshirani. Lead with the bridge framing. Note that the abstract has already been pre-reviewed (after Step 1).
