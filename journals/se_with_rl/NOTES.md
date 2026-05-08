# se_with_rl — Structural Estimation with RL

## Source chapter

`ch05_econ_models/tex/rl_in_se.tex` (and surrounding chapter material).

## Targets

### Primary: Foundations and Trends in Econometrics
- **Editor-in-Chief:** William H. Greene, NYU Stern (founding/continuing)
- **Length:** ~80–120 pp monograph
- **Submission policy:** Abstract-first, same as FnT-ML
- **OA / fees:** Subscription, no APC

### Backup: Journal of Economic Surveys
- **Editors (2024– ):** Brian Lucey (Trinity College Dublin), Sushanta Mallick (Queen Mary), Tom Stanley (TU Chemnitz)
- **Length:** flexible; technical depth tolerated
- **Submission portal:** Wiley Research Exchange (open, unsolicited)
- **Caveat:** Lucey was subject of a Retraction Watch investigation (Jan 2026); Wiley confirmed he remains EiC. Mallick and Stanley uncontroversial.

## Critical: avoid overlap with ORE_main

The sister survey `../ORE_main/` (Rust & Rawat, Jan 2026, "Structural Econometrics and Inverse Reinforcement Learning: Inferring preferences and beliefs from human behavior") covers DDC estimation + IRL formally. **This carve-out must NOT replicate that material.** Instead:

- Reference ORE_main for foundational DDC/IRL formalism rather than re-deriving.
- Emphasize the *RL-side* framing: function approximation, deep RL for DDC, sample-efficient inference, off-policy methods.
- Position the contribution as how modern RL changes the SE workflow, not as a fresh exposition of SE itself.

## Cuts / additions plan

TODO:
1. Cut any IRL formalism that appears in ORE_main; one-paragraph summary + reference suffices.
2. Add a section on deep RL for high-dimensional DDC that is *not* in ORE_main.
3. If targeting FnT-Econometrics, expand to ~80 pp; if JES, compress to ~40–60 pp.

## Cover letters

- `submissions/fnt_econometrics/cover_letter.tex` — address Greene; lead with the RL-side bridge framing.
- `submissions/jes/cover_letter.tex` — address the editorial board; emphasize technical depth and absence of duplication with ORE_main.
