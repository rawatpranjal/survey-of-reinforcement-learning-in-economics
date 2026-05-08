# causal_rl — Causal RL: off-policy evaluation, confounded MDPs

## Source chapter

`ch10_causal/tex/causal_rl.tex` and the chapter's worked simulation (Confounded OPE).

## Targets

### Primary: Statistical Science
- **Publisher:** IMS
- **Editor-in-Chief (2026–28):** Lutz Dümbgen, University of Bern (preceded by Moulinath Banerjee 2023–25, Sonia Petrone 2020–22)
- **Mandate:** review-style technical papers; reviews placed in larger statistical context.
- **Length:** ~40–60 pp typical
- **OA / fees:** Hybrid; IMS open access option

### Backup: Journal of Economic Surveys
- See `../se_with_rl/NOTES.md` for editor and caveat details.

## Framing

Per `journal_target.md`: "Stat Sci/Stat Surveys are the natural homes for off-policy evaluation and identification-style methodology surveys; JES if you keep an econometrics framing with backdoor adjustment / IV instruments."

Pitch as a methodology survey at the intersection of:
- RL (off-policy evaluation, doubly-robust estimators, confounded MDPs)
- Causal inference (instrumental variables, proxy methods, sensitivity analysis under unmeasured confounding)
- Econometrics (dynamic treatment regimes, panel-data policy learning)

## Cuts / additions plan

TODO:
1. Drop any general RL exposition (move to a 4–6 pp Background section that points back to the master draft).
2. Add a worked example contrasting OPE under no confounding vs confounded behavior policies.
3. Tighten to ~40–60 pp for Stat Sci; can extend for JES.

## Cover letters

- `submissions/stat_sci/cover_letter.tex` — address Dümbgen; lead with the identification framing.
- `submissions/jes/cover_letter.tex` — address the editorial board; emphasize the IV/backdoor angle.
