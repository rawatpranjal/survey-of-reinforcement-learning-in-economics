# offline_rl_rlhf — Offline RL + RLHF

## Source chapters

- `ch08_offline_rl/tex/offline_rl.tex` (FQI, CQL, IQL, BCQ; perishable inventory pricing simulation)
- `ch09_rlhf/tex/rlhf.tex` (RLHF, DPO; job-search preference-learning simulation)

Per `claude.md`, ch09 has been merged into ch08 conceptually, but the simulation code remains in `ch09_rlhf/sims/`. Both `tex` files are still canonical sources.

## Targets

### Primary: Journal of Economic Perspectives
- See `../bandits_economics/NOTES.md` for JEP details (Williams/Kling, Taylor; outline-first; ~25–35 pp; OA, no APC).

### Technical alt: Foundations and Trends in Microeconomics
- See `../bandits_economics/NOTES.md` for FnT-Micro details (Viscusi; ~80–120 pp; abstract-first).

## Framing

Per `journal_target.md` Theme 3: "JEP is the right venue for showing economists how rationality axioms shape RLHF/LLM alignment for general readers."

Pitch:
- RLHF is a discrete-choice problem dressed in ML notation. Bradley-Terry preference modelling is exactly McFadden's logit; DPO replaces the explicit reward parameterisation with a likelihood that economists already know.
- Offline RL is "off-policy evaluation" with different vocabulary. The pessimism principle (CQL, BCQ) maps to bounding-based identification under unmeasured confounding.
- Open problems: identification under strategic raters; Slutsky-like restrictions on reward models; transitivity violations in human preference data.

## JEP outline-first sequencing

Write `submissions/jep/proposal.md` first (~2–5 pp). Only fill `main.tex` after proposal is greenlit.

## Cuts / additions plan

TODO:
1. Drop FQI/CQL/IQL/BCQ algorithm exposition for the JEP version — keep the conceptual contrast (pessimism) and reference the master draft for formalism.
2. For FnT-Micro: expand the Bradley-Terry / DPO derivations; add an axiomatic treatment of preference data; include a worked example connecting RLHF to logit demand estimation.

## Cover letters / proposals

- `submissions/jep/cover_letter.tex`
- `submissions/jep/proposal.md` — ~2–5 pp outline (JEP requires this before full submission)
- `submissions/fnt_micro/cover_letter.tex`
