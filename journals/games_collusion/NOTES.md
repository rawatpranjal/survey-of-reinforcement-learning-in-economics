# games_collusion — Multi-agent RL & algorithmic collusion

## Source chapter

`ch06_games/tex/rl_in_games.tex` (Coase conjecture / durable goods monopoly, Kuhn poker / CFR-FP equilibrium).

## Targets

### Primary: Journal of Economic Perspectives
- See `../bandits_economics/NOTES.md`. Outline-first.

### Backup: Journal of Economic Surveys
- See `../se_with_rl/NOTES.md`.

### CS-first alt: ACM Computing Surveys
- See `../csur_full_50pp/NOTES.md`. Reframe with CS-first taxonomy if the JEP/JES route fails.

## Framing

Per `journal_target.md` Theme 4: "Algorithmic collusion is currently an active topic at JEP-style venues."

Anchor citation: Calvano, Calzolari, Denicolò, & Pastorello, "Artificial Intelligence, Algorithmic Pricing, and Collusion," *American Economic Review* 110, no. 10 (Oct 2020): 3267–97 (DOI: 10.1257/aer.20190623). Their finding — *"the algorithms consistently learn to charge supracompetitive prices, without communicating with one another"* — is the policy-relevant anchor.

Pitch:
- For **JEP**: practitioner / policymaker framing of independent-learning collusion, what RL methods produce it, how to detect it.
- For **JES**: more technical — equilibrium-selection problem, function approximation under multi-agent dynamics, MARL convergence guarantees.
- For **CSUR**: CS-first taxonomy of MARL methods, multi-agent RL surveys are a recurring CSUR topic.

## Cuts / additions plan

TODO:
1. For JEP: foreground the antitrust framing; cut Kuhn poker (CFR-FP equilibrium) detail; keep durable-goods monopoly only.
2. For CSUR: keep both simulations; add a taxonomy of independent-learners vs centralized-critic vs opponent-modelling.
3. Always add a section on antitrust enforcement implications when targeting econ venues.

## Cover letters / proposals

- `submissions/jep/cover_letter.tex` (+ `proposal.md`)
- `submissions/jes/cover_letter.tex`
- `submissions/csur/cover_letter.tex`
