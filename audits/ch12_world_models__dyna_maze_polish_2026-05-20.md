# Polish Report: ch12_world_models/sims/dyna_maze.py

**Date:** 2026-05-20
**Predecessor audit:** `audits/ch12_world_models__dyna_maze_2026-05-19.md` (Bullshit score 15%)
**Tex file edited:** `/Users/pranjal/Code/rl/ch12_world_models/tex/s03_dyna_q.tex`
**Sim script changes:** none (framing-only polish, no re-run required)
**Compiled PDF:** `/Users/pranjal/Code/rl/docs/ch12_world_models.pdf` (41 pages, regenerated 2026-05-20)

## Nicks addressed

### Nick 1: "Faithful Schmidhuber 1990" overstatement

**Predecessor diagnosis (audit §1, hostile-reviewer summary):** The script implements REINFORCE on imagined rollouts through a learned forward model. The original Schmidhuber 1990 FKI-148 architecture propagates analytic gradients through a differentiable model directly into the controller; the variant used here uses a stochastic-policy estimator and so is not the exact 1990 formulation, even though it sits in the same family.

**Tex changes in `s03_dyna_q.tex`:**

- Line 57 (model description). Replaced "a faithful realization of the Schmidhuber 1990 controller-model architecture" with "REINFORCE on imagined rollouts under a learned forward model, related to the Schmidhuber 1990 controller-model architecture of \S\ref{section:fc_origins_schmidhuber} but distinct in the gradient pathway, since the original 1990 formulation propagates analytic gradients through a differentiable model into the controller while the variant used here treats the rollout as a stochastic-policy estimator." This is the precise framing the audit asked for: same family, distinct gradient pathway, no overclaim.

- Line 60 (verdict paragraph). Replaced "The differentiable Schmidhuber architecture" with "The neural controller-model agent" and replaced "a differentiable model with gradient-based planning" with "a learned neural model with rollout-based policy gradients" in the three-commitments coda. The compiled language no longer asserts gradients flow through the model, since they do not.

A substantive backprop-through-model implementation is explicitly out of scope per the polish brief.

### Nick 2: Phase-2 recovery rates do not separate Dyna-Q+ from Dyna-Q

**Predecessor diagnosis (audit §5):** Sutton-Barto Figure 8.5 reports Dyna-Q+ recovering faster than Dyna-Q after the blocking-maze wall flip. The thirty-seed reproduction here yields Phase-2 gains of $9.8 \pm 4.0$ (Dyna-Q) and $9.6 \pm 2.8$ (Dyna-Q+), which a Welch $t$-test cannot separate at $p > 0.5$. The original tex acknowledged the gap with "the recovery rates are statistically indistinguishable here" but did not cite Sutton-Barto Fig.~8.5 or attribute the non-replication to hyperparameter choices.

**Tex change in `s03_dyna_q.tex` line 60 (results paragraph):**

Replaced the single hedge sentence with:

> "Dyna-Q+ matches that rate at $9.6 \pm 2.8$, with the curiosity bonus driving the agent to revisit untried actions on the opposite side of the wall. The two rates are statistically indistinguishable on a Welch $t$-test with $p > 0.5$, and the Phase 1 deficit shrinks but does not close by $t = 3000$. \citet{sutton2018} Figure 8.5 reports Dyna-Q+ recovering faster than Dyna-Q after the maze flip; the thirty-seed reproduction here does not separate the two methods on the recovery slope, and a longer post-flip budget or a larger bonus coefficient than $\kappa = 10^{-4}$ would likely be needed to recover the Fig.~8.5 effect size cleanly."

This converts an implicit hedge into an explicit non-replication that names the reference figure, names the test, and names the two parameters (post-flip budget length and $\kappa$) that the literature most often cites as drivers of the Dyna-Q+ recovery advantage on this maze.

## Items the reviewer would still raise

A hostile reviewer can still push for:

1. A $\kappa$ sweep to map the Phase-2 gap as a function of bonus strength. The polish does not add one. The tex now names this as the likely explanation, which is the honest move when re-running is out of scope.
2. A shortcut-maze companion experiment (Sutton-Barto Ex 8.3) where the Dyna-Q+ effect is larger. Out of scope per brief.
3. A literal Schmidhuber 1990 backprop-through-model implementation. Out of scope per brief. The tex now describes what the agent actually does, so this is no longer a misattribution complaint, only a "you could have added another method" complaint.

These are scope rather than correctness items.

## Verification

- Edits applied to `/Users/pranjal/Code/rl/ch12_world_models/tex/s03_dyna_q.tex` at the two places named above.
- `pdflatex -> bibtex -> pdflatex -> pdflatex` ran cleanly; final PDF is `/Users/pranjal/Code/rl/docs/ch12_world_models.pdf` (41 pages, 2,550,656 bytes).
- One bibtex warning, "can't use both volume and number fields in Talvitie2017," is a pre-existing `refs.bib` style nit unrelated to this polish; does not affect compilation.
- No `.py`, `.png`, `_results.tex`, or `_stdout.txt` files were modified, so numerical results in the table and figure remain consistent with the audited 30-seed run.

## Bullshit score: 10%

Reviewer 2's available complaints have narrowed materially. The "faithful Schmidhuber 1990" overclaim is gone, replaced by language that names the gradient-pathway difference explicitly. The Phase-2 non-replication is now an in-text acknowledgment that names Sutton-Barto Fig.~8.5, names the Welch $t$-test, and names the two hyperparameters the literature attributes the effect to. The remaining 10% accounts for the residual scope complaints, no $\kappa$ sweep, no shortcut-maze companion, no analytic-gradient variant of the controller-model agent. These are revision-letter items rather than substance complaints. The algorithms and environment remain correct (audit §§1-2), the data pipeline is honest (§3), the comparison is fair on sample budget (§4), and seeds, SEs, and reproducibility protocol are clean (§7).
