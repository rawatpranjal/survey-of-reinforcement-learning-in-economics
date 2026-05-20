# Polish: ch03_theory/sims/deadly_triad_geometry.py

**Date:** 2026-05-20
**Prior audit:** `ch03_theory__deadly_triad_geometry_2026-05-19.md` (score 25%, diagram cap)
**Diagram-only:** yes (cap 25%)
**Scope:** notation alignment with surrounding prose, caption disclosures.
Substantive re-implementation explicitly out of scope.

---

## Changes applied

### Fix 1 — Notation alignment (Option A, figure → prose)

The prior audit flagged that `planning_learning_v3.tex` line 276 fixes $\mu$ as
the *behavior* (off-policy) distribution and $d^\pi$ as the target-policy
stationary distribution, while the figure used $\mu$ for the on-policy panel
and $\nu$ for the off-policy panel — the opposite convention. Chose Option A
(update the figure) so the chapter prose convention persists unchanged across
later sections.

Edits in `/Users/pranjal/Code/rl/ch03_theory/sims/deadly_triad_geometry.py`:

- Panel (a) projection arrow label: `\Pi_\mu\, TV` → `\Pi_{d^\pi}\, TV`
- Panel (a) length annotation: `\|\Pi_\mu\, TV\| < \|TV\|` →
  `\|\Pi_{d^\pi}\, TV\| < \|TV\|`
- Panel (b) projection arrow label: `\Pi_\nu\, TV` → `\Pi_\mu\, TV`
- Panel (b) length annotation: `\|\Pi_\nu\, TV\| > \|TV\|` →
  `\|\Pi_\mu\, TV\| > \|TV\|`, and "expansion causes divergence" weakened to
  "expansion can defeat contraction" (the weaker, accurate claim; expansion of
  a single projection is necessary, not sufficient, for the iterate to
  diverge).

Re-ran the script to regenerate the PNG. Numerical output unchanged (still
`||proj_orth||=2.0000 < ||TV||=2.0396 < ||proj_obl||=3.0990`, expansion ratio
1.5194); only label TeX strings changed. The new PNG was verified visually:
panel (a) shows the green orthogonal projection labeled $\Pi_{d^\pi}\, TV$;
panel (b) shows the red oblique projection labeled $\Pi_\mu\, TV$.

### Fix 2 — Disclosure of hand-tuned geometry

Added a notation-and-geometry note to the script's module docstring stating
that `theta_sub = 20°`, `delta_deg = 20°` are illustrative (chosen for figure
legibility) and that the 1.52 expansion ratio is a property of this
configuration, not derived from any specific MDP / behavior distribution
pair. Mirrored the same caveat into the caption (see Fix 3).

### Fix 3 — Caption framing (deadly triad scope)

Edit in `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex`
(figure environment around line 284–289):

Old caption: "Geometry of the projected Bellman operator in $\mathbb{R}^2$
[...]. (b) Off-policy: the oblique projection $\Pi_\nu$ (under the behavior
distribution) reaches a point further from the origin than $TV$ itself,
causing expansion."

New caption: "Geometric intuition for the projection-norm leg of the deadly
triad in $\mathbb{R}^2$. The gray line is the function approximation subspace
span($\Phi$); the blue arrow is $TV$, the Bellman update. (a) On-policy: the
orthogonal projection $\Pi_{d^\pi}$ (under the target-policy stationary
distribution) drops $TV$ perpendicularly onto the subspace, preserving the
contraction. (b) Off-policy: the oblique projection $\Pi_\mu$ (under the
behavior distribution) reaches a point further from the origin than $TV$
itself, so the projection can expand distances. Bootstrapping and off-policy
sampling, the other two legs of the triad, are not depicted; their joint
effect on divergence is exhibited numerically in Baird's seven-state star
counterexample \citep{Baird1995}. The displayed expansion ratio is
illustrative; the subspace and projection-direction angles are chosen so
contraction in (a) and expansion in (b) are visible at figure scale rather
than derived from a specific MDP."

This addresses all three audit nicks: notation now matches prose, the
"deadly triad" framing is replaced by "projection-norm leg of the deadly
triad" with an explicit pointer to the Baird counterexample for the
bootstrapping and off-policy-sampling legs, and the hand-tuned ratio is
acknowledged in the caption (with the longer-form rationale in the script
docstring).

### Out-of-scope and not touched

- The substantive geometric content (no actual feature matrix, no $\mu$- or
  $d^\pi$-weighted inner product, no spectral-radius condition tying the
  visual 1.52 to a $\gamma$ at which $\Pi T^\pi$ stops contracting). This
  remains a chalkboard sketch by design.
- The orthogonality of `d` (projection direction) to span($\Phi$) under any
  specific $\mu$-norm. The figure picks $d$ visually; for a 1-D subspace
  any non-degenerate $d$ parameterizes *some* oblique projection, so the
  picture is geometrically valid even though the link to a particular
  behavior distribution is asserted, not derived.

## Verification

- Re-ran `python3 ch03_theory/sims/deadly_triad_geometry.py >
  ch03_theory/sims/deadly_triad_geometry_stdout.txt 2>&1`. Exit 0. Stdout
  numbers unchanged (geometry constants unchanged). PNG regenerated with new
  labels; visually inspected the rendered figure and confirmed panel (a)
  labels $\Pi_{d^\pi}\, TV$ (green) and panel (b) labels $\Pi_\mu\, TV$
  (red).
- Recompiled the chapter: `pdflatex / bibtex / pdflatex / pdflatex` from
  `docs/`. Output: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (40 pages,
  2,520,442 bytes). No compilation errors. Figure appears with updated
  caption.

## Residual hostile-reviewer surface

After polish, what remains for a hostile reviewer:

- The figure still depicts only the projection-norm leg of the triad; the
  caption now flags this and points to Baird's counterexample. A reviewer
  who insists the figure should also exhibit divergence dynamics will still
  prefer a phase-plane plot of the iterate $\theta_k$, but the caption no
  longer overclaims.
- The 1.52 expansion ratio is hand-picked. The caption now says so. A
  reviewer can still ask "what behavior distribution would produce this?"
  but the figure does not pretend to answer that.
- Single panel (b) configuration. The figure does not show a family of
  oblique projections (e.g., varying `delta_deg`) to make the dependence on
  the projection direction visual. This was deemed out of scope for a polish
  pass.

These are diagram-quality limits, not algorithm-correctness problems. The
notation collision Reviewer 2 caught in the prior audit is gone.

**Bullshit score: 12%** — The geometric content is correct, notation
matches the surrounding prose, the caption no longer overclaims the
"deadly triad" framing, and the hand-tuned ratio is disclosed. A hostile
reviewer can still wish for a phase-plane plot of $\theta_k$ divergence,
but the artifact's claims now line up with its scope and the chapter
prose. Diagram cap of 25% applies; we are comfortably below it.

## Artifacts

- Edited script: `/Users/pranjal/Code/rl/ch03_theory/sims/deadly_triad_geometry.py`
- Regenerated figure: `/Users/pranjal/Code/rl/ch03_theory/sims/deadly_triad_geometry.png`
- Regenerated stdout: `/Users/pranjal/Code/rl/ch03_theory/sims/deadly_triad_geometry_stdout.txt`
- Edited tex: `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (caption only)
- Recompiled chapter PDF: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (40 pp)
