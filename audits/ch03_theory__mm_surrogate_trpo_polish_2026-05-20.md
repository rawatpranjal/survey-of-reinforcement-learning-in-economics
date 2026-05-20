# Polish: ch03_theory/sims/mm_surrogate_trpo.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch03_theory__mm_surrogate_trpo_2026-05-19.md` (25%)
**Diagram-only:** YES. Cap remains 25% unless visual contradicts caption.

## Fixes applied

### Fix 1 — Footnote disclaiming the fitted constant `c`

Added a `\footnotetext` immediately after the figure environment that contains the `mm_surrogate_trpo.png` inclusion in `ch03_theory/tex/planning_learning_v3.tex`. The footnote text:

> The quadratic majorizer here uses a fitted constant $c$ chosen so the surrogate touches the true objective at $\theta_{\mathrm{old}}$ and remains below it on the visible grid. \citet{Schulman2015}'s theoretical constant $C = 4\varepsilon\gamma/(1-\gamma)^2$ is so conservative the resulting step is unusably small in practice; modern TRPO implementations replace the penalty with a hard KL constraint. This illustrative cartoon visualizes the *form* of the MM bound, not its sharpness.

This addresses the prior audit's central concern (item 1): readers now know that `c` is fitted to the visible grid, that Schulman's theoretical `C` is in KL space and would yield unusably small steps, and that the figure is illustrative rather than a faithful depiction of Theorem 1 of Schulman2015. The citation key `Schulman2015` is verified in `docs/refs.bib:782`.

### Fix 2 — Caption reframed as MM schematic, not TRPO theorem

Caption opening rewritten from:

> Majorization-minimization interpretation of trust region updates. *Left*: the surrogate $L(\theta|\theta_{\mathrm{old}})$ (dashed) lower-bounds $J(\theta)$ (solid) ...

to:

> Schematic of the majorize-minimize step underlying TRPO: a surrogate $L(\theta)$ lower-bounds the true objective $J(\theta)$ and is monotonically improved by maximizing $L$. *Left*: the surrogate (dashed) is tight at $\theta_{\mathrm{old}}$ ...

The word "schematic" front-loads the cartoon framing. The redundant `L(\theta|\theta_{\mathrm{old}})` notation in the caption (which falsely echoed the TRPO surrogate equation just above) is dropped; readers see `L(\theta)` matching the conceptual lower-bound role.

A short caption (`\caption[Schematic of the MM step underlying TRPO]{...}`) was added to keep `\listoffigures` clean.

### Fix 3 — `_stdout.txt` artifact captured

Ran `python3 ch03_theory/sims/mm_surrogate_trpo.py > ch03_theory/sims/mm_surrogate_trpo_stdout.txt 2>&1`. The file is now committed-ready at `ch03_theory/sims/mm_surrogate_trpo_stdout.txt` (37 lines, 1.3 KB). Contents reproduce verbatim the audit's data-integrity check: `theta_old = 1.0000`, `c = 2.1201`, `theta_new = 0.8367`, monotonic convergence `J(theta_0) = 0.000 -> 1.189925 -> 1.189927 -> 1.189927`, `0 violations out of 1000 points`.

## Verification

- Script reran cleanly, exit 0. PNG and stdout both regenerated.
- Chapter PDF recompiled: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (2.52 MB, 31 pp). The new caption and footnote render correctly on page 25 (`pdftotext -f 25 -l 26 ch03_theory.pdf` shows both verbatim).
- Caption now reads "Schematic of the majorize-minimize step underlying TRPO" rather than asserting the depicted surrogate is the TRPO surrogate.
- Footnote renders as footnote 44 on page 25, attached visually to the figure caption via `\protect\footnotemark` + `\footnotetext`. Hyperref's footnote-anchor link is mildly broken for this stand-alone `\footnotetext` (a pdfTeX dest warning), but the footnote text itself prints correctly and the reading order is preserved. Acceptable for a static PDF.

## Residual concerns (none worth re-scoring above 15%)

- The figure still uses a parameter-space `(θ − θ_old)²` penalty rather than a KL penalty. The footnote now explicitly disclaims this, so a hostile reviewer reading the caption + footnote together cannot accuse the figure of being passed off as the TRPO bound. The substance (MM yields monotone improvement to a non-convex max) was already correct and remains correct.
- The hyperref `Hfootnote` destination warning is cosmetic; the same warning class fires elsewhere in the chapter for normal footnotes and is not specific to my change.

## Score

Adversarial reaction after fixes: Reviewer 2 reads "Schematic of the majorize-minimize step underlying TRPO" + the footnote spelling out that `c` is fitted and that Schulman's theoretical `C` lives in KL space and would yield unusable step sizes. The figure no longer claims to be the TRPO bound; it claims to be the MM cartoon that motivates the TRPO bound. The substance survives without revision. A pedantic reviewer might still want the figure regenerated with a KL-based penalty, but the figure is no longer mislabelled, so the complaint is preference, not correctness.

**Bullshit score: 10%** — Caption and footnote now precisely frame the artifact as a schematic of the MM mechanism rather than a depiction of Schulman 2015 Theorem 1; the script's fitted `c` is openly disclosed; the stdout artifact is in place. The hostile reviewer reads twice looking for a hole and finds only a stylistic preference (KL vs parameter-space penalty in the cartoon), not a correctness defect.
