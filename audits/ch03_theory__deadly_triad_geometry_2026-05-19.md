# Audit: ch03_theory/sims/deadly_triad_geometry.py

**Date:** 2026-05-19
**Diagram-only:** yes (no Monte Carlo, no semi-gradient TD iteration; pure analytic R^2 geometry)
**Cited tex file(s):**
- `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (figure inserted at line 286, `\label{fig:deadly_triad_geometry}`, caption lines 287-288, discussion lines 256-300)
- `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v2.tex` (same figure referenced)

**Cited paper PDFs read:**
- `papers/tsitsiklis1997_td_learning_function_approximation.md` (skimmed; confirms the on-policy projection result and references the Baird/TVR counterexamples)
- `papers/baird1995_residual_algorithms.md` (present, not re-read in detail; cited only as the source of the canonical divergence story in the tex)
- The script does not directly implement Baird's 6-state star or the TVR 2-state counterexample; it is a 2-D geometric illustration only.

---

## 1. Algorithm Identity

This is a *cartoon diagram*, not an algorithm. The script does not iterate any TD update, does not build $\Pi T^\pi$, and does not exhibit divergence dynamics. It draws:

- A 1-D subspace `span(Φ)` at 20° in $\mathbb{R}^2$.
- A hand-chosen vector `TV` with along/perp components (2.0, 0.4).
- The orthogonal projection onto that subspace (label `Π_μ TV`, panel a).
- An oblique projection along a direction `d` chosen 20° below the subspace, i.e. along the x-axis at 0° (label `Π_ν TV`, panel b).

Two algorithm-identity problems for a hostile reviewer:

(a) **Label swap between figure and prose.** The tex prose (line 276) uses $\mu$ for the *behavior* distribution (off-policy) and $d^\pi$ as the on-policy / target distribution. In the figure, panel (a) ("on-policy") is labeled $\Pi_\mu TV$ and panel (b) ("off-policy") is labeled $\Pi_\nu TV$. So the figure uses $\mu$ for the *on-policy* projection and $\nu$ for the off-policy projection, which is the *opposite* of the convention established three sentences above the figure. A careful reader notices immediately. The caption itself even says "panel (b) ... oblique projection $\Pi_\nu$ (under the behavior distribution)" — which reinforces that $\nu$ is behavior — but then the prose says behavior is $\mu$. This is a notation inconsistency, not a math error, but it lives inside the same `\subsection`.

(b) **The figure illustrates *projection-norm expansion*, not the deadly-triad divergence per se.** Norm expansion of a single application of an oblique projection is a *necessary* ingredient for divergence of $\Pi T^\pi$ but not sufficient. The caption "expansion causes divergence" telescopes a delicate condition ($\|\Pi_\nu\| > 1/\gamma$ in the appropriate norm with bootstrapping closing the loop) into a single dot. As a *cartoon* this is defensible; as a *proof object* it is not. The script is honest in being a diagram, so I will not double-penalize.

## 2. Environment / MDP Fidelity

No MDP, no states, no actions, no transitions, no behavior policy. The script makes no claim of reproducing Baird's 7-state star or the Tsitsiklis-Van Roy 2-state chain. It is pure geometry.

What it *does* claim implicitly (via the caption) is that the depicted oblique projection corresponds to the behavior-distribution projection $\Pi_\nu$ (under the off-policy norm). It does not derive `d` from any actual $\nu$-weighted inner product on a feature matrix; `d` is *postulated* at 20° below the subspace because that "produces moderate, visually clear expansion/contraction" (comment in code, lines 36-37). The angle is chosen for visual punch, not from any underlying MDP. A hostile reviewer can fairly say: "you drew a picture where the expansion ratio (1.52, line 18 of stdout) is whatever you wanted it to be."

Note also: the oblique projection in the figure is drawn from `TV` *along direction d* onto `span(Φ)`. For this to literally be an oblique projection under a weighted inner product, `d` must equal the direction orthogonal *in the $\nu$-norm* to `span(Φ)`. The script does not verify this consistency; it just picks a direction and slides `TV` along it to the subspace. For a 1-D subspace in $\mathbb{R}^2$ this *is* equivalent to *some* oblique projection (any non-degenerate direction parameterizes one), so the figure is geometrically valid — but the link from "pick d at 0°" to "this is the off-policy projection of some real problem" is asserted, not shown.

## 3. Data Integrity

The numbers in `deadly_triad_geometry_stdout.txt` are recomputable from the code:

- $TV = 2.0 e_{\text{sub}} + 0.4 n_{\text{sub}}$ at $\theta_{\text{sub}} = 20°$ gives `[1.7426, 1.0599]` ✓
- $\|TV\| = \sqrt{2^2 + 0.4^2} = \sqrt{4.16} \approx 2.0396$ ✓
- Orthogonal projection length = 2.0 (drops the perp component) ✓
- Oblique projection: parameter $t = -TV \cdot n_{\text{sub}} / d \cdot n_{\text{sub}} = -0.4 / \cos(20°) \cdot \ldots$, projection lands on $[2.9121, 1.0599]$, length $\sqrt{2.9121^2 + 1.0599^2} \approx 3.099$ ✓

So nothing is hardcoded; the stdout matches the code. The expansion ratio 1.52 is real *for this hand-picked configuration*.

## 4. Comparison Fairness

The "comparison" is between two analytic projections of the *same* fixed `TV`, which is fair. Panel (a) uses the orthogonal projection of $TV$ onto a 1-D subspace, panel (b) uses an oblique projection along a chosen direction. The figure is symmetric in setup. Fairness inside the diagram is fine.

What is *not* fair to the reader: the on-policy case is also drawn with a non-zero perpendicular component (`TV_perp = 0.4`), so the orthogonal projection in panel (a) is *strictly shorter* than `TV` (2.0 < 2.0396). Good. But the off-policy panel reuses *the same `TV`* and gets expansion only because the chosen `d` makes that happen. A reviewer can object that this is a "look how much I can expand if I pick the worst d" picture, not a "this is what happens in a typical off-policy problem" picture. The caption hedges nothing.

## 5. Theoretical Sanity Checks

What theory predicts here, beyond a single picture:

- An orthogonal projection (panel a) has operator norm exactly 1 in the weighted norm and *never* expands. ✓ The code reflects this.
- An oblique projection can have arbitrarily large operator norm depending on the angle between the projection direction `d` and `span(Φ)`. ✓ The code reflects this.
- Tsitsiklis-Van Roy (1997) Theorem 1: on-policy TD($\lambda$) converges *because* $\Pi$ is a non-expansion in the $d^\pi$-norm; the off-policy 2-state counterexample (TVR 1996, cited in this script's tex) shows divergence for any $q \neq \pi$. The figure correctly motivates this, but does *not* derive a TVR-style spectral-radius condition, does *not* show $\|\Pi_\nu\| > 1/\gamma$ as the precise threshold, and does *not* relate the visual expansion ratio (1.52) to a specific $\gamma$ at which $\Pi T^\pi$ stops being a contraction.

For a diagram in a survey chapter this is acceptable. For a survey that elsewhere uses *Baird's actual counterexample* as a sibling sim (`bairds_counterexample.py` is in the same `sims/` directory, with its own stdout and figure), the choice to render the geometric intuition with a synthetic 2-D toy instead of using the Baird projection geometry directly is a soft mark against rigor.

## 6. Information Leakage

Not applicable. No agent, no rewards, no bootstrap target. The script has no notion of "current weight vector" vs "oracle." It draws two arrows. No leakage to flag.

## 7. Seed & Reproducibility

The script sets no `np.random.seed` because it does no sampling. All geometry is deterministic from the hardcoded constants `theta_sub = 20°`, `TV_along = 2.0`, `TV_perp = 0.4`, `delta_deg = 20`. Re-running produces the same `.png` and the same stdout. Reproducibility is fine.

Two minor reproducibility nits a hostile reviewer might raise:

- The visual choices (`theta_sub = 20`, `delta_deg = 20`) are magic numbers with a comment "produces moderate, visually clear expansion/contraction" — i.e. tuned for the figure, not derived from any principle. A reader cannot reverse-engineer "what $\nu$ does this correspond to?"
- No `--data-only` actually computes anything (the script exits with a message). That is consistent with the diagram-only protocol stated in `CLAUDE.md`, so this is honest, but it does mean there is no cached `data.pkl` to compare against if the constants change silently.

---

## Hostile-Reviewer Summary

This is a diagram-only sim and CLAUDE.md caps such sims at 25% unless the diagram visually contradicts its caption. The diagram does *not* contradict its caption; the geometry is computed correctly and the picture honestly shows orthogonal vs oblique projection. The substantive grievances are:

1. **Notation collision between figure and surrounding prose.** Prose uses $\mu$ for behavior (off-policy) — figure uses $\mu$ for on-policy and $\nu$ for behavior. Caption inherits the figure's convention. This is the kind of thing Reviewer 2 will circle on the PDF.
2. The figure is presented as "the deadly triad" but only illustrates one of three ingredients (the projection norm leg). The other two legs (bootstrapping, off-policy sampling) are not in the picture and the caption does not flag the abstraction.
3. The numerical expansion ratio 1.52 is a hand-tuned visual choice, not derived from a stated MDP / behavior distribution pair. Caption does not disclose this.

None of these break the substance. None of them are "method as implemented ≠ method as named" (this is not a method — it is a chalkboard sketch). The 25% diagram cap applies and the notation issue is exactly the "specific small thing" Reviewer 2 catches.

**Bullshit score: 25%** — Reviewer 2 catches the $\mu/\nu$ swap between figure labels and the prose's behavior-distribution convention three sentences above; the geometric content is correct and the substance survives a sed-replace.
