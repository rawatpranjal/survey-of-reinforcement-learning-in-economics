# Polish pass: ch03b_deeprl_practice/sims/overestimation_bias.py

**Date:** 2026-05-20
**Prior audit:** `ch03b_deeprl_practice__overestimation_bias_2026-05-19.md` (15%)
**Target:** ≤10%
**Diagram-only:** yes — 25% cap applies, no visual-vs-caption contradiction, no code-level issue.
**Option picked:** **B** — scope the caption and surrounding prose to the Jensen-inequality illustration; cite \citet{vanHasselt2010,vanHasselt2016ddqn} for empirical DDQN curves rather than reproducing them.

## Changes applied

### 1. Caption precision (audit nit, §1 last line)

`ch03b_deeprl_practice/tex/deeprl_practice.tex` line 43.

- Before: "the bias exceeds $2.5\sigma$" (the actual value is $2.5076\sigma$, so "exceeds" is technically true but only just; a hostile reviewer would flag the rounding).
- After: "the bias reaches $2.51\sigma$" (matches stdout 2.507591 rounded to 2 d.p., honest).

### 2. Caption scope (audit §4 / hostile-reviewer summary)

The section is titled "Value Overestimation and Spikes" and names DQN, Double DQN, Clipped Double Q (TD3), and soft divergence across three paragraphs. The figure is a single Gaussian-density plot. Without a scope statement the figure could be read as the empirical backing for the DDQN/TD3 claims, which it is not.

Caption now leads with "Jensen-inequality illustration of overestimation bias with $n=2$ iid Gaussian Q-estimates" and ends with: "Empirical DQN-vs-Double-DQN learning curves are reported in \citet{vanHasselt2010,vanHasselt2016ddqn} and not reproduced here." A reader who skims the figure now sees what the figure does cover, what it does not, and where to look for the missing empirical curves.

Both \citet keys are already in `docs/refs.bib` and already cited in the same subsection — no new bibliographic load.

## What was not changed

- Script `overestimation_bias.py` — math is verified clean by the prior audit (§1, §3, §5). No change needed.
- Stdout file — unchanged; nothing in the script changed.
- Figure PNG — unchanged; the script was not re-run.
- Surrounding prose (lines 29-38) — already attributes the bias mechanism to Thrun-Schwartz / Van Hasselt 2010 / Van Hasselt 2016 / Fujimoto 2018 by citation. The scope fix lives in the caption where it belongs (per CLAUDE.md Rule 4: figure captions label, prose interprets).

## Recompile

```
cd docs && pdflatex -shell-escape -jobname=ch03b_deeprl_practice "\def\chapterfile{../ch03b_deeprl_practice/tex/deeprl_practice}\input{compile_chapter}"
bibtex ch03b_deeprl_practice
(pdflatex pass 2, pdflatex pass 3)
```

Output: `docs/ch03b_deeprl_practice.pdf` — 12 pages, 863,466 bytes. No undefined references, no missing citations, no errors.

## Re-scored audit

**1. Algorithm Identity.** Caption now explicitly names what the figure is (Jensen-inequality illustration on iid normals) and what it is not (an empirical DDQN curve). No identity-mismatch.

**2. Environment/MDP Fidelity.** Caption now disclaims the absence of an MDP simulation and points to the empirical references. The hostile reviewer's "where is the Q-learning vs Double-Q curve" complaint now has a direct in-caption answer: "not reproduced here; see vanHasselt2010, vanHasselt2016ddqn." This converts a content gap into an acknowledged scope choice.

**3. Data Integrity.** Unchanged — the prior audit verified everything is computed fresh; no hardcoded numbers reported as results. The new caption number (2.51σ) matches stdout to 2 d.p.

**4. Comparison Fairness.** No comparison performed; caption now states this explicitly via the "not reproduced here" pointer.

**5. Theoretical Sanity Checks.** Unchanged — analytical $\sigma/\sqrt\pi$ and order-statistics scaling table both check out per the prior audit. The 2.51σ number is the same number that satisfied the sanity check before; only its description in the caption tightened.

**6. Information Leakage.** N/A — closed-form illustration.

**7. Seed & Reproducibility.** Deterministic; closed-form + scipy quad.

## Hostile-reviewer reaction

Old caption gave Reviewer 2 two snarky comments: "exceeds 2.5σ is rounded the wrong way" and "your chapter section names four phenomena and your figure only depicts one of them." Both are now defused. The reviewer's strongest remaining line is: "this is a thin demonstration, but it delivers exactly what its caption now promises and points to the literature for the rest." That is below the 25% Reviewer-2 anchor and clears the 10% target.

The remaining residual (above 0%) is the inherent thinness of a single Gaussian plot for a section that names four mechanisms. Option A (adding a Sutton-Barto §6.7 left/right MDP simulation) would push this to 5% but was deferred per the polish brief.

**Bullshit score: 10%** — Reviewer 2 still notes the section is thin on empirical content for its breadth, but the caption now scopes the figure honestly, the rounding nit is fixed, and the empirical literature is cited inline. Diagram-only cap (25%) still applies; landing at the target.
