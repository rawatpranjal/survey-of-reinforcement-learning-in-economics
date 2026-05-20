# Polish: ch10_causal/sims/confounded_ope.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch10_causal__confounded_ope_2026-05-19.md` (20%)
**Polish target:** ≤10%

## Changes applied (tex only, no re-run)

All three fixes are in `ch10_causal/tex/causal_rl.tex`. No code touched, no simulation rerun, cache untouched.

### Nick 1 — DR backdoor advertised but not implemented (line 138)

Resolution: **Option (B), drop the sentence.** Removed the sentence "The doubly robust variant combines the fitted action-value function $\hat{Q}(s,a)$ with backdoor-adjusted propensities, achieving consistency if either $\hat{Q}$ or the propensity model is correctly specified." from the backdoor-OPE subsection. The remaining prose now describes only what the code computes: the plug-in backdoor estimator $\hat P(s'\mid s,\operatorname{do}(a)) = \sum_z \hat P(s'\mid s,a,z)\,\hat P(z\mid s)$.

### Nick 2 — Wald vs IVVI naming gap

Resolution: Added a footnote attached to the sentence introducing Liao 2024's IV-aided Value Iteration. Footnote text states that the simulation implements the binary-action Wald formula `(E[Y|Z=1] - E[Y|Z=0]) / (E[A|Z=1] - E[A|Z=0])` of Equation~\eqref{eq:wald}, while Liao's full IVVI solves the conditional moment restriction of Equation~\eqref{eq:iv_moment} via a primal-dual reformulation which is not reproduced. The footnote is honest about the simplification and frames the Wald estimator as a demonstration of IV's unbiasedness under exclusion. Anchored to the existing equation labels so the reader can find both forms in one glance.

### Nick 3 — P_TRANS values missing from tex

Resolution: Added all four transition probabilities inline in the DGP description (§3.7), immediately after the sentence stating that the next state depends on $M_t$ and $Z_t$. Reader can now reconstruct the DGP from the tex alone: $P(s{+}1\mid s, M{=}1, Z{=}1)=0.90$, $P(\cdot\mid M{=}1, Z{=}0)=0.50$, $P(\cdot\mid M{=}0, Z{=}1)=0.40$, $P(\cdot\mid M{=}0, Z{=}0)=0.15$. The footnote that already gave rewards, $\gamma$, target policy, and the marginalised interventional probability ($0.615$) stays.

## Recompile

Chapter PDF recompiled via three pdflatex passes + bibtex from `docs/`:

```
cd docs && pdflatex -shell-escape -interaction=nonstopmode -jobname=ch10_causal \
  "\def\chapterfile{../ch10_causal/tex/causal_rl}\input{compile_chapter}" && \
bibtex ch10_causal && (pass 2) && (pass 3)
```

Output: `docs/ch10_causal.pdf`, 21 pages, 1.24 MB. One pre-existing undefined reference (`section:rl_for_ci`, the companion chapter, which isn't compiled standalone). All chapter-internal `\ref{}`s resolve; the new footnote references to `eq:wald` and `eq:iv_moment` both resolve.

## Residual issues after polish

- **Engineered DGP.** All four identification strategies hold simultaneously by construction. The tex acknowledges this (line 274) and the audit flagged it as cosmetic rather than substantive. Not addressed in this pass; would require redesigning the DGP, which is out of scope.
- **Weak-instrument fallback threshold (`|first_stage| < 0.01`).** Loose enough to admit noisy Wald ratios at instrument coefficient $0.05$, producing the dramatic outliers visible in panel (c). The audit flagged this as theoretically correct (weak-instrument pathology is *real*) but visually startling. A one-line caption note would close it; opted not to expand the caption per the chapter style rule that captions identify and prose interprets. The body paragraph on line 278 already attributes the panel-c variance to "the Wald ratio's sensitivity to instrument strength."
- **Proximal RMSE dip near $\rho \approx 0.2$–$0.4$.** Plausible (more signal in $W$ as $A\leftarrow U$ link strengthens) but not articulated in tex. Hostile reviewer would shrug rather than gripe; left untouched.

## Hostile-reviewer re-read

Reviewer 2 reading the revised section:
- Backdoor prose now matches the code line-for-line. No DR-claim-without-DR-code. Snarky comment retracted.
- IV section now has an explicit footnote acknowledging the Wald-vs-IVVI gap and pointing to both equation labels. Reviewer's "Wald is not IVVI" complaint is pre-empted and disarmed; the simplification is now part of the chapter's contract rather than a gap.
- DGP fully reproducible from prose alone now that the four transition values are tabulated inline. No need to read the code to recover what the simulation runs.

Remaining bite: the DGP is engineered to let every estimator win, and the proximal dip and weak-IV outliers are not explained. Both are below the threshold for a snarky comment but a careful reviewer might note them.

**Bullshit score: 10%** — Reviewer 2 has one mild reservation about the engineered DGP (acknowledged by the authors) and possibly a question about why proximal RMSE dips at low $\rho$; nothing rises to a methodology-attack comment. The three prose-vs-code mismatches that previously dragged the score to 20% are all closed.

**Path:** `/Users/pranjal/Code/rl/audits/ch10_causal__confounded_ope_polish_2026-05-20.md`
