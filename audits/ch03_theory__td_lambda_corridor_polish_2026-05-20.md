# Polish Report: ch03_theory/sims/td_lambda_corridor.py

**Date:** 2026-05-20
**Original score:** 50% (audit 2026-05-19)
**Post-Phase-1 score:** 10–15% (fix 2026-05-19)
**New estimated score:** ~5% — A hostile reviewer no longer finds the off-by-one bait or any quantitative misalignment between prose and table; remaining residuals (deterministic env cannot show bias-variance U-shape, accumulating-trace variant unnamed in tex, SE = 0 entries from determinism) are scope choices already disclosed honestly by the subsection title and are not reviewer-stoppers.

## Files modified
- `ch03_theory/tex/planning_learning_v3.tex` — single prose edit on the results sentence (line 143). Previously: "TD($\lambda = 1$) reaches RMSVE $< 0.05$ in fewer episodes than TD(0), which must wait for many episodes before the reward signal diffuses back to early states through one-step bootstrapping alone." Now: "TD($\lambda = 1$) crosses RMSVE $< 0.05$ at episode 52 and converges to $V^*$ to numerical precision, while TD(0) does not cross the threshold within 200 episodes because the reward must diffuse back through one-step bootstrapping alone." This pins the prose to the actual table numbers (52 episodes, $> 200$) rather than the vaguer "fewer episodes than" framing, and surfaces the now-true claim that MC converges to numerical precision on this deterministic chain.

## Bug fixes (always-applied)
- (none — Phase 1 closed the only algebra bug; this is a prose-tightening pass)

## Relabels / disclosures
- (none)

## Re-run verification
- No sim re-run (no Python change). Existing stdout and table reflect Phase 1 numbers: $\lambda=0.0 \to 0.4012 \pm 0.0056$ ($> 200$), $\lambda=0.4 \to 0.1902 \pm 0.0028$ ($> 200$), $\lambda=0.8 \to 0.0108 \pm 0.0002$ (141 $\pm$ 1), $\lambda=1.0 \to 0.0000 \pm 0.0000$ (52 $\pm$ 0).
- Chapter PDF recompiles cleanly: 3-pass pdflatex → bibtex → pdflatex → pdflatex, all exit 0. Output: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (2,542,313 bytes, last-modified 2026-05-19 17:08). Only undefined references are cross-chapter (`section:history`, `def:fqi`, `sec:fvi_fqi_algorithms`, `eq:fvi_normal`, `subsubsec:alphago_zero`, `section:rlhf`), expected for single-chapter compile.
- Prose-to-table cross-check: 52 in prose matches `eps_to_thresh_mean` in `td_lambda_corridor.tex` line 12. "$> 200$ for TD(0)" matches table line 9. "converges to $V^*$ to numerical precision" matches the $0.0000 \pm 0.0000$ final RMSVE in line 12.

## Residual issues
- Deterministic environment cannot illustrate a bias-variance U-shape (Sutton-Barto Fig 7.6 territory). Already disclosed via the section title "Credit Assignment in a Corridor"; rank order on this metric is monotone in $\lambda$, which the table and figure both show. No action.
- Accumulating-trace variant not named in the tex (audit §1). Footnote 134 mentions replacing / Dutch traces as practical variants but does not specify which the script uses. Cosmetic; a sharp reviewer might raise it, but it does not affect any number. No action this pass.
- Canonical-order vs rank-order presentation: the table and figure walk $\lambda$ in ascending order (0.0, 0.4, 0.8, 1.0) rather than rank order by RMSVE (1.0, 0.8, 0.4, 0.0). For a sweep parameter that traces a theoretical curve, sweep-order is the convention in the TD($\lambda$) literature and reads more naturally with the prose argument ("higher $\lambda$ ... than TD(0)"). Flagged for awareness; no action this pass.

**Bullshit score: 5%** — Prose, table, figure, and algebra all agree. The only quibbles a hostile reviewer has are scope choices (deterministic env, trace variant unnamed), and the subsection title "Credit Assignment" makes the scope honest.
