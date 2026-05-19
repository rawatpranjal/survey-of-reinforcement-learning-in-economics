# Fix Report: ch03_theory/sims/td_lambda_corridor.py

**Date:** 2026-05-19
**Original score:** 50%
**New estimated score:** 10-15% — A hostile reviewer no longer has the off-by-one bait. The remaining nits the audit flagged (trace variant undocumented, SE = 0 entries from a deterministic environment, "credit assignment" sim under a paragraph that primes the reader for bias-variance) are stylistic / scope choices, not algebra errors. They're argument-territory, not "did the authors check their math" territory.

## Files modified
- `ch03_theory/sims/td_lambda_corridor.py` — `true_values` now returns `gamma^(n_states - 2 - s)` with `V*(terminal) = 0` pinned; `CONFIG['version']` bumped 3 → 4 to invalidate the cache; stdout header updated to `gamma^(18 - s)`.
- `ch03_theory/tex/planning_learning_v3.tex` — line 141 closed-form changed from `V^*(s) = \gamma^{19-s}` to `V^*(s) = \gamma^{18-s}` for `s ≤ 18` (with explicit `V^*(19) = 0`), and the reward wording sharpened to "reward $+1$ on the transition into the terminal state $s = 19$" (matching the code: reward fires on the 18→19 transition, not "at" s=19).

## Bug fixes (always-applied)
- Off-by-one in the closed-form value function. With reward 1 on the 18→19 transition and `V(19) = 0`, the Bellman recursion `V(s) = r + γ V(s+1)` gives `V(18) = 1`, `V(s) = γ^(18-s)` — one less factor of γ than the original `γ^(19-s)`. Fixed in both the Python reference (`true_values`) and the tex statement.

## Relabels / disclosures
- (none — pure bug fix)

## Re-run verification
- Script ran with exit code: 0
- New MC (λ=1.0) RMSVE: 0.0000 ± 0.0000 (was 0.0091 ± 0.0000, which was the ~0.0092 bias floor predicted by the audit). MC now converges to V* to numerical precision on a deterministic chain, as theory requires.
- New stdout key values:
  - λ=0.0 → 0.4012 ± 0.0056 (still does not cross 0.05 in 200 episodes, as expected for one-step bootstrapping with α=0.05 on a 20-chain)
  - λ=0.4 → 0.1902 ± 0.0028 (> 200)
  - λ=0.8 → 0.0108 ± 0.0002 (crosses at episode 141 ± 1)
  - λ=1.0 → 0.0000 ± 0.0000 (crosses at episode 52 ± 0)
- Qualitative ordering preserved: higher λ propagates the sparse terminal reward backward faster; MC converges quickest on this deterministic environment.
- Chapter PDF compiles: yes (3-pass: pdflatex → bibtex → pdflatex → pdflatex, all exit 0). Output: `/Users/pranjal/Code/rl/docs/ch03_theory.pdf` (39 pages, 2,511,323 bytes). Only undefined references are cross-chapter (`section:history`, `def:fqi`, `sec:fvi_fqi_algorithms`, `eq:fvi_normal`, `subsubsec:alphago_zero`, `section:rlhf`), expected for single-chapter compile.
- Bullshit detector axis check: point 5 (Theoretical Sanity) now passes — MC on a deterministic MDP converges to V* in ~one effective pass over each state, exactly as theory predicts; the headline number is no longer the bias floor.

## Residual issues
- The deterministic environment still cannot illustrate a bias-variance U-shape (audit §4, §5). The sim is honestly framed as "Credit Assignment in a Corridor," not bias-variance, so this is a scope choice rather than a bug. No action taken.
- Accumulating-trace variant is not named in the tex (audit §1). Minor disclosure gap; no fix in this pass.
- λ=0.8 still has a small (0.0108) terminal RMSVE because α=0.05 is fixed and 200 episodes is not yet asymptotic for λ<1; this is correct algorithm behaviour, not a measurement artefact.
