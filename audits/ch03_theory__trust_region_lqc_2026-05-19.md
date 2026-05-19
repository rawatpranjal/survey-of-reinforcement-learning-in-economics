# Audit: ch03_theory/sims/trust_region_lqc.py

**Date:** 2026-05-19
**Diagram-only:** YES — fully analytical, no Monte Carlo RL training, no torch. Computes
J(θ) on a 100×100 grid via closed-form Lyapunov, Fisher info via closed-form stationary
covariance, TRPO step in closed form, PPO mask via 200 sampled state-action pairs from
the stationary distribution.
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex`
(lines 386–439, equations `eq:trpo`, `eq:trpo_step`, `eq:ppo`, figure `fig:trust_region_lqc`).
**Cited paper PDFs read (in `ch03_theory/papers/`):**
- `Schulman2015_trpo.pdf` (present, not re-read this audit but referenced)
- `kakade2002_natural_policy_gradient.pdf` (present)
- `shani2020_adaptive_trust_region.pdf` (present)

Diagram-only cap of 25% per CLAUDE.md applies *unless the diagram visually contradicts
the caption or has a substantive algorithmic identity defect*. I argue below that a
real algorithmic-identity tension exists (factor of 2 in step scale), so the score is
nudged toward the cap but not above.

## 1. Algorithm Identity

TRPO step in code (lines 194–204):
```
scale = np.sqrt(delta / (g @ Minv_g))
theta_trpo = theta_old + scale * Minv_g
```

This implements `Δθ = sqrt(δ / (gᵀ M⁻¹ g)) · M⁻¹ g`, which solves
`Δθᵀ M Δθ = δ` exactly (verified at line 297: `kl_trpo = 0.10000000`).

But the tex equation `eq:trpo_step` (line 394) states:
`Δθ = sqrt(2δ / (gᵀ F⁻¹ g)) · F⁻¹ g`

with the standard convention `KL ≈ (1/2) Δθᵀ F Δθ`. The factor of 2 is missing
in the code's step. Under the standard convention, the code's step has
`(1/2) Δθᵀ F Δθ = δ/2`, i.e. it only saturates *half* the KL budget. The
"verification" at line 297 confirms `Δθᵀ M Δθ = δ`, not `(1/2)Δθᵀ M Δθ = δ`,
so the verification is self-consistent with the *code's* convention, not the tex's
convention.

This is partly papered over by the figure caption at line 380–381 of the tex
(for a *different* figure, info_geometry), which defines the KL unit ball as
`Δθᵀ F Δθ ≤ δ` (without the 1/2). The chapter is therefore internally inconsistent:
two different conventions for the KL second-order approximation appear within the
same section. The trust_region_lqc figure inherits the code's no-1/2 convention.

A hostile reviewer who reads eq:trpo_step (the canonical TRPO formula in the
literature, e.g., Schulman 2015 Eq. 12) will compute that the displayed TRPO step
under-shoots the saturating step by a factor of sqrt(2) ≈ 1.414. The qualitative
picture (ellipse shape, axis orientations, which side of the unstable region
θ_trpo lands) is correct; the precise step magnitude is off by sqrt(2) relative
to the tex's stated formula. Score-relevant.

PPO step (lines 207–225): finds the max scalar `t` along normalized gradient
direction such that θ_old + t·g_norm stays in the PPO mask (fraction of samples
with ratio in [1−ε, 1+ε] ≥ 0.5). This is a heuristic visualization of a
non-convex feasible region, not a faithful PPO update. PPO in practice does
multiple gradient steps on the clipped surrogate; the code shows a *boundary*
of the region, not the trajectory PPO would take. The tex caption describes
it accurately as a "50% ratio-clip band over 200 sampled state-action pairs" —
honest, but I note this is illustrative.

Fisher information `M = Σ_x / σ²_a` (lines 95–107): for Gaussian policy
π(u|x) = N(−θᵀx, σ²_a), Fisher info wrt θ is `E[xxᵀ]/σ²_a = Σ_x/σ²_a`.
This is correct under the on-policy state distribution under θ_old. ✓

## 2. Environment / MDP Fidelity

Code sets:
- `A = [[0.5, 0.2], [0.0, 0.8]]`, `B = [[0.5], [1.0]]`, `Q = diag(2, 1)`, `R = 0.5`,
  `γ = 0.95`, `σ²_w = 0.5`, `σ²_a = 1.0`.

Tex caption (line 426–432) describes "central bank learns a Taylor rule
u = −(θ₁ x₁ + θ₂ x₂)" with x₁ = output gap, x₂ = inflation gap. The script's
panel-1 labels match. ✓

The dynamics `A = [[0.5, 0.2], [0.0, 0.8]]` with `B = [0.5, 1.0]ᵀ` are not stated
explicitly in the tex (only "IS-Phillips dynamics" — a one-line inline comment in
the code). For a stylized illustration this is fine, but a reviewer might want
to see the matrices justified or sourced. The matrices are stable in open loop
(eigenvalues 0.5, 0.8), and the optimal closed-loop is also stable. ✓

Cost is `(xᵀ Q x + uᵀ R u)`, discounted with γ. Lyapunov equation for
`J(θ) = -tr(P · Σ_0)` with `Σ_0 = I` is implemented correctly via the
`scipy.linalg.solve_discrete_lyapunov` reparameterization for the discount factor
(lines 73–79). DARE solution via `solve_discrete_are(√γ A, √γ B, Q, R)` (line 248) is
the standard discount-folding trick. Lyapunov residual at θ* is 2.78e-17 (line 562
of stdout): the optimal P satisfies the algebraic equation to machine precision. ✓

## 3. Data Integrity

`compute_data()` (line 340) calls `compute_or_load` with the trust_region component,
which calls `_run_trust_region_analysis` (line 230). All quantities (θ*, J grid,
M, λ, θ_trpo, θ_bad, ppo_mask, KL values) come from the function. No hardcoded
expected values appear in the printed verification — the printed `delta = 0.1` and
`9*delta = 0.9` are constants from the CONFIG, and the computed KL values (`kl_trpo`,
`kl_bad`) are checked against them.

The stdout shows `Cache hit: trust_region`, meaning the last run used a cached
result. Cache config (line 44–52) includes A, B, Q, R, γ, σ_w, σ_a, δ, ε,
θ_old, and N_GRID, so config-driven invalidation works. ✓

The figure was generated from the cached data dict; no leakage from a stale
run since the cache is keyed on the full parameter set. ✓

One observation: `n_samples=200` and `frac_threshold=0.50` for PPO are passed
as arguments inside `_run_trust_region_analysis` (line 309–312), but they are
*not* in the CONFIG dict (line 44–52). Changing these defaults would not
invalidate the cache, even though they affect `ppo_mask`. Minor cache-correctness
defect.

## 4. Comparison Fairness

There is no real method-vs-method comparison (no learning curves, no convergence
plots over iterations). The figure compares the *geometry* of three feasible
regions (Euclidean / KL ellipse / PPO band) at a single iterate, plus the
three resulting step targets. All three steps start from the same θ_old, use
the same J(θ) surface, the same gradient g, and the same PPO sampled states.

The PPO step uses 200 samples; TRPO uses the exact stationary covariance.
This is asymmetric — TRPO gets the population Fisher, PPO gets a 200-sample
Monte Carlo estimate — but the asymmetry favors TRPO's smoothness, not its
position. The TRPO step would be unchanged if PPO used 10⁶ samples. Acceptable
for a diagram. ✓

## 5. Theoretical Sanity Checks

- DARE solution: `θ* = [θ₁*, θ₂*]` from `solve_discrete_are` (line 248), Lyapunov
  residual at θ* is 2.78e-17 < 1e-10. ✓
- `J(θ*) > J(θ_old)`: -3.6837 > -5.7399, monotonic in the right direction. ✓
- TRPO step on the ellipse boundary: `Δθᵀ M Δθ = 0.10000000` = δ to within 1e-17.
  This means the code's KL-budget convention is `Δθᵀ M Δθ ≤ δ`, NOT
  `(1/2) Δθᵀ M Δθ ≤ δ`. See §1 above. ⚠
- `θ_bad` outside ellipse: `Δθᵀ M Δθ = 0.9 = 9δ`, so θ_bad is on the 3·radius
  shell. The code constructs θ_bad as `θ_old + 3·(θ_trpo − θ_old)`. By
  construction, this is *artificial* — there's no claim it is what gradient
  descent would produce; it's a chosen illustration that "lands in the
  unstable region." The caption is honest ("unconstrained gradient step
  overshoots into the unstable region"), but the relationship between θ_bad
  and any actual unconstrained PG step is by construction, not by analysis. ⚠
- `is_stable(theta_bad)`: stdout reports it (need to check) — script prints it
  but the stdout shown didn't include those lines. The figure does show θ_bad
  inside the hatched (unstable) region in panel 2.
- PPO feasible region "non-ellipsoidal": Yes — at line 161 the mask prints what
  fraction of grid points are feasible; the figure shows it as a non-elliptical
  blob. ✓

The figure correctly conveys the qualitative claim of the tex passage: TRPO
constraint is an ellipse defined by the Fisher metric, unconstrained gradient
overshoots into instability, PPO band is non-ellipsoidal. No theoretical
sanity check is contradicted.

## 6. Information Leakage

The script is *fully analytical* — there is no notion of "training" or
"learning." It computes J(θ) closed-form (Lyapunov), the optimal θ* closed-form
(Riccati), the gradient by finite differences on the closed-form J grid (which
is fine since this is a visualization), the Fisher info closed-form, and the
TRPO step closed-form. The gradient at θ_old is computed from the J grid via
finite differences (line 164–191). This is internally consistent — no agent
is "cheating" because no agent exists.

The PPO step uses on-policy samples (states drawn from Σ_x, actions from
old-policy distribution), which is the correct sampling for PPO. ✓

The figure caption explicitly states: "policy contour lines in state space"
and "Expected return J(θ₁, θ₂) in parameter space" — the analytical nature
is reasonably implicit (parameter-space figure of the closed-form return).
But the script's header does say "Fully analytical: no RL simulation, no
torch." A reviewer reading the tex caption alone might not realize the
"step" is the closed-form NPG step, not an actual TRPO iterate from
samples. The diagram is honest about being illustrative, however. ✓

## 7. Seed & Reproducibility

`np.random.seed(42)` at line 22. The only stochastic component is the 200
PPO samples (line 117–120). N_seeds = 1 (the seed is fixed once). Since the
PPO mask is a coarse visualization (50%-fraction threshold over 200 samples),
single-seed is reasonable for a diagram, but reviewer 2 may note that a
1-seed PPO mask is a single realization of a stochastic estimator —
shading variability across seeds would be more honest. The CLAUDE.md
standard of "≥10 seeds" applies to MC simulations, and this is a
diagram, so I do not penalize.

The cache config does not include the random seed, so re-running after
changing the seed at line 22 would not invalidate the cache. Minor
cache-correctness defect. ⚠

## Hostile-Reviewer Summary

Diagram-only sim. The figure conveys the intended qualitative story
(ellipse-vs-clip-band geometry, KL trust region containing the unstable
region beyond ~3 radii, TRPO step saturating the ellipse boundary). The
analytical machinery (DARE, Lyapunov, Fisher) is implemented correctly
to machine precision.

**Real defects:**
1. The code's TRPO step uses `sqrt(δ/(gᵀM⁻¹g))` (KL = Δθᵀ M Δθ ≤ δ convention),
   but the tex equation `eq:trpo_step` displays `sqrt(2δ/(gᵀF⁻¹g))`
   (KL ≈ (1/2)Δθᵀ F Δθ convention). These differ by a factor of sqrt(2).
   The chapter is internally inconsistent: line 381 uses the no-half convention,
   line 394 uses the half convention. A careful reviewer comparing the tex
   formula to the printed kl_trpo = δ verification will catch this.
2. PPO step is a heuristic boundary search along the gradient, not an actual
   PPO update; the caption is honest enough that this likely survives review.
3. Cache config omits `n_samples`, `frac_threshold`, and `np.random.seed(42)`,
   so a maintainer changing these would silently re-use stale results.
4. `θ_bad` is constructed by stretching θ_trpo by 3×, not derived from any
   "unconstrained gradient step." Caption frames it as the gradient overshoot,
   which is illustrative-true but not literally what the code does.

None of these undermine the figure's pedagogical claim. The factor-of-2
convention drift (defect 1) is the only one a hostile reviewer would write
a snarky note about; the rest are caveats a charitable reader accepts.

**Bullshit score: 25%** — Reviewer 2 catches the sqrt(2) inconsistency
between code-step `sqrt(δ/...)` and tex eq:trpo_step `sqrt(2δ/...)` (and
the within-tex convention drift between line 381 and line 394), and notes
that θ_bad is hand-constructed rather than gradient-derived, but the
substance — TRPO ellipse vs PPO clip-band geometry on the LQC return
landscape — survives revision. Diagram-only cap of 25% reached, not
exceeded: the geometry on the page is consistent with the code, and
the closed-form quantities (Riccati, Lyapunov, Fisher) check out to
machine precision.
