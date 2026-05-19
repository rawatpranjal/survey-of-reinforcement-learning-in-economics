# Audit: ch03_theory/sims/lqc_fvi_fqi.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (Section 3, `\label{sec:lqc_fvi_fqi}`, lines 173-193). Theory feed-in at `sec:fvi_fqi_theory`, lines 154-170. Contrasted with `sec:bm_fvi_fqi` (Brock-Mirman, lines 195+).
**Cited paper PDFs read:** `papers/munos_szepesvari2008_finite_time_fvi.md` (abstract + Sections 1-2). Also relevant in `papers/`: `Analyzing_Approximate_Value_Iteration_Algorithms.md`, `gaur2023_fitted_q_iteration_nn.md`, `farahmand2010_error_propagation_api_avi.md`, `bertsekas2011_approximate_policy_iteration_survey.md`. Bertsekas DP textbooks available but not opened for this audit.

## 1. Algorithm Identity

The update rules are correct *as projected-exact VI / projected-exact Bellman-iteration*, not strictly as the sampling-based Fitted Value Iteration in Munos & Szepesvari 2008.

- **FVI (lines 173-218):** `Phi_V = [x, x^2]`; per iteration, computes `V_target = (R + gamma * V_k(x'))).max(axis=1)` on the full (N_X × N_U)=301×201 grid, then solves OLS `theta = lstsq(Phi_V, V_target)`. Bellman target uses bootstrapped `V_k = Phi @ theta_V`, not oracle `V*`. ✓
- **FQI (lines 221-308):** `Phi_Q = [x, x^2, u, u^2, xu]`; per iteration, computes target `y = R + gamma * max_u' Q_k(x', u')` *parametrically* (Q is reparameterized via `theta_Q`, no interpolation over u' — line 248-253), then OLS-fits new `theta_Q`. Target uses bootstrap, not oracle. ✓
- **DQN (lines 311-401):** Standard online DQN with replay buffer, hard target network (every 500 steps), ε-greedy, MSE loss on TD target. The output head emits Q-values for all `N_U=201` actions from a 1-d state input. Reward scaled by 1/20.

**Issue (Reviewer 2 level):** The script's "FVI / FQI" use the *full deterministic grid* (every (x,u) pair) and the *true transition map* (`Xnext = a*XX + b*UU`, plus `np.interp` onto the V_k grid for FVI, or analytical state-prop for FQI). There is no Monte-Carlo sampling, no random sample set of size N per iteration, no noise. Munos & Szepesvari's FVI is sampling-based — its whole point is the variance term `O(1/sqrt(N))` plus concentrability under a sampling distribution μ. What this script implements is closer to *exact projected value iteration* with a known model — sometimes called approximate VI (AVI). The bias from the iteration is the inherent Bellman error term in the bound (equation \eqref{eq:fvi_error_bound} in the tex), which is zero here because Q* ∈ span(Φ_Q). The "fitted" label is technically permissible (lstsq is a projection step) but pedagogically blurs the distinction the chapter wants to make about *sample complexity*. The tex sidesteps this by speaking only of "projected Bellman equations" in §sec:fvi_fqi_theory paragraph 3, but the bound in eq. \eqref{eq:fvi_error_bound} explicitly references N samples — and the sim has no N to vary. Reviewer 2 will note this gap.

**Issue (minor):** The tex claim (line 170) "FVI converges to V* in a single projected iteration" when V* ∈ span(Φ) is mathematically true *iff* the iteration is V_{k+1} = Π T V_k starting from V* itself, or if the operator is started near the fixed point. The sim shows 9 FVI iterations to converge from V_0 = 0 (because each iteration contracts at rate γ even with exact projection; you need ~log(1/eps)/log(1/γ) ≈ 270 iters to hit eps=1e-10, but the convergence test on Δθ < 1e-9 trips early). Not wrong, but the tex assertion is misleading. The sim quietly contradicts it.

## 2. Environment / MDP Fidelity

- a=0.5, b=1.0, γ=0.95: matches tex line 176. ✓
- Reward `r(x,u) = -(x² + u²)`: matches tex. ✓
- State grid 301 pts on [-4,4], action grid 201 pts on [-2,2]: matches tex. ✓
- Invariance check: `Xnext.min() ≥ X[0]`, `Xnext.max() ≤ X[-1]` enforced via `assert` at line 553. ✓
- Riccati P solved two ways (closed-form ARE + fixed-point) and `assert abs(P - P_fp) < 1e-6` at line 101. ✓ Yields P ≈ 1.1294, matches tex's "P ≈ 1.129."
- Q* coefficients (c_xx = -1.2682, c_xu = -1.0729, c_uu = -2.0729) match tex's "-1.268 x² - 1.073 xu - 2.073 u²" exactly.

No mismatches between code and tex.

## 3. Data Integrity

- `compute_exact_vi`, `compute_fvi`, `compute_fqi`, `compute_dqn` all actually run the iterations and return computed values. No hardcoded results masquerading as data.
- Riccati P is computed analytically and used as ground truth — appropriate for an oracle.
- Stdout reports match what the code computes (verified by cross-checking `compute_fvi`'s `print` blocks with the stdout file: FVI 9 iters, error 3.23e-04, P recovered 1.1294 — all consistent).
- Cache hits on all four components (`exact_VI`, `FVI`, `FQI`, `DQN`) on the stored run, so reported numbers are reproducible from cache.
- LaTeX table written from `theta_V`/`theta_Q` directly (lines 467-477), not from boilerplate.

Clean on this axis.

## 4. Comparison Fairness

This is the weakest axis. The three methods are *not* comparable on a sample-budget basis:

| Method | "Data" budget per iteration | Iterations | Stochasticity |
|--------|----------------------------|------------|---------------|
| FVI | Full grid: 301×201 = 60,501 (x,u) targets, exact transitions | 9 | None |
| FQI | Same full grid: 60,501 targets, analytical Q-targets | 10 | None |
| DQN | 256 sampled transitions × 100,000 steps = 2.56e7 transitions, with replay; transitions from grid indices | 100,000 | ε-greedy exploration |

FVI/FQI effectively have *zero sampling noise* and a *known transition operator*. DQN has stochastic exploration, replay-buffer dynamics, neural-network optimization noise, and reward scaling that re-shapes the loss landscape. The errors reported (3e-4, 9e-5, 5.6e-1) reflect this asymmetry, not a fair head-to-head. The tex (line 178) does say "DQN... with no prior knowledge of the feature basis," which acknowledges the asymmetry, but does not frame the comparison as unfair-by-design. Reviewer 2 will write: "you give the linear methods a noise-free oracle on the full grid and the deep method 100k noisy samples and report the linear methods 'beat' DQN — this is not a comparison, it is a setup."

Same evaluation metric used for all three (max-norm error of V_hat vs V_star on the X grid), so the eval protocol is consistent. ✓

The chapter's narrative purpose — "when Q* ∈ span(Φ), the linear-feature methods get exact recovery; DQN is a sanity check that a generic deep approximator also converges" — is defensible *if* the tex states the comparison is illustrative rather than horse-race. Currently the tex does not say this explicitly. Soft flag.

## 5. Theoretical Sanity Checks

- **FVI weight recovery:** `theta_V[1] = -1.1294`, matching `-P = -1.1294` to 4 decimal places. ✓ Asserted in code (line 208).
- **FQI weight recovery:** `theta_Q = [0, -1.2682, 0, -2.0730, -1.0729]` vs analytical `[0, -1.2682, 0, -2.0729, -1.0729]`. ✓ Coefficients match to within 1e-4. Asserted.
- **Munos-Szepesvari bound:** Since the inherent Bellman residual is zero (Q* exactly representable) and there is no sampling variance (full grid), the bound predicts only the geometric `γ^K` decay term should drive error. After 9-10 iterations at γ=0.95, the theoretical floor is `0.95^10 × ||V_0 - V*||_∞ ≈ 0.60 × max|V*| = 0.60 × P × 16 ≈ 10.8`. But the iteration is on θ, not on V directly, so the per-iter contraction acts on `||θ_k - θ*||`, and once OLS hits the fixed-point of the projected Bellman map (γ-contraction in θ-space), residual error is at machine-precision-level. The reported 3.23e-4 vs V* and 1.04e-3 vs exact VI are consistent with this regime. ✓
- **Exact VI error (1.12e-03 vs analytical):** This is the *discretization* error of tabular VI on a 301-point grid — `O(h_X²) = O((8/300)²) ≈ 7e-4`. Reported value 1.12e-3 is consistent. ✓
- **Curious-but-correct:** FVI error vs V* (3.23e-04) is *smaller than* exact VI's error vs V* (1.12e-03), because the smooth quadratic basis interpolates *through* the discretization grid. This is correct and worth noting in the tex — currently it isn't.
- **DQN error 0.56:** Plausible given 201 discrete actions, ε-greedy with final eps=0.05, reward scaling, and 100k steps. Not a tight match to any specific theoretical bound, but the right order of magnitude.

No theoretical violations.

## 6. Information Leakage

- FVI/FQI do **not** use P, K_opt, c_xx, c_xu, c_uu during iteration. The Riccati values are computed only for *reporting* (error vs analytical) and for assertions. ✓
- FVI/FQI features include `x^2`, `u^2`, `xu` — i.e., the basis is hand-engineered to *contain* Q*. This is *not* leakage in the dishonest sense; it is the explicit pedagogical point. The tex states this openly (line 176-177).
- FVI/FQI use the *true deterministic transition* `Xnext = a*XX + b*UU` (lines 110-116). This is "knowing the model" — Munos-Szepesvari's FVI assumes a generative model, so this is consistent with the paper. But again, it is *not* sampling-based — it is model-based projected VI. The "fitted" label can mislead readers into thinking sampling-noise was simulated.
- DQN does not see P, K_opt, or the analytical Q*. It uses the grid as an environment but does not exploit the polynomial structure. ✓
- DQN uses `Xnext_idx` (precomputed transition grid) — this is the environment, not leakage.

No outright cheating; one soft "model is known to FVI/FQI" flag.

## 7. Seed & Reproducibility

- `np.random.seed(42)` at module level (line 39); `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)` again at start of `compute_dqn` (lines 313-315). ✓
- **But: only one seed is run.** FVI/FQI are deterministic given the grid and features, so seed-variation is moot for them. DQN is highly stochastic (replay sampling, ε-greedy action choice, weight init) and is reported from a single seed.
- CLAUDE.md project rule: "Run each method across multiple seeds (minimum 10) and report means and standard errors." This sim runs N=1 for DQN.
- **No standard errors anywhere.** The error 5.64e-01 reported for DQN is a single-seed point estimate.
- Reproducibility from the same cached run is fine (deterministic given the seed). Reproducibility across seeds — unknown.

Hostile reviewer flag: the chapter's tex claims "DQN also converges (error 5.6×10^-1)" as if this is a robust finding. From one seed it could be a fluke.

## Hostile-Reviewer Summary

The core arithmetic is right and the algorithm-identity claims hold for FVI and FQI as *projected exact value iteration* on a known model with hand-picked polynomial bases. The Riccati oracle is solved two ways and verified. Weight recovery to 4 decimals is genuine. The cited tex is internally consistent with the sim numbers.

The hostile reviewer's complaints are:

1. **Single seed for DQN.** The chapter project rules require ≥10 seeds with SEs. DQN's 5.6e-1 error is a point estimate of unknown variance. Cheap to fix (loop over seeds; replace prose with mean ± SE).
2. **"Fitted" methods are actually model-based projected VI.** No sampling noise, no `O(1/sqrt(N))` term, no concentrability story — yet the theory paragraph (eq. \ref{eq:fvi_error_bound}) is built around exactly those terms. The sim demonstrates only the `γ^K` and inherent-Bellman-error terms, not the variance term. The tex should either (a) state explicitly that the sim is in the noise-free limit and demonstrates the bias term only, or (b) inject Gaussian noise into the targets and show variance shrinkage as N is varied. As stands, the sim does not exercise the full bound it cites.
3. **DQN-vs-linear comparison is asymmetric by design** (full grid + known model + correct basis vs 100k noisy samples + no basis). The tex hints at this ("with no prior knowledge of the feature basis") but does not frame the comparison as illustrative rather than competitive.
4. **Tex line 170 says "FVI converges in a single projected iteration" when V* ∈ span(Φ); sim shows 9 iters.** Not contradictory if you read carefully (the algorithm starts from V_0 = 0, not V*), but a careful reader will catch the apparent inconsistency.

None of these are method-vs-name violations or wrong-attachment issues. The sim is what it claims to be in code; the gap is between *what the tex theory paragraph promises the sim will illustrate* and *what the sim actually exercises.* That gap is a Reviewer-2 issue: snark, request for clarification, no rejection.

**Bullshit score: 30%** — Reviewer 2 catches the single-seed-DQN, the noise-free-FVI-vs-noisy-DQN asymmetry, and the "fitted ≠ sampling-based here" wording mismatch with the cited bound; the substance (weight recovery, Riccati match, ordinal ranking) survives revision.
