# Audit: ch03_theory/sims/mm_surrogate_trpo.py

**Date:** 2026-05-19
**Diagram-only:** YES (1D toy objective, no MDP, no stochasticity, no seeds). Cap applies: 25% unless visual contradicts caption.
**Cited tex file(s):** `ch03_theory/tex/planning_learning_v3.tex` lines 398–412 (the `\subsubsection{Trust Region Methods}` block; figure inclusion at line 402; caption at lines 403–410). Surrogate-equation context at lines 386–396; explicit lower-bound prose at line 398.
**Cited paper PDFs read:** `ch03_theory/papers/Schulman2015_trpo.md` (Sections 2–3 specifically: surrogate L, Theorem 1, Algorithm 1, the MM identity M_i(π) = L_πi(π) − C·D_KL^max(πi, π), and the explicit constant C = 4εγ/(1−γ)²). Kakade-Langford `kakade2002_natural_policy_gradient.md` present but not re-read; the relevant Kakade-Langford lower bound is reproduced verbatim in Schulman2015 Section 2.

## 1. Algorithm Identity

The script does NOT implement TRPO's surrogate L(θ) = E[π_θ(a|s)/π_old(a|s) · A^π_old(s,a)]. There is no MDP, no policy, no importance ratio, no advantage. It computes a generic quadratic majorizer of a 1D toy function J(θ) = sin(2θ) + 0.3θ − 0.08θ². The "surrogate" is

  L(θ | θ_old) = J(θ_old) + J'(θ_old)(θ − θ_old) − c(θ − θ_old)².

This is the *MM template* (linear-plus-quadratic underestimate of a smooth function), not the *TRPO surrogate*. Two specific deviations from Schulman2015:

- Schulman's lower bound is η(θ) ≥ L(θ) − C · D_KL^max(θ_old, θ) with C = 4εγ/(1−γ)² and ε = max_{s,a}|A^π_old(s,a)|. The penalty is in KL space. The script's penalty is in raw parameter space (θ − θ_old)². Without a parameterization linking θ to a policy, this conflation passes only as a cartoon.
- c is *fitted numerically* (`compute_c`) as the smallest constant that makes L ≤ J on a 1000-point grid in [−1, 5], times a 5% safety margin. It is not the theoretical C. The fitted c (2.12 around θ_old=1, 1.42 around θ_old=0) drifts with θ_old in a way the theory does not predict, because c is doing curve-fitting work, not bounding work. Plotting this and labeling it L(θ|θ_old) per Theorem 1 of Schulman2015 is misleading at the caption level — the caption (line 405) explicitly invokes "the trust region" and "the guaranteed improvement," tying the picture to the TRPO theorem rather than to a generic MM cartoon.

The tex prose around the figure (line 398) reads "L(θ) is a local lower bound on J(θ) that is tight at θ_old: L(θ_old) = J(θ_old) and L(θ) ≤ J(θ) within the trust region." That is the property the picture *actually* shows. So the prose-claim is verifiable from the figure; the issue is that the *named object* in the caption is "the surrogate L(θ|θ_old)" with TRPO context immediately preceding, which oversells a generic MM cartoon as a TRPO-specific picture.

## 2. Environment / MDP Fidelity

No MDP, no environment, no policy class. J(θ) is invented to look like a non-convex bumpy function. This is appropriate for a 1D illustration but it means the picture is *not* in any sense a TRPO step on any MDP; it is "what an MM step looks like, drawn in 1D." Tex caption does not promise an MDP, so this is not a fidelity violation per se, but a reader who took the caption literally would expect L to be the TRPO importance-weighted surrogate. It is not.

## 3. Data Integrity

`generate_outputs()` actually runs. Numbers reported in stdout match the figure: the single-step c = 2.12 around θ_old = 1.0 is recomputed from the grid each call; iterates θ_0..θ_4 are produced by genuine MM iteration with `surrogate_argmax` (closed-form max of the quadratic clipped to the trust region [θ − δ, θ + δ], δ = 1.2). No hardcoded curves. Computation is deterministic (no randomness, so no seeds needed). The `--data-only` flag exits with a message; `--plots-only` runs normally — matches the diagram-only convention in CLAUDE.md.

Minor: the `surrogates_c` list only stores `c_k` for the iterate panel; the single-panel uses its own `c_single`. Both are recomputed in the same call. Cache state is irrelevant since the script does not cache.

## 4. Comparison Fairness

Not applicable in the conventional sense (no baseline to compare against). The implicit comparison is L vs J on the same θ grid, which is fair: both are evaluated on `theta_range = np.linspace(-1, 5, 1000)` with the same closed-form expressions.

## 5. Theoretical Sanity Checks

Three properties must hold for an MM surrogate; all are verified empirically:

1. **Tangency at θ_old.** By construction, L(θ_old | θ_old) = J(θ_old) + 0 − 0 = J(θ_old). Algebraically exact, not numerical.
2. **First-order match (gradient tangency).** dL/dθ|_{θ_old} = J'(θ_old), again by construction. Schulman2015 emphasizes this as the property that lets the bound be tight to first order; the script inherits it for free from the Taylor expansion. Not explicitly verified in stdout but mechanically guaranteed.
3. **Global lower bound L ≤ J on the visible domain.** Stdout reports "0 violations out of 1000 points" for the single-step panel and no warnings during iteration. This is enforced by `compute_c`, which fits c to be the smallest value that achieves the bound (plus 5% margin). The bound holds *only on the grid*, not analytically; for a more aggressive θ the bound could fail. Acceptable for a 1D cartoon.
4. **Monotonic improvement.** Stdout: "Monotonic improvement: YES." J(θ_0) = 0.000 → J(θ_1) = 1.189 → 1.1899 → 1.1899. Converges to numerical global max θ* ≈ 0.826 with J(θ*) ≈ 1.190. Matches the figure.

One reviewer-grade concern: because c is fitted globally on [−1, 5], it is roughly twice as large as a *local* MM majorizer would need (since the surrogate is then required to dominate J on a wide range, not just inside the trust region). This makes the right panel's surrogates look very tight, almost coincident with J at the iterate. A theoretical-C choice (Schulman's 4εγ/(1−γ)², or the simpler Kakade-Langford analogue) would give a looser quadratic. The picture's "L is a tight lower bound" impression therefore overstates the tightness of the actual TRPO bound, which in practice is so loose that Schulman explicitly switches from penalty to constraint form for this reason ("if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small" — Schulman2015 Section 4). This is a *pedagogical* misrepresentation rather than a math error, but it matters because the figure is used to motivate the MM picture.

## 6. Information Leakage

No π_new, no advantage estimation, no Monte-Carlo evaluation. The surrogate is a deterministic function of J', J(θ_old), and a fitted c. There is nothing to leak. (In a real TRPO sim, leakage would mean using π_new's advantages instead of π_old's; here, both L and J are closed-form, so the question does not apply.) Note however that c itself is fit by *peeking at J* across the whole θ range, which is the moral analogue of leakage: in a real setting, you would not have access to J(θ) for all θ to choose your majorizer's curvature. This is fine for a diagram but worth flagging — it is the reason the picture looks tight.

## 7. Seed & Reproducibility

Deterministic: no randomness anywhere, no seeds needed. Output reproduces bit-identically on rerun. Stdout file is regenerated each run. There is no `_stdout.txt` file currently saved (the convention says scripts should write one, but for diagram-only scripts the runner usually captures stdout separately; this is a minor housekeeping miss, not an audit-blocker).

## Hostile-Reviewer Summary

The diagram demonstrates the MM lower-bound mechanism: surrogate tangent at θ_old, dominated by J globally, sequential maximization producing monotone improvement to θ*. Mechanically correct as a 1D illustration of majorization-minimization.

But it is labeled as the TRPO surrogate (caption invokes "L(θ|θ_old)" with TRPO equation cited immediately above), and it is not. The script's quadratic penalty lives in parameter space, not KL space; the curvature c is fitted to the visible grid rather than derived from Schulman's C = 4εγ/(1−γ)². The picture works as a *cartoon* of MM-for-policy-optimization but does not depict TRPO's actual bound. A reviewer who cared about this distinction (and Schulman explicitly does — he warns that the theoretical C gives steps too small to use, motivating the switch from penalty to constraint) would flag this as imprecise framing of a TRPO-specific theorem. Whether that gets a snarky comment or a major-revise depends on the reviewer's pedantry.

Diagram-only cap applies. The visual does NOT contradict the caption — L is below J, touches at θ_old, iterates climb monotonically. So the cap stays at 25% unless one argues the conflation of (θ − θ_old)² with KL penalty rises to a visual contradiction, which is a stretch.

**Bullshit score: 25%** — Reviewer 2 catches that the quadratic-in-θ majorizer is being passed off as the TRPO KL-penalized surrogate, and that c is curve-fitted rather than the theoretical 4εγ/(1−γ)². The substance (MM produces monotone improvement to a non-convex optimum) holds, so it survives revision with a tightened caption ("schematic MM illustration; the TRPO surrogate's penalty is in KL space and uses Schulman's constant C").
