# Simulation Audit: Envelope (Danskin) Theorem

- **Sim:** `appA_preliminaries/sims/envelope_theorem.py`
- **Date:** 2026-07-14
- **Type:** FULL (condensed variant — small pedagogical appendix sim, first audit)
- **Files read:**
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/envelope_theorem.py`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/envelope_theorem_stdout.txt`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/envelope_theorem.tex`
  - `/Users/pranjal/Code/rl/appA_preliminaries/sims/envelope_theorem.png` (viewed)
  - `/Users/pranjal/Code/rl/appA_preliminaries/tex/preliminaries.tex` (lines 374-408)
  - `/Users/pranjal/Code/rl/CLAUDE.md` (rubric)

## Step-3 Statement

**(i) What result is the appendix presenting?** The envelope theorem (Milgrom-Segal / Danskin), Theorem `thm:prelim_envelope`, eq. `eq:prelim_envelope`: for `V(θ)=max_a f(a,θ)` with `A` compact and `f, ∂_θ f` jointly continuous, if the maximizer `a*(θ)` is unique then `V` is differentiable and `V'(θ) = ∂_θ f(a*(θ),θ)`. The maximizer's own movement drops out to first order. The non-unique case gives one-sided derivatives equal to the extreme values of `∂_θ f` over the argmax set (Danskin's directional-derivative statement, which leaves a kink).

**(ii) What is the sim/figure evidence FOR?** Two numerical confirmations of the identity. (i) On smooth strictly-concave families `f(a,θ)=θa − a^p/p` (`p∈{2,4,6}`, closed forms `a*=θ^{1/(p-1)}`, `V=((p-1)/p)θ^{p/(p-1)}`), a central-difference derivative of the grid-maximized `V` — computed without differentiating `a*` — reproduces `∂_θ f(a*)=a*` down to the numerical floor, and numerical `V`, `a*` match their closed forms. (ii) On families of random lines `V(θ)=max_a(α_a+β_a θ)`, `V` is the upper envelope tangent to the active member, `V'=β_active` off kinks, and the kinks are the non-unique-maximizer (Danskin) case. Prose line 399 and caption `fig:prelim_envelope` / Table `tab:prelim_envelope` use this to "confirm that differentiating a numerically maximized value reproduces the partial derivative at the maximizer."

## Criteria Verdicts

### (a) CORRECTNESS — PASS
The code computes exactly the object the theorem is about.
- Smooth family: FOC `θ − a^{p-1}=0 ⇒ a*=θ^{1/(p-1)}` (line 80) and `V=((p-1)/p)θ^{p/(p-1)}` (line 81) are the correct closed forms (verified: `p=2` gives `a*(3)=3.0, V(3)=4.5`; `p=4` gives `1.442, 3.245`; `p=6` gives `1.246, 3.114`). `f_θ=a`, so `∂_θ f(a*)=a*` (line 77). The test `resid=|Vp − a_star|` (line 78) with `Vp=np.gradient(V,θ)` (line 75, no knowledge of `a*`) is precisely eq. `eq:prelim_envelope`.
- `f` is strictly concave in `a` on `a>0` for `p≥2` (`f''=−(p-1)a^{p-2}<0`), so the maximizer is unique — matching the theorem's differentiability hypothesis. The line family supplies the non-unique/kink case, so both branches of the theorem are illustrated.
- Numerics match the theorem's guarantee. Max envelope residual `~1e-4` (8.75e-05 / 1.42e-04 / 1.22e-04) is at the grid/finite-difference floor: the a-grid spacing is `3.5/19999 = 1.750e-4`, half-spacing `8.750e-05`, which equals the reported `max_astar_err` (8.75e-05) to all printed digits. For `p=2`, `max_resid = max_astar_err = 8.75e-05` exactly, because `V` is near-exact (3.83e-09) and the residual is dominated by the argmax grid quantization — internally consistent, not a defect. Line family: identity holds on fraction `1.0000` off kinks (a straight segment's `np.gradient` returns its slope to machine precision, so `resid<1e-6` trivially). No information leakage: deterministic optimization, `Vp` uses only `V`.

### (b) PRESENTATION / NUMBERS — PASS
Every number traces to the artifact and stdout ↔ .tex ↔ figure agree.
- stdout lines 12-14 (`8.75e-05, 3.83e-09, 8.75e-05` etc.) reproduce verbatim in `envelope_theorem.tex` lines 9-11. Line-family stats (`frac 1.0000`, `kinks 1.0`, stdout line 17) reproduce in .tex line 13.
- Figure caption right panel `|dV/dθ − ∂_θ f(a*,θ)|` matches code `resid=|Vp − partial|`; axes labelled `θ` and `|dV/dθ − f_θ(a*,θ)|` on a log scale (matplotlib panel B). Panel A shows 8 gray lines + red envelope with one kink near `θ≈0.52`, matching `n_lines=8` and `n_kinks=1` for seed 0.
- Note (cosmetic): stdout lines 18-20 record cache/figure/table paths under `…/rl-theory-proofs/…`, i.e. the artifacts were generated in a git worktree rather than the primary `rl` checkout. Numbers are unaffected and match the committed files.

### (c) CHAPTER FIT — PASS
Figure + caption + theorem teach the result to a cold reader. Panel A is the canonical "value = upper envelope, tangent to the active member, kinks where the maximizer switches" picture; Panel B shows the identity holds at the numerical floor. The worked scalar example (line 376, `V=max_a(θa − a²/2)`) primes intuition before the formal statement. One minor cold-reader friction: the two panels use different `θ` ranges (line family `[0,2]`, smooth family `[0.2,3]`), not flagged in the caption.

### (d) EFFICIENCY / STANDARDS — PASS (minor nits)
- Seeds: line family runs 15 seeds (`seed_base=66000..66014`, ≥10 required); smooth family is deterministic (no seed needed). Compliant.
- Flags: uses `add_component_args`, `parse_force_set`, `compute_or_load` (multi-component convention); `main()` handles `--data-only`/`--plots-only`; `compute_data()` writes no plots and `generate_outputs()` runs no training — boundary rules respected.
- stdout: facts only, one line per configuration, no opinions. Compliant.
- Nit: the line-family stat is reported as a mean over 15 seeds with no standard error (CLAUDE.md asks for means and SEs). Here the quantity is deterministically `1.0` so `SE=0`, making the omission immaterial.

## 7-Point Checklist

1. **Algorithm identity** — PASS. Grid maximization + central-difference derivative + comparison to `∂_θ f(a*)`; term-for-term the envelope identity. No placeholder or stubbed component.
2. **Environment/MDP fidelity** — PASS. Both families (`θa − a^p/p`, `p∈{2,4,6}`; `α_a+β_a θ`, `K=8` random lines) match the theorem hypotheses (compact set, joint continuity, unique vs non-unique maximizer). Closed forms in code match the math.
3. **Data integrity** — PASS. `_run_experiment()` (lines 138-177) actually grid-maximizes and finite-differences; reported numbers equal the computed variables (verified the a-grid floor `8.750e-05` independently). No hardcoded results. Result cached under a hash of `CONFIG`.
4. **Comparison fairness** — PASS. Same grids and same `θ` for the numerical `V'` and the `∂_θ f(a*)` reference within each family; closed forms computed on the identical `θ` grid.
5. **Theoretical sanity** — PASS. Residual sits at the known grid/FD floor (`~1e-4`, matching half the a-grid spacing); identity holds to machine precision off the piecewise-linear kinks; no method "beats" the analytic reference.
6. **Information leakage** — PASS. `Vp=np.gradient(V,θ)` uses only `V`; deterministic optimization, no true model fed into a "model-free" estimate.
7. **Seed/reproducibility** — PASS. Line seeds fixed (`66000+si`), 15 runs; smooth family deterministic. SE not printed for the (zero-variance) line statistic — immaterial.

Diagram-only cap: **does not apply.** The script genuinely computes grid optima, finite-difference derivatives, and upper envelopes across 15 random seeds, not a static diagram.

## Findings (severity-ordered)

1. **(cosmetic) stdout records worktree paths.** `envelope_theorem_stdout.txt:18-20` writes cache/figure/table paths under `…/rl-theory-proofs/…` instead of the primary `rl` checkout — a provenance smell only; the committed artifacts and all numbers match.
2. **(cosmetic) Figure panel-B title is mildly conclusory and slightly imprecise.** The matplotlib title "Envelope identity holds (finite-difference floor)" both asserts a conclusion and attributes the floor to finite differencing, whereas for `p=2` the residual is dominated by argmax grid quantization (`max_resid = max_astar_err = 8.75e-05`). The .tex caption (`envelope_theorem.tex:3`, "finite-difference and grid-quantization floor") is fully precise, so nothing shipped in the paper text is wrong.
3. **(cosmetic) Panels use different `θ` ranges** (`[0,2]` vs `[0.2,3]`), unremarked in the caption. Harmless since they are separate experiments.
4. **(cosmetic) No standard error on the line-family mean.** Deterministically `1.0`, so `SE=0`; omission is immaterial.

No correctness, data-integrity, leakage, or fairness defects found.

**Bullshit score: 20%** — A hostile Reviewer 2 snarks that the panel-B title asserts "holds" and pins the floor on finite differencing when the `p=2` residual is really argmax grid quantization, but the theorem identity is computed correctly, the a-grid floor reproduces to the digit, and every number traces stdout → table → figure, so the substance is airtight.
