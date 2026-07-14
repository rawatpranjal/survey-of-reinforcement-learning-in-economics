# Audit: ch03_theory/sims/bairds_counterexample.py

**Date:** 2026-07-14
**Type:** FULL independent re-audit (fresh auditor, no prior opinion; prior 2026-05-19 bullshit-detector read only after forming verdicts)
**Script:** `/Users/pranjal/Code/rl/ch03_theory/sims/bairds_counterexample.py`
**Consuming tex:** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` — subsubsection "Why Off-Policy Learning Diverges" (prose tex:424), Theorem `thm:baird` + proof (tex:426-437), Resolutions `sec:deadly_triad_resolutions` (tex:448-457). The deadly-triad figure at tex:441-446 embeds `deadly_triad_geometry.png`, NOT this sim.
**Primary source:** `/Users/pranjal/Code/rl/ch03_theory/papers/baird1995_residual_algorithms.pdf` (md5 `01884100dc0956ca97f48b52cb36570f`, 9 pp, Baird "Residual Algorithms", 1995). Figure 1 "The star problem" and the §3 star-problem description read by rendering the PDF page at 200 dpi (the .md drops the figure).

**Files read this turn (end to end where relevant):**
- `ch03_theory/sims/bairds_counterexample.py` (full, 328 lines)
- `ch03_theory/sims/bairds_counterexample_stdout.txt` (full, 51 lines)
- `ch03_theory/sims/bairds_counterexample.png` (viewed)
- `ch03_theory/tex/planning_learning_v3.tex` (tex:400-467, plus grep of every `Baird`/`counterexample` occurrence)
- `ch03_theory/papers/baird1995_residual_algorithms.pdf` (pp.1-4 rendered; Fig.1 cropped at 200 dpi and read)
- `ch03_theory/sims/bullshit-detector_bairds_counterexample_2026-05-19.md` (prior audit, read at Step 6)
- `audits/ch03_theory__brock_mirman_newton_2026-07-14.md` (format reference)

**Output attribution.** The `__main__` block writes exactly one artifact: `bairds_counterexample.png` (py:325-326). The stdout above is captured to `bairds_counterexample_stdout.txt`. Neither artifact is `\includegraphics`'d or numerically quoted anywhere in the tex tree (grep across `docs/`, `ch03_theory/`, `thesis/`, `thesis_v2/`, `journals/` is clean). This sim is a numerical companion to `thm:baird` in the demonstrative sense only; the shipped proof is fully symbolic and cites no number this script produces.

---

## Thesis statement (what this sim is evidence FOR)

The surrounding tex argues that off-policy semi-gradient TD diverges on Baird's six-state star even though `V*≡0` is exactly representable (`thm:baird`), driven by a 5-versus-1 counting imbalance on the shared weight under uniform (non-`d^\pi`) transition weighting; and that three algorithm classes each restore stability by neutralizing one leg of the deadly triad (`sec:deadly_triad_resolutions`: target networks weaken bootstrapping, gradient-TD fixes the projection, `ℓ2` regularization shrinks the projection). The sim is evidence FOR this on the exact MDP of Baird (1995) Figure 1: the expected (population) semi-gradient update diverges, while fitted value iteration, TDC, and `ℓ2`-regularized TD each stay bounded.

---

## Primary-source verification (Baird 1995, Figure 1) — MATCH

Rendered Figure 1 at 200 dpi and read the node labels directly:
- Spokes: `V(1)=w0+2w1`, `V(2)=w0+2w2`, ..., `V(5)=w0+2w5`.
- Hub: `V(6)=2w0+w6` (coefficient 2 on the shared weight, coefficient 1 on the hub's own weight, **plus** sign).
- Topology: all five spokes have arrows into the hub; the hub carries a self-loop.
- §3 text (p.2-3): "there are six states ... value of each state is given by the linear combination of two weights ... Every transition yields a reinforcement of zero ... each possible transition is observed equally often."
- p.3 divergence text: "if all weights are initially positive, and V(6) is initially much larger than all of the other values ... w0 is increased five times for every time that it is decreased ... all of the weights go to positive infinity, except for w6, which goes to negative infinity."

Code `make_features()` (py:37-45): spokes `X[i,0]=1, X[i,i+1]=2` → `V(spoke)=w0+2wi`; hub `X[5,0]=2, X[5,6]=1` → `V(6)=2w0+w6`. `make_A()` weights each of the six transitions by `1/6` (py:59-72). `make_w0()` sets `w0..w5=1, w6=10` → `V(spoke)=3`, `V(6)=12`, all-positive with `V(6)` dominant (py:48-56). Every element of Baird's Figure 1 — six states, seven weights, the two coefficient patterns, the `+w6` sign, zero rewards, uniform transition weighting, the spokes→hub→hub topology, the all-positive `V(6)`-dominant init — reproduces exactly. The independent run confirms `w6` drifts down (10.00 → 7.88 over 1000 epochs), the correct direction toward Baird's `-∞` ("more slowly", as the paper states).

Note: this corrects the central defect of the 2026-05-19 audit (35%), which found the code then implemented the Sutton & Barto seven-state variant while the tex claimed six states. The script has since been rewritten to the genuine Baird six-state form, and the tex was aligned; the old "V(6)=2w1−w6" minus sign (never in Baird's actual figure) is gone from both.

---

## Independent numeric verification

Wrote a standalone recomputation from scratch (`/tmp/baird_verify.py`, `/usr/local/bin/python3`) of every headline number, and re-ran the sim itself. **Full-script re-run is byte-identical to the committed stdout (`diff` empty).** Independent values, all matching:

| Quantity | Committed stdout | Independent recompute | Match |
|---|---|---|---|
| `A` eigenvalues (real) | +0.0708, +0.0708, 0, −0.6667 ×4 | +0.0708, +0.0708, 0, −0.6667 ×4 | ✓ |
| spectral radius `(I+αA)` | 1.000708 (diverges) | 1.000708 | ✓ |
| fitted-VI iter-matrix `ρ` | 0.9900 (= γ) | 0.990000; closed form `γ·(x_hub^T X⁺ 1)=γ·1` | ✓ |
| L2 max real eig `(A−ηI)` | −0.9292 | −0.9292 | ✓ |
| L2 `ρ(I+α(A−ηI))` | 0.990708 | 0.990708 | ✓ |
| hand anchors V(1),V(6),δ(1),δ(6),w0(1) | 3, 12, 8.88, −0.12, 1.0736 | 3, 12, 8.88, −0.12, 1.0736 | ✓ |
| semi-grad w0 @0/100/500/1000 | 1.00/8.88/52.49/143.12 | identical | ✓ |
| semi-grad max\|V\| @0/100/500/1000 | 12.00/27.72/114.50/294.13 | identical | ✓ |
| fitted-VI max\|V\| @1000 | 0.00 | 0.0005 | ✓ |

Extra sanity checks: `A` has a one-dimensional null space (rank 6), which is why `V*≡0` is not the unique TD fixed point; TDC (5000 epochs) converges to a bounded nonzero point (`max|V|≈2.44`, `‖Aw‖≈0.009`), and `ℓ2`-reg converges to exactly `w=0`. Both confirm the code's docstrings.

---

## 7-point checklist

1. **Algorithm identity — PASS.** Semi-gradient TD is the expected update `w += α A w` with `A=(1/6)Σ_s x(s)(γx_hub−x(s))^T` (py:59-93) — the population semi-gradient TD(0) with the bootstrap target held fixed under differentiation, exactly the object the proof (tex:432-435) sums "one increment per transition over an epoch." Fitted VI freezes the target, solves the feature-space regression in closed form via `X⁺` each step (py:101-121). TDC (py:129-156) matches Sutton et al. (2009) term-by-term: `dw = δ x(s) − γ x_hub (x(s)^T h)`, `dh = (δ − x(s)^T h) x(s)`, two-timescale with `β=α·η_h`, `η_h=10`. `ℓ2` (py:164-178) is `w += α(A w − η w)`, i.e. Lim et al. (2024)'s `−ηθ` penalty. No placeholders; no penalty-that-is-always-zero.
2. **Environment/MDP fidelity — PASS.** Six-state star, seven weights, `V(spoke)=w0+2wi`, `V(hub)=2w0+w6`, zero rewards, all transitions to hub, hub self-loop, uniform `1/6` weighting. Matches Baird Figure 1 (verified visually) and the tex prose (tex:424) exactly.
3. **Data integrity — PASS.** Re-run is byte-identical to committed stdout; every table/anchor traces to a live `np.linalg` computation, none hardcoded. The "hand-derived" anchors (py:277-294) are an independent algebraic derivation printed alongside the computed value, and they agree.
4. **Comparison fairness — PASS.** All four methods share `X`, the same `make_w0()` init, the same `γ=0.99`, `α=0.01`, and 1000 epochs, all in the expected-update regime. TDC's auxiliary `h` and second step size, and `ℓ2`'s `η`, are intrinsic algorithm hyperparameters, not a rigged advantage.
5. **Theoretical sanity — PASS.** Semi-gradient diverges (`ρ=1.0007>1`); fitted VI contracts at exactly `ρ=γ=0.99`; `ℓ2` contracts at `ρ=0.9907<1` and reaches `w=0`; TDC stays bounded and lands on a biased (nonzero) TD fixed point because `A` is singular. `V*≡0` is representable, and the two converging-to-truth methods reach it. No method beats the oracle. Consistent with Baird (1995) and Tsitsiklis-Van Roy deadly-triad theory.
6. **Information leakage — PASS.** No method reads `V*=0`, the true model, or an optimal policy. Fitted VI's `X⁺` is a projection onto the (given) feature span, legitimate. No peeking.
7. **Seed/reproducibility — PASS.** Deterministic by construction (expected updates, zero sampling), so bit-reproducible; the ≥10-seed rule does not bind (nothing stochastic to average). Two runs agree exactly.

---

## Figure ↔ stdout consistency — PASS

The 2×2 PNG panels plot `max_s|V(s)|`. Panel 1 (semi-grad, red) rises 12→294; panel 2 (fitted VI, green) decays 12→0; panel 3 (TDC, orange) drops to a ~2.44 plateau; panel 4 (`ℓ2`, purple) humps to ~12.3 then decays to 0. All four traces match the committed `max|V(s)|` table (stdout:37-42), including the `ℓ2` early hump (t=0:12.00 below the transient peak, then 10.21/0.76/0.01).

---

## Prior-audit comparison

Prior: `bullshit-detector_bairds_counterexample_2026-05-19.md` scored **35%**, driven by two MED findings — (1) code was the seven-state / eight-weight Sutton-Barto variant while the tex claimed six states, and (2) a value-function sign/coefficient swap (`V(6)=2w1−w6` in tex vs no-minus in code). Both are **RESOLVED**: the current code is the genuine six-state Baird form, matching both Baird's Figure 1 and the current tex. The old prose's spurious `−w6` (absent from Baird's actual figure) is gone. The prior LOW findings — unsourced "Section 5a" anchors, and a "personal understanding, not for the paper" disclaimer — are also gone (the current verification block derives its anchors inline, py:277-294; the docstring is now a clean chapter-companion description). Nothing regressed.

---

## Findings (severity-ordered)

**Finding 1 (LOW, conditional-on-shipping) — Fix 1 panel is fitted value iteration, but the tex's first resolution is "target networks."** `sec:deadly_triad_resolutions` lists target networks / gradient-TD / regularization; the sim's three fixes are fitted-VI / TDC / `ℓ2`. Fixes 2 and 3 map one-to-one. Fix 1 substitutes fitted value iteration (exact inner regression, frozen target) for a literal target network. The code discloses this as "the theoretical limit of what target networks approximate (K→∞)" (py:104-110), which is conceptually sound — a target network with infinite inner optimization is fitted VI, and freezing the bootstrap target is precisely the "weaken bootstrapping" lever. Because the figure is not shipped, this touches nothing in the PDF. If it is ever included, the caption should say "fitted value iteration, the exact-regression (K→∞) limit of a target network" so a reviewer does not read Fix 1 as a DQN-style target network.

**Finding 2 (LOW) — "samples all transitions equally often" (prose) vs the deterministic expected update (implementation).** The tex prose (tex:424) and Baird both phrase training as stochastic uniform sampling; the sim computes the population mean-field iterate `w += α A w`. The code is explicit about this ("expected (population) updates, the limit of sampling every transition equally often", docstring lines 5-7), and the shipped proof itself is a mean-drift argument summing one increment per transition per epoch (tex:432-435), so tex and code describe the same object. No sampling figure is claimed. Nuance only.

**Finding 3 (INFO) — sim is unshipped.** No `\includegraphics` and no quoted number anywhere in the tex tree. This is the dominant risk-reducer: the residual nuances above cannot mislead a reader of the paper. It also means that if the sim is later promoted into the chapter, Findings 1-2 should be closed first (caption wording + an "expected-update" note).

No finding reaches MED in the current (post-rewrite, unshipped) state.

---

**Bullshit score: 12%** — Reviewer 2 could quibble that the Fix-1 panel labels fitted value iteration where the resolutions prose says "target networks," but the MDP reproduces Baird's Figure 1 exactly, every stdout number recomputes independently and the re-run is byte-identical, the algorithms are textbook-faithful, and nothing here is shipped or cited in the paper.
