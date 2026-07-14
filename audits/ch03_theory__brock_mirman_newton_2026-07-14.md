# Audit: ch03_theory/sims/brock_mirman_newton.py

**Date:** 2026-07-14
**Type:** FULL calibration re-audit (treated as never-audited; prior audits read only at Step 6)
**Script:** `/Users/pranjal/Code/rl/ch03_theory/sims/brock_mirman_newton.py`
**Consuming tex:** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex`, subsection "Simulation Study: The Brock--Mirman Economy" (lines 105-134), inside the Newton-method section (lines 12, 61-121).

**Files read this turn (end to end where relevant):**
- `ch03_theory/sims/brock_mirman_newton.py` (full)
- `ch03_theory/sims/brock_mirman_newton_stdout.txt` (full)
- `ch03_theory/sims/brock_mirman_results.tex` (full)
- `ch03_theory/sims/brock_mirman_convergence.png` (viewed)
- `ch03_theory/sims/brock_mirman_lp_dual.png` (viewed)
- `ch03_theory/tex/planning_learning_v3.tex` (lines 1-135, plus grep of all cross-refs)
- `appA_preliminaries/tex/preliminaries.tex` (label check only)
- `docs/refs.bib` (citation-key checks)
- `audits/ch03_theory__brock_mirman_newton_2026-05-19.md`, `..._polish_2026-05-20.md`
- git history of the script and the tex file (restructure diff)

**Output attribution (which artifacts THIS script writes).** `generate_outputs` calls only `plot_convergence` and `generate_table` (py:682-689). So this script writes exactly: `brock_mirman_convergence.png` + `.pdf` (py:627-630) and `brock_mirman_results.tex` (py:676-679). It does **not** write `brock_mirman_lp_dual.png` (no plotting code for occupation measures exists in the current script; file mtime 2026-03-14 pre-dates the script's 2026-05-19 mtime) and it does **not** write `newton_framework_results.tex` (that file, mtime 2026-01-29, is produced by `ch03_theory/sims/lqr_convergence.py`). Both are stale/foreign to this sim.

---

## Step 3 — Thesis statement (what this part of the chapter argues, what the sim is evidence FOR)

(i) The theoretical claim of the surrounding section is that **policy iteration is Newton's method applied to the Bellman equation** (tex:12, subsection at tex:61). Concretely: writing the Bellman residual `G(V)=V-TV`, on each affine piece `G'(V_k)=I-γP^{π_k}`, and the Newton step `V_{k+1}=V_k-[G'(V_k)]^{-1}G(V_k)=(I-γP^{π_k})^{-1}r^{π_k}` is exactly Howard policy evaluation (tex:110-115, eq:pi_newton). The consequence: PI inherits Newton-like finite/superlinear convergence (iteration count set by the number of policy switches, not the state dimension), whereas value iteration is a `γ`-contraction limited to linear rate `β^k`. A third method, the Manne (1960) LP, recovers the same `V*`.

(ii) The sim is used as evidence FOR that dichotomy on a real 1,000-state discretized economy: VI converges in 567 iterations at rate `0.96^k`; PI converges in 11 (a ~50x iteration reduction); PI iteration counts are near-constant (7-11) across grid sizes 10-500 (Santos-Rust grid-independence); and the LP value matches VI to solver precision. The scalar 3-policy figure panels (a)/(b) illustrate the geometry (staircase vs Newton-jump); panel (c) is the actual convergence data.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS
- **VI** (py:136-152): standard `Q=R+γ(P@V)`, `V=max_a Q`, sup-norm stopping. Correct.
- **Howard PI** (py:155-193): exact policy evaluation via `np.linalg.solve(I-γP^π, r^π)` (py:170-171) then greedy improvement, terminating on exact policy equality (py:184). This *is* the Newton iteration the tex names (eq:pi_newton). Correct.
- **Manne LP** (py:196-270): primal `min Σα(s)V(s)` s.t. `V(s)≥r+γPV` over feasible `(s,a)` (py:209-223); dual flow-conservation `Σ_a μ(s,a)-γΣP μ=α(s)`, `μ≥0` (py:244-259). Sign handling (`c_dual=-R` for min) correct. Independent sanity check passes: dual occupation mass `Σμ=25.0000` (stdout:20) equals `1/(1-γ)=1/0.04=25` exactly.
- **Closed form** `k'=αβzk^α` (py:119-129): correct Brock-Mirman log-utility/Cobb-Douglas/full-depreciation savings rule; the constant savings rate `αβ` holds even under the Markov `z`, so it is a valid exact benchmark.
- **Theory consistency:** VI 567 iters linear at `β^k` (panel c tracks the `β^k` line); PI finite-terminates in 11; timing sweep shows VI iters constant at 567 across all grids (VI rate depends on `β`, not `n_k` — theoretically expected) and PI iters 7,7,9,9,10 rising mildly (stdout:23-27), matching Santos-Rust grid-independence within a small factor. LP-VI agreement `2.32e-9` (stdout:18). No method beats the oracle; no contradictions.

### (b) PRESENTATION / NUMBERS — PASS (checked against the CURRENT post-restructure tex)
Every number in the current prose/caption traces to a generated artifact:
- "500-point grid ... (1,000 states)", `α=0.36`, `β=0.96`, `z∈{0.9,1.1}`, persistence 0.8 (tex:108) → py:30-34, stdout:2,5.
- "11-iteration convergence" (tex:115), "PI converges in 11" (tex:121,130) → table row (results.tex:10), stdout:12.
- "567 iterations" (tex:117,121,130) → table (results.tex:9), stdout:9.
- "7-11 iterations across all grid sizes" (tex:115 footnote) → table shows 7 (n_k=10), 10 (n_k=200), 11 (regime 1); stdout:23-27.
- "50x reduction" (tex:121): 567/11 = 51.5 → rounds to 50x. Footnote "n_k=200 PI roughly 50x faster": 2.467/0.051 = 48x (results.tex:15-16). Both round to 50x, consistent.
- "‖V_LP − V_VI‖∞ < 10^{-8}" (tex:121) → stdout:18 and results.tex:12 report 2.3e-9.
- Figure (c) legend "VI (567 iters)", "PI (11 iters)" and the `β^k` bound line match the PNG I viewed.

**Restructure check (the load-bearing part of this re-audit):** the 2026-07-13 restructure (commits 48bff41, 45fad87) touched this subsection at exactly one point — line 115 gained the clause "the Neumann-series resolvent of Theorem~\ref{thm:prelim_neumann}". No number changed. The new cross-ref resolves: `\ref{thm:prelim_neumann}` (planning_learning_v3.tex) matches `\label{thm:prelim_neumann}` (appA_preliminaries/tex/preliminaries.tex), both verified verbatim. Citations `santos2004`, `manne1960`, `puterman1979`, `brockmirman1972`, `bertsekas2022newton` all present in refs.bib.

### (c) CHAPTER FIT — PASS
The sim demonstrates precisely the Step-3 claim. VI 567 (linear `β^k`) vs PI 11 (Newton/finite) is the core VI-vs-PI-as-Newton contrast; the grid sweep supports "iteration count depends on policy switches, not dimension"; the LP row instantiates the third (Manne) solution method the section discusses. Panels (a)/(b) are explicitly labelled as a scalar illustration (tex:119), not passed off as data.

### (d) EFFICIENCY / STANDARDS — PASS (minor nits)
- Per-component `compute_or_load` caching with sane config decomposition; `--data-only`/`--plots-only`/`--algo r1|r2|r3` flags via `add_component_args` + regime shorthand (py:69-73, 497-521). `--plots-only` correctly recomputes from cache only.
- Seeds: `SEED=42`, `np.random.seed` per component (py:280, 362, 415). Sim is fully deterministic (VI/PI/LP have no sampling), so seeds are cosmetic — legitimately so; iteration counts are bit-reproducible. Timing reported as median of 5 runs (deterministic iteration counts, only wall-clock varies) — the standards' "≥10 seeds" rule does not bind a deterministic planner.
- stdout is factual, tabular, no opinion words.
- Nits: `plot_convergence` hardcodes `figsize=(13,4)` (py:569) though `FIG_*` constants are imported; occupation-measure array `mu` is computed but only its summary is printed (its figure was removed) — see Finding 1.

---

## 7-point checklist

1. **Algorithm identity** — PASS. VI, Howard PI, Manne LP all textbook-faithful; PI eval is the exact Newton step (py:170-171). No placeholders.
2. **Environment/MDP fidelity** — PASS. Log utility, `zk^α` technology, resource constraint `c+k'=zk^α`, `z∈{0.9,1.1}` persistence-0.8 chain all match tex:108 (py:32-34, 107-114).
3. **Data integrity** — PASS. Table/stdout numbers come from live `compute_*` functions through `compute_or_load`; no hardcoded results. (Cache pkls are gitignored and absent on disk now, but stdout+results.tex are internally consistent and were regenerated together on 2026-05-19/20.)
4. **Comparison fairness** — PASS. Regime 1 VI vs PI: same R,P,γ,tol, both effectively start from V=0. Regime 3: MDP rebuilt per grid, both algos run on it. Fair.
5. **Theoretical sanity** — PASS. VI tracks `β^k`; PI finite-terminates; `Σμ=1/(1-γ)=25` exact; LP=VI to 2.3e-9; PI iters ~grid-independent. All align with Banach/Howard/Manne/Santos-Rust.
6. **Information leakage** — PASS. Model-based planning; PI/LP legitimately use true P,R. Closed form is a benchmark only, never consulted by the solvers (py:311,342 are comparison-only). No leak.
7. **Seed/reproducibility** — PASS (deterministic). Substantive claims (567, 11, 7-11) bit-reproducible; wall-clock reported as median-of-5 with min/max disclosed.

---

## Prior-audit comparison

Prior audits: `..._2026-05-19.md` (10%) and `..._polish_2026-05-20.md` (5%).

Open items from 2026-05-19, and their status today:
- **PI "final error" column tautological (0.0e+00 vs its own final iterate)** — RESOLVED. Polish pass changed the PI cell to `---` (results.tex:10) and the caption discloses why (results.tex:3 region).
- **Closed-form policy-match % absent from stdout artifact** — RESOLVED as designed. Gated behind `--verbose` (py:316-317, 346-347); value cached as `vi_cf_match`/`pi_cf_match`. The tex makes no numeric match claim, so nothing is unsupported.
- **Single-sample wall-clock** — RESOLVED. `TIMING_REPS=5`, medians reported, protocol disclosed in caption and stdout banner (stdout:3), `t_*_runs` pickled.

Open item from 2026-05-20 (5%): VI cell reports the successive-difference stopping error `9.7e-11` rather than true `‖V−V*‖∞`. STILL PRESENT (results.tex:9) and STILL acceptable — it is the VI termination criterion, and PI/LP cross-checks pin the true value to 1e-9 separately.

Nothing REGRESSED. The restructure that motivated this re-audit left every sim number intact (only a cross-ref was added, and it resolves).

**Did my fresh pass find anything ≥25% the prior audits missed?** No. The two genuinely new observations below (stale orphan PNG; stdout op-count of 1.0x) are real but both sit well under 25%: neither appears in the shipped PDF, neither is cited by any prose number, and neither touches a correctness claim. The prior 5-10% band stands, and this calibration confirms no skip-tier-invalidating miss.

---

## Findings (severity-ordered)

**Finding 1 (~10%, repo hygiene, does not ship) — stale orphan figure `brock_mirman_lp_dual.png`.** The file (mtime 2026-03-14, 189 KB; an LP-primal-vs-VI value plot + occupation-measure bars) is generated by no code in the current script (`generate_outputs` writes only the convergence figure and the table, py:682-689) and is referenced by no `\includegraphics` anywhere in the tex (grep clean). It is a leftover from a prior script version whose LP-dual plotting function was removed; the current `compute_lp_r2` still computes `mu` but only prints its summary. Harmless to the paper (never included), but a candidate for deletion so a future edit cannot resurrect a stale figure. Prior audits did not flag it.

**Finding 2 (~10%, stdout-only) — misleading "Ratio (VI/PI): 1.0x" operation count.** stdout:28-32 reports VI total ops = PI total ops = 453,600 and a 1.0x ratio, using a dense `O(n^3)` model for PI's linear solve at n_k=20 that coincidentally equals VI's `iters × n_s·n_a`. Read in isolation this superficially contradicts the paper's "50x reduction" headline. It is never cited in the tex (the paper's 50x is iteration-count and wall-clock, both well-supported), and the actual wall-clock shows PI ~48-50x faster, so the FLOP proxy merely understates PI's real advantage. Cosmetic; consider dropping the op-count block or annotating that it ignores sparse-solve efficiency.

**Finding 3 (<10%, illustrative) — VI-count closed form presented as an exact equality.** tex:117 writes `k=⌈log(ε/‖TV_0−V_0‖∞)/log β⌉ = 567`; the RHS depends on `‖TV_0−V_0‖∞` (order 1-2, not pinned), so the "= 567" is an idealization that happens to land on the real run count. The 567 itself is the genuine measured value (stdout:9), so no number is wrong; the derivation is illustrative rather than exact.

No finding reaches the 25% (Reviewer-2-writes-a-snark) threshold in the shipped artifact. The substance — PI-as-Newton, VI 567 vs PI 11, LP=VI to 1e-9, grid-independent PI counts — is exactly what the tex claims and is consistent with Banach contraction, Howard's theorem, Manne (1960), and Santos-Rust (2004).

**Bullshit score: 12%** — Reviewer 2 grumbles about a stale orphaned `lp_dual.png` in the sims folder and a confusing 1.0x op-count line in the stdout, but neither is in the paper, every shipped number traces to the artifacts, the algorithms are textbook-faithful, and the 2026-07-13 restructure left the sim's numbers and its one new cross-reference intact.
