# Audit (DELTA) — `ch09_rlhf/sims/axiom_aware_aggregation.py`

**Date:** 2026-07-14
**Type:** DELTA (prior audit 2026-05-22)
**Sim:** Bradley-Terry MLE vs Leximax Copeland subject to PO (LCPO) on the 6-candidate construction of Ge et al. (NeurIPS 2024), Theorems 3.1 and 4.3.
**Anchor:** `ge2024axioms` — Ge, Halpern, Micha, Procaccia, Shapira, Vorobeychik, Wu, "Axioms for AI Alignment from Human Feedback," NeurIPS 2024.
**Consuming tex:** `ch09_rlhf/tex/rlhf.tex` §5.7, subsubsections "Leximax Copeland subject to Pareto optimality" (`sec:lcpo`) and "Simulation Study: Axiom-Aware Aggregation on Heterogeneous Voters" (`sec:sim_axiom_aware`, lines 125-144).

**Files read this turn (full or cited spans):**
- `ch09_rlhf/sims/axiom_aware_aggregation.py` (end to end)
- `ch09_rlhf/sims/axiom_aware_aggregation_stdout.txt`
- `ch09_rlhf/sims/axiom_aware_aggregation.tex`
- `ch09_rlhf/sims/axiom_aware_aggregation.png` (viewed)
- `ch09_rlhf/tex/rlhf.tex` (lines 84-145, plus grep sweep)
- `ch09_rlhf/papers/ge2024_axioms_ai_alignment_feedback.md` (lines 150-205 + grep)
- `audits/ch09_rlhf__axiom_aware_aggregation_2026-05-22.md`

## Delta summary

- **Sim code, stdout, .tex, .png:** all committed once, in `f65b2a2` (2026-05-22 19:30). No commit touches the `.py` after the prior audit. Working tree clean. The sim is byte-identical to what was audited.
- **The change since the audit is entirely in the consuming tex.** Commit `45fad87` (2026-07-13) "add Nash-Q and Bradley-Terry proofs" inserted a full inline proof of Theorem 3.1 into `rlhf.tex` (lines 103-109). Verified this proof was ABSENT at `f65b2a2` (`git show f65b2a2:...rlhf.tex` has zero `\begin{proof}` in the section and no "δ∈(0,1)" line; `git log -S 'delta \in (0,1)'` attributes the line to `45fad87`).
- **Consequence:** the newly added proof states "Choose δ∈(0,1)" (line 106) while the sim uses δ=2 (line 128, `ENV_PARAMS['delta']=2.0`). At the prior audit the δ discrepancy was sim-vs-external-paper; the delta moved it onto the page as sim-vs-in-document-proof.
- **Data-provenance flag:** committed stdout shows "Cache hit" on all three components; the `.png`/`.tex`/stdout mtimes are 02:25:03 while the `.py` mtime is 02:53:06 (28 min later, same pre-commit session). Outputs are cache-derived and were not the last thing regenerated. All oracle invariants in the committed stdout match the committed code, so there is no evidence of stale numbers (see criterion (b)).

## Step 3 — what this sim is evidence for

(i) **Theoretical claim being advanced.** RLHF's standard Bradley-Terry reward model, once labelers are heterogeneous, is a hidden social-welfare aggregator, and as a loss-based linear rank-aggregation rule it provably fails two elementary social-choice axioms: Pareto optimality (PO) and pairwise-majority consistency (PMC) (Theorem 3.1). An axiom-aware alternative, Leximax Copeland subject to PO (LCPO), repairs both (Theorem 4.3).

(ii) **What the sim is used FOR.** It is the empirical illustration that (a) BT-MLE's PO and PMC violation rates rise to 1 as the number of comparisons N grows, "exactly the asymptotic behavior predicted by Theorem 3.1" (line 130), and (b) LCPO holds PO at every N and reaches zero PMC violation by N=100, while (c) the worst-group utility of the winner converges to -2 for both, so the LCPO advantage is axiomatic consistency, not welfare efficiency in the winner.

## Criteria

### (a) CORRECTNESS — QUALIFIED PASS on the sim; one on-page proof↔sim inconsistency

The sim is substantively faithful and its numbers are genuine.

- **BT-MLE** (`bt_mle_linear`, lines 154-185): logistic MLE over `r_θ(c)=<θ,x_c>` via L-BFGS-B, standard `logaddexp` NLL, mild L2. Matches paper §3.1. Correct.
- **LCPO** (`leximax_copeland_po`, lines 217-289): Copeland score = count of pairwise-majority wins, leximax tiebreak on sorted margin vectors, sequential position-filling gated by an LP feasibility test (`feasibility_lp`, `linprog`/highs), with Pareto-dominance pairs added as hard constraints. Matches paper §4 (md lines 235-239, 265). Correct.
- **Mechanism verified by hand.** With ε=0.01, δ=2, feature map gives c'=(-0.01,0.02), c=(0,0). Voter (1,1): r(c')=+0.01>0; voter (-1,0): r(c')=+0.01>0 — both prefer c', so (c',c) is the sole Pareto pair. Matches stdout line 9 `[('cp','c')]`. Type-1 ranking a≻a'≻b≻b'≻c'≻c matches stdout line 8. At the BT-MLE optimum the fitted θ has large θ1, small θ2, so r(c')-r(c)=-0.01·θ1+0.02·θ2<0 → BT ranks c above c', a real PO violation on a unanimous pair. This is genuinely Theorem 3.1's phenomenon.
- **PROOF↔SIM δ CONTRADICTION (delta finding).** The inline proof requires "Choose δ∈(0,1)" (line 106). The sim uses δ=2 (line 128 states "δ=2"; `ENV_PARAMS['delta']=2.0`). These are directly contradictory and now sit in the same subsection with no reconciliation. Independently confirmed which regime the sim needs: with the literal position x_{c'}=(-ε,δε) and voter (1,1), the p-fraction voter ranks c'≻c only if δ>1 (δ=0.5 → r(c')=-0.005<0, no Pareto pair; δ=2 → r(c')=+0.01>0, Pareto pair holds). So δ=2 is the correct choice for the sim's concrete construction, and it is the proof's "δ∈(0,1)" that is inconsistent with the sim (and internally: the proof asserts "every voter ranks c'≻c" for a perturbation and δ-range that, for the (1,1)-induced voter, force c≻c'). Root cause traced to the paper: md line 175 says "0<δ<1," while footnote 7 (md line 197) claims the (1,1) voter's reward on c' is (1-δ)ε, whereas <(1,1),(-ε,δε)>=(δ-1)ε — opposite sign. The tex proof reproduced the paper's δ∈(0,1) without reconciling it against the δ=2 the sim requires. Net: the sim numbers are correct; the adjacent proof prints a contradictory δ.
- **LCPO PO "by construction."** `leximax_copeland_po` receives `dominance` from `shared['dominance']` = `pareto_dominance(p)`, the ORACLE (true) dominance set, not a sampled estimate. This is why CP_PO=0.000 at every N including N=5. The paper (md line 273) says dominance may be "approximate[d] ... through sampling"; the sim idealizes it as exact. Given the (c',c) reward gap is only 0.01 (per-label prob sigmoid(50·0.01)=0.62), a sampled-dominance LCPO would plausibly show occasional small-N PO violations. This is disclosed in the figure caption "(PO by construction)" (line 135). Faithful to the LCPO spec (LCPO is defined to enforce PO), and disclosed; see prior-audit disposition.

### (b) PRESENTATION / NUMBERS — PASS

Complete number-consistency sweep across stdout, `.tex`, figure, and prose:

- **stdout ↔ .tex.** Table is built at N=Ns[-1]=2000. stdout line 19 (N=2000): BT_PO 1.000, BT_PMC 1.000, CP_PO 0.000, CP_PMC 0.000, BT_wu -2.000, CP_wu -2.000. `.tex` rows: BT-MLE 1.00 (0.00) / 1.00 (0.00) / -2.000 (0.000) / top a; LCPO 0.00 (0.00) / 0.00 (0.00) / -2.000 (0.000) / top a. All match.
- **stdout ↔ figure.** Left panel: BT_PO orange-solid 0.60→0.53→0.40→0.60→0.77→0.93→1.00; BT_PMC orange-dashed 0.83→0.70→0.50→0.63→0.77→0.93→1.00; LCPO_PO blue-solid flat 0; LCPO_PMC blue-dashed 0.50→0.47→0.37→0.13→0→0→0. Right panel: BT_wu -1.53→-2.0, LCPO_wu -1.33→-2.0, both flattening at -2 by N=100. All trace to the stdout table. Legend says "over 30 seeds"; error bars present.
- **Oracle lines ↔ code.** stdout header ε=0.01, δ=2.0, p=0.6, β=50.0 match `ENV_PARAMS`. PMC ranking `['a','ap','b','bp','cp','c']` and Pareto pair `[('cp','c')]` reproduced by hand from the committed feature matrix and voter types.
- **prose ↔ artifacts.** Line 130 "BT-MLE ... violates both PO and PMC on every seed by N=2000; LCPO satisfies PO on every seed at every sample size and satisfies PMC ... by N=100; worst-group utility ... converges to -2 for both" — all match. Caveat: the main-text sentence presents LCPO's perfect PO as an empirical finding; the honest qualifier "(PO by construction)" appears only in the caption (line 135). Minor framing tension, not a number error.
- No fabricated/hand-typed numbers detected; every published figure traces to the stdout table.

### (c) CHAPTER FIT — PASS

The sim demonstrates the step-3 thesis. BT-MLE's PO/PMC violation → 1 as N grows is a clean, non-cheated empirical instantiation of Theorem 3.1 (BT-MLE consumes only sampled labels). LCPO's PMC → 0 by N=100 is data-driven and supports Theorem 4.3; its PO=0 is enforced-and-disclosed. The worst-group-utility panel honestly shows no winner-level welfare gap, and the prose correctly frames the LCPO benefit as axiomatic rather than efficiency. The one blemish is that the section's own proof (δ∈(0,1)) and its own experiment (δ=2) disagree on the load-bearing parameter, which a careful reader routes straight through.

### (d) EFFICIENCY / STANDARDS — PASS

`compute_or_load` per component (`shared`, `BT_MLE`, `Leximax_Copeland`) with per-config MD5 keys; `add_component_args`/`parse_force_set`; `--data-only`/`--plots-only` honored (lines 558-570). Palette from `sims.plot_style` (`COLORS`, `FIG_DOUBLE`), no hardcoded hex. N_SEEDS=30 (≥10); means and standard errors computed (`std(ddof=1)/sqrt(n)`) and reported in table and figure. `_stdout.txt` present with param header, sweep table, output paths, no opinion words. Meets Simulation Standards.

## 7-point checklist

1. **Algorithm identity — PASS.** BT-MLE = logistic MLE over linear reward (matches §3.1); LCPO = Copeland + leximax + sequential LP-feasibility + Pareto constraints (matches §4). No placeholders.
2. **Environment fidelity — PASS with note.** 6-candidate construction matches paper lines 177-179 (feature positions, voter params (1,1) and (-1,0), p=0.6). Note: δ=2 vs the paper/tex-proof δ∈(0,1); see finding F1. The sim's δ=2 is the value consistent with the literal feature map producing the (c',c) Pareto pair.
3. **Data integrity — PASS with provenance flag.** `compute_data()` runs the real estimators on freshly sampled data per seed; no hardcoded outcomes; stdout reads from `data` only. Flag: committed outputs are "Cache hit"-derived and mtime-predate the final `.py` write by 28 min. All oracle invariants and cross-artifact numbers are internally consistent with the committed code, so no stale-number evidence; a cold re-run (`--data-only` after clearing cache) would be the only way to certify the sweep numbers byte-for-byte, and that is out of scope for a read-only audit.
4. **Comparison fairness — PASS.** Identical seeded samples, sweep grid, and 30 seeds for both methods; criteria fixed in advance. Asymmetry that LCPO receives oracle dominance is the algorithm's own PO-enforcement spec, disclosed in the caption.
5. **Theoretical sanity — PASS.** N=2000: BT PO=PMC=1.00±0.00 (Thm 3.1 asymptote); LCPO PO=0 all N, PMC=0 by N=100 (Thm 4.3). Neither method beats the oracle; both winners give worst-group utility -2, as expected since both top-rank a.
6. **Information leakage — PASS (with the LCPO-oracle caveat, disclosed).** BT-MLE sees only sampled tuples. LCPO additionally sees the oracle Pareto-dominance set, per paper spec and caption disclosure; no method sees θ_i, per-comparison types, or p.
7. **Seed / reproducibility — PASS.** `np.random.default_rng(seed)`, 30 seeds, mean+SE reported, config-hashed caches.

## Prior-audit open-item disposition

- **δ=2 vs paper's δ∈(0,1) ("delta parameter override").** REGRESSED in prominence (not in the code). At 2026-05-22 the discrepancy was sim-vs-external-paper and the audit scored it 15%. The 2026-07-13 tex proof now prints "Choose δ∈(0,1)" three paragraphs above the sim's "δ=2," so the contradiction is on the page of the survey itself and is more referee-catchable. My independent computation confirms the prior audit's diagnosis: δ=2 is correct for the sim (needed for the (c',c) Pareto pair with voters (1,1),(-1,0)), and the paper's footnote 7 reward `(1-δ)ε` is sign-inconsistent with position `x_{c'}=(-ε,δε)` (direct value `(δ-1)ε`). The sim is right; the proof text is the inconsistent artifact. STILL OPEN and now more visible; recommend reconciling the proof's δ range with the sim (or annotating that the sim's concrete voters require δ>1 to realize the Pareto unanimity the theorem needs).
- **LCPO consuming oracle Pareto-dominance.** RESOLVED at the disclosure level. The prior audit noted it is the paper's algorithm spec, not external info. The figure caption now reads "LCPO violation rates remain at zero (PO by construction)" (line 135), so the enforced-not-earned nature of CP_PO=0 is disclosed. Residual: the main-text sentence (line 130) still reads as an empirical result; consider echoing "(PO by construction)" there. Not a substance defect.

## Findings (severity-ordered)

**F1 (medium) — On-page proof↔sim contradiction on δ.** `rlhf.tex:106` (proof, added 2026-07-13) requires δ∈(0,1); `rlhf.tex:128` and `axiom_aware_aggregation.py:48` use δ=2. Same subsection, unreconciled. The sim's δ=2 is correct (verified: it is what makes (c',c) Pareto-unanimous under voters (1,1),(-1,0)); the proof's δ∈(0,1), inherited from a paper extraction whose footnote-7 reward sign disagrees with its feature position, is the inconsistent statement. A referee reading the proof then the sim sees contradictory δ. Fix: reconcile in the tex (state that with the concrete voter parameters the perturbation requires δ>1, or adjust the perturbation sign so δ∈(0,1) holds).

**F2 (low) — LCPO PO=0 is enforced by oracle dominance, framed empirically in main text.** `axiom_aware_aggregation.py:359,400` feed the true `pareto_dominance(p)` set as hard constraints, so CP_PO=0 at every N including N=5. Disclosed in the caption ("PO by construction," line 135) but the results sentence (line 130) presents "LCPO satisfies PO on every seed at every sample size" as a finding. Given the (c',c) label signal is weak (prob 0.62), a sampling-based dominance estimate would not be perfect at small N. Fix: mirror the "(PO by construction)" qualifier into the main text.

**F3 (low) — Data provenance: published outputs are cache-derived and predate the final code edit.** stdout lines 1-3 show "Cache hit" on all components; `.png`/`.tex`/stdout mtimes (02:25:03) precede the `.py` mtime (02:53:06). No stale-number evidence found (all oracle invariants and cross-artifact numbers match the committed code), but the sweep numbers were not regenerated as the last step. Fix: run once with the cache cleared and recommit stdout/figure/table so the artifacts demonstrably come from the committed code.

**Bullshit score: 25%** — Reviewer 2 catches that the section's own freshly added proof requires δ∈(0,1) while its own simulation runs δ=2, and that LCPO's perfect PO is enforced by an oracle rather than earned from data; both are specific, snarky-comment-worthy inconsistencies, but the implemented methods are the methods as named, the numbers are internally consistent and reproduce Theorems 3.1 and 4.3, and δ=2 is in fact the correct choice for the construction, so the substance survives revision.
