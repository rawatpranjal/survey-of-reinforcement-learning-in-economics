# Audit: ch12_world_models/sims/cobweb_paradigms.py (DELTA)

**Date:** 2026-07-14
**Type:** DELTA (prior audits 2026-05-19 at 25%, 2026-05-20 polish at 12%; code changed 2026-07-13)
**Auditor stance:** hostile journal referee. Read-only. No scripts re-run; numbers read from committed stdout/tex/PNG artifacts and checked for mutual consistency and against hand-derivations.

## Delta summary — what changed since the last audit

One commit touches the sim after the 2026-05-20 polish:

- `d532741` (2026-07-13 22:45): "add MB-LG-Pathwise paradigm to cobweb sim; rewrite intro to two-instance framing." Adds an eighth paradigm `MBPathwisePolicy` (display name `MB-LG-Pathwise`), a model-based learner that plans by analytic (pathwise) policy gradients via a forward-sensitivity recursion, sharing the REINFORCE learner's ensemble model-fit and linear policy class. Regenerates the three PNGs and both `.tex` tables (regret table +1 row, recovery table +3 rows). Rewrites `tex/s01_intro.tex`.

**Critical delta fact:** the consuming subsection `tex/s09_dual_sim.tex` was NOT modified in this commit (empty diff `e79f828..d532741 -- s09_dual_sim.tex`). The 2026-05-20 polish had propagated the previous rename across eight specific lines of s09; the 2026-07-13 delta did no such propagation. The result is that every output artifact now shows eight paradigms while the prose still describes seven.

**Files read end to end:** `sims/cobweb_paradigms.py`, `sims/cobweb_env.py`, `sims/cobweb_paradigms_stdout.txt`, `sims/cobweb_paradigms_results.tex`, `sims/cobweb_paradigms_final_recovery.tex`, `tex/s09_dual_sim.tex`, `tex/s01_intro.tex`, `sims/tests/test_mbpo_real.py`, the three PNGs (regret, param recovery, policy distance), and the three prior audits. Delta diff via `git show d532741`.

---

## Step 3 — What the section claims and what the sim is evidence for

(i) **Theoretical claim.** The subsection (`s09_dual_sim.tex` §Cobweb, `section:fc_dual_sim_cobweb`) advances an "inductive-bias frontier" thesis: on a low-dimensional, smooth, stably-parametrized single-agent decision problem, learners that bake in more correct structure (functional form of demand/cost, closed-form planning) pay lower cumulative regret and reach smaller asymptotic policy error than gradient-free or model-free learners; and regret (integrated learning cost) and asymptotic policy quality can order methods differently (RLS wins regret via correct form + known cost; model-based LQ wins asymptotic policy distance by estimating curvature). The delta additionally sharpens a within-family point: two model-based learners with the same model-fit but different planners (closed-form LQ vs a gradient-based policy update) separate, and among gradient-based updates a low-variance analytic (pathwise) gradient should beat a high-variance score-function (REINFORCE) gradient, sharpest where the return surface is flat.

(ii) **What the sim is used as evidence for.** The cobweb panel is the quantitative backbone: Table `fc_cobweb_results` (cumulative regret at T=500 per paradigm/regime), Figure `fc_cobweb_curves` (regret trajectories), Figure `fc_cobweb_recovery` + Table `fc_cobweb_recovery` (parameter recovery), and Figure `fc_cobweb_policy_distance` (asymptotic policy error). These are evidence that the paradigms sort onto the frontier in the claimed order and that regret vs policy-distance orderings diverge as claimed.

The MB-LG-Pathwise addition is scientifically the most interesting piece of new evidence: pathwise gradients drive the linear policy to near-RLS accuracy (stable policy distance 0.053 vs REINFORCE 1.462, stable regret 47 vs REINFORCE 657), which is exactly the textbook low-variance-analytic-gradient story. Yet the prose never uses it.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS

- **Oracle Riccati.** Re-derived the LQ fixed point by hand from `r_t = a q - A_c q^2 + phi q q_prev - 0.5 phi q_prev^2` with `A_c = b + c/2 + phi/2`. The update rules `P' = phi^2/(2D) - phi/2`, `R' = (a+gamma R)phi/D`, `S' = (a+gamma R)^2/(2D) + gamma S`, `K0 = (a+gamma R)/D`, `Kq = phi/D`, `D = 2(A_c - gamma P)` (`cobweb_env.py:63-87`) all match my derivation term for term. Independently cross-checked by the file's own 401-point grid Bellman smoke test (`cobweb_env.py:100-133`).
- **New paradigm MB-LG-Pathwise (`cobweb_paradigms.py:681-879`).** The forward-sensitivity recursion is a correct analytic gradient. Verified by hand: expected model reward `r_t = a_hat a_t - b_hat a_t^2 - 0.5 c_hat a_t^2 - 0.5 phi_hat(a_t-q_t)^2` (line 804-807); `dr/da_t = a_hat - 2 b_hat a_t - c_hat a_t - phi_hat(a_t-q_t)` (line 811); `dr/dq_t = phi_hat(a_t-q_t)` (line 813); sensitivities `da/dK0 = 1 + Kq dq/dK0`, `da/dKq = q_t + Kq dq/dKq`, `dq_{t+1}/dtheta = da_t/dtheta` (lines 798-821); total derivative `dJ/dtheta = sum_t gamma^t (dr/da * da/dtheta + dr/dq * dq/dtheta)` (lines 815-816). All correct. Gradient ascent with L2-clip at 1.0 (lines 843-848). Legitimately distinct from the REINFORCE score-function estimator; the "low variance ⇒ larger step (lr 0.05, 20 rollouts)" rationale is sound.
- **REINFORCE (MBPOPolicy).** Score `(a_unclipped - mean)/sigma^2` and baseline-subtracted advantage update verified correct (lines 611-636). Minor known approximation: score uses unclipped action while reward uses clipped action; standard, non-fatal.
- **Regret sign / common random numbers.** Env is seeded `seed=s` for both oracle and every paradigm, and eps is drawn independently of the action, so all paradigms face an identical noise sequence per seed (`rollout`, lines 905-910; `compute_shared` 1006-1009 vs `compute_paradigm` 1039). Paired regret `oracle - paradigm` is clean; no paradigm beats the oracle in the mean (`stdout` 25-32). Theory-consistent.
- **Placeholder line flagged in the task.** `cobweb_env.py:116` (`X = a + gamma * np.interp(qg, qg, V_grid) * 0  # placeholder`) is dead code inside the `__main__` smoke test only: the `* 0` zeroes it and `X` is never read (the next line recomputes `r_vec` correctly). Not on any sim path. Cosmetic, not a correctness defect.

### (b) PRESENTATION / NUMBERS — FAIL (this is the headline)

Every number in stdout, both `.tex` tables, and the three PNGs is mutually consistent, and the regret table is rank-ordered by mean-across-regimes (verified: RLS 5.38 < LQ 24.5 < Pathwise 43.32 < GA 178.4 < Naive 211.2 < REINFORCE 272.5 < QL 928.8, matching `PARADIGM_ORDER`). The regret prose numbers all trace (RLS ~5, LQ 12→43, GA 90→300, REINFORCE 657/112/49, QL ~1000). So the artifacts are internally sound.

The failure is prose-vs-artifact drift introduced by the delta. The eighth paradigm MB-LG-Pathwise appears in the regret table (`cobweb_paradigms_results.tex:9`), the recovery table (rows 18-20), and all three figures (visually confirmed cyan/teal line + legend/title in every PNG), but `s09_dual_sim.tex` still describes seven:

- "compares seven learning paradigms" (line 4), "Seven paradigms share the budget" (line 15), figure caption "for seven learning paradigms" (line 29) — all now wrong for the cobweb panel (eight). (The fishery panel's "seven", lines 64/70/78, is a separate, unaffected script and is correct.)
- Param-recovery figure caption (line 43) names only "recursive least squares, the model-based LQ learner, and the MB-LG-REINFORCE ensemble mean" and asserts "All three methods converge" — but the figure title and lines show FOUR methods (adds MB-LG-Pathwise). Prose body line 20 repeats the same three-method framing and "All three methods recover the demand coefficients."
- Recovery-table caption (line 49): "both model-based learners estimate all four" — the table now has THREE model-based learners (Model-Based LQ, MB-LG-REINFORCE, MB-LG-Pathwise).
- The Results (line 18), policy-distance (line 22), and verdict (line 24) paragraphs never name MB-LG-Pathwise, though it is the third-ranked method by mean regret and sits in the middle of the ordering the prose walks through.

A referee opening the PDF sees an unexplained table row and an unexplained figure line in every cobweb figure, under a headcount that contradicts the legend. This is the "wrong attachment" smell in miniature — text and figures describe different sets of methods.

Second, a genuine (pre-existing, delta-independent) ordering error survives: line 22 states the MB-LG-REINFORCE learner "sits between the genetic algorithm and tabular Q-learning in the stable regime" for policy distance. The numbers say the opposite — stable policy distance is GA 0.177, Q-Learning 1.117, MB-LG-REINFORCE 1.462 (`stdout` 61-67; confirmed in the figure's stable panel where the orange REINFORCE line is the topmost, above blue Q-Learning). REINFORCE is the WORST, not "between." The author appears to have imported the regret ordering (where REINFORCE 657 does sit between GA 93 and QL 954) into the policy-distance paragraph.

Third, verdict line 24 calls MB-LG-REINFORCE "the third position on this frontier." With Pathwise added, the third position by mean regret is MB-LG-Pathwise (43.32); REINFORCE is sixth (272.5). Stale after the delta.

### (c) CHAPTER FIT — PARTIAL

The sim as computed strongly supports the frontier thesis, and the new paradigm strengthens it (clean pathwise-vs-score variance separation). But as published, the section under-delivers its own evidence: the single most illustrative new result (pathwise gradient rescuing the flat-regime policy that REINFORCE fails to converge) is present in every figure and table yet argued nowhere. The prose demonstrates the claim for seven of eight methods and silently drops the eighth.

### (d) EFFICIENCY / STANDARDS — PASS with one gap

- Per-component caching correct: `MB_PATHWISE_CONFIG = {**SHARED_CONFIG, ...}` (lines 55-57), own registry entry and cache key, `stdout:14` shows "Cache hit: MB-LG-Pathwise." Changing its hypers invalidates only its component.
- 20 seeds (>=10), SE reported everywhere. Flags `--data-only`/`--plots-only` present via `add_component_args`. Stdout format compliant (tables, no opinions).
- **Gap:** the new paradigm has no test. GA has `test_cobweb_ga_no_param_leak.py` (a real monkey-patch guardrail) and MBPO has `test_mbpo_real.py`; `MBPathwisePolicy` has neither a no-leak guardrail nor an analytic-gradient finite-difference check, despite its docstring asserting "Uses ONLY learned parameters. Does NOT read true regime params. Does NOT call solve_oracle_lq." (I verified that claim by reading — the only `solve_oracle_lq` token in the class is the docstring line — but nothing enforces it against regression.)

---

## 7-point checklist

1. **Algorithm identity — PASS.** All eight paradigms implement what they name; Pathwise gradient re-derived and correct; REINFORCE/GA/RLS/Oracle unchanged and previously verified. Honest relabels (MB-LG-REINFORCE not MBPO; Arifovic-without-election).
2. **Environment fidelity — PASS.** `CobwebEnv` matches the tex setup (p = a - bq + eps; r = pq - c/2 q^2 - phi/2 (q-q_prev)^2). Riccati grid-validated.
3. **Data integrity — PASS.** stdout ↔ both tables ↔ figures mutually consistent; numbers trace to `final_mean`/`final_se`; no hardcoding. Commit reproduced from scratch per message; artifacts self-consistent.
4. **Comparison fairness — PASS (with disclosed asymmetry).** Common random numbers per seed; T=500; 20 seeds. RLS-knows-(c,phi) asymmetry and the model-based warmup are disclosed in footnotes (carried over from polish). Pathwise uses the same model-fit as REINFORCE, isolating the gradient estimator — a fair within-family comparison.
5. **Theoretical sanity — PASS.** Regret non-negative, oracle unbeaten; (c,phi) recovered to machine precision under noiseless reward, correctly footnoted (line 20); pathwise beats score-function REINFORCE where the return surface is flat — expected.
6. **Information leakage — PASS.** Verified `MBPathwisePolicy` never reads `regime_params` and never calls `solve_oracle_lq`; fits from replay buffer only. Oracle's true-param use is legitimate; reference state used only for the evaluation metric, symmetric across paradigms.
7. **Seeds / reproducibility — PASS.** 20 seeds, distinct per-paradigm RNG offsets, SE = sigma/sqrt(N) throughout.

---

## Prior-audit open-item disposition

Prior audits (2026-05-19 at 25%, 2026-05-20 polish at 12%) both predate the delta by ~2 months and audited the seven-paradigm version. Their two tracked items:

1. **"MBPO" naming overshoot → RESOLVED (and preserved).** Renamed to MB-LG-REINFORCE across code, tests, figures, tables, and the eight s09 lines the polish enumerated. The delta did not regress this; the label is consistent in the new artifacts.
2. **Asymmetric structural prior (RLS given c,phi) → STILL OPEN as disclosed deferral.** The footnote at `s09:15` still flags the confound and defers the fourth panel to follow-up. Unchanged. Acceptable disclosure, not a new problem.

**New item the prior audits could not have seen (introduced by the delta):** the seven-vs-eight prose/artifact mismatch (Finding 1 below). The polish audit explicitly documented careful tex-artifact synchronization discipline for the rename; the delta abandoned that discipline. **REGRESSED** relative to the sync standard the polish set.

**Item prior audits MISSED (pre-existing, still live):** the "MB-LG-REINFORCE sits between the genetic algorithm and tabular Q-learning" policy-distance ordering error (Finding 2). Re-examined against stdout and the figure; the claim is contradicted by the numbers.

---

## Findings, severity-ordered

**1. [major / delta regression] The eighth paradigm MB-LG-Pathwise is in every artifact but absent from the prose; the section still says "seven paradigms" and three captions are numerically wrong.**
Evidence: added by `d532741`; `s09_dual_sim.tex` unchanged in that commit. "seven" at `s09:4,15,29`; param-recovery caption names three methods and "All three methods converge" at `s09:43` and `s09:20` while `cobweb_paradigms_param_recovery.png` shows four; recovery-table caption "both model-based learners" at `s09:49` while `cobweb_paradigms_final_recovery.tex` has three model-based learners (rows 10-20); Results/policy-distance/verdict paragraphs (`s09:18,22,24`) never mention the third-ranked method. Fix: add MB-LG-Pathwise to the enumeration, correct counts to eight (cobweb only), correct the two figure captions and the table caption, and add the pathwise-vs-score sentence the data already supports.

**2. [moderate / pre-existing, live] Policy-distance ordering claim is false.**
`s09:22` says MB-LG-REINFORCE "sits between the genetic algorithm and tabular Q-learning in the stable regime." Stable policy distance: GA 0.177, Q-Learning 1.117, MB-LG-REINFORCE 1.462 (`stdout:64,66,67`; confirmed in the policy-distance figure's stable panel). REINFORCE is the largest (worst), not between. Fix: restate as "above tabular Q-learning" or move the "between GA and QL" phrasing to the regret paragraph where it is true.

**3. [moderate / delta] Verdict mislabels the frontier's third position.**
`s09:24` calls MB-LG-REINFORCE "the third position on this frontier." By mean regret the third position is now MB-LG-Pathwise (43.3); REINFORCE is sixth (272.5). Fix in the same rewrite as Finding 1.

**4. [minor / standards] New paradigm has no test.** Unlike GA and MBPO, `MBPathwisePolicy` has no no-leak guardrail and no finite-difference check of its analytic gradient. Add a test that (i) monkey-patches `solve_oracle_lq`/regime-param access to raise inside the pathwise path, and (ii) checks `_pathwise_rollout`'s gradient against a numerical difference of `total_return` in (K0, Kq).

**5. [minor] Stale code header.** `cobweb_paradigms.py:1,4` says "six learning paradigms" and enumerates six (omits Model-Based LQ and MB-LG-Pathwise); registry has eight.

**6. [trivial] Dead placeholder.** `cobweb_env.py:116` computes an unused `* 0` term in the `__main__` smoke test. Harmless; delete on next touch.

---

The science is sound and the numbers reproduce and cohere; the defect is entirely that the writeup was not carried forward when the eighth paradigm was added, and it is pervasive across the subsection (headcount, two figure captions, one table caption, and every interpretive paragraph), plus one contradicted ordering claim. That is well past a single Reviewer-2 nick and lands in "the figures and the text describe different objects" territory, though the correct subset and correct numbers keep it short of a wrong-attachment reading.

**Bullshit score: 50%** — A hostile referee opens the PDF, counts eight lines in a figure captioned "seven," finds a table row and figure line the text never names, two captions that miscount their own panels, and a policy-distance sentence the table contradicts, and concludes the section was not proofread against its own regenerated artifacts.
