# Audit: ch10b_rl_for_ci/sims/causal_bandit_parallel.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch10b_rl_for_ci/tex/rl_for_ci.tex` (§4a Causal Bandits, §simstudy Sim 2)
**Cited paper PDFs read:**
- `papers/lattimore2016causal.md` (Lattimore, Lattimore, Reid, NeurIPS 2016 — Algorithm 1 pseudocode, Theorems 1-2, m(q) construction)
- `papers/bareinboim2015mabuc.md` (Bareinboim, Forney, Pearl, NeurIPS 2015 — Algorithm 1 TS_C pseudocode, RDC, greedy-casino Table 1)

---

## 1. Algorithm Identity

**Successive Reject (Audibert-Bubeck 2010).** Implementation in `successive_reject()` (lines 191-249) matches the canonical schedule `n_k = floor((T-K)/(log_bar*(K-k+1)))` with `log_bar = 0.5 + sum_{i=2..K} 1/i`. Drops worst arm after each phase. Graph-blind — only observes `y`, ignores realized parents. Identity check: PASS.

**Lattimore Algorithm 1 (parallel bandit).** Compare lines 308-403 to the paper's pseudocode (lattimore2016causal.md lines 85-107):

| Paper step | Script step | Match? |
|---|---|---|
| Phase 1: T/2 do() pulls, observe X_t, Y_t | Lines 327-334 | OK |
| Estimate p_a = 2·T_a/T (frequency that X_i=x in obs phase) | `q_hat = X_obs.mean(axis=0)` then `p_y_given[i,j]` regression | OK |
| Compute m_hat = m(q_hat), set A_low = {a : p_a ≤ 1/m_hat} | Lines 357-374 — iterate tau in [2,N], minimize max(tau, |I_tau|) | OK |
| Phase 2: re-sample each a ∈ A_low for T_A = T/(2|A_low|) rounds | Lines 379-386 with `pulls_per_arm = (T-T_obs)//len(unbalanced_arms)` | OK, modulo integer division |
| Phase 2 re-estimates: `hat_mu_a = (1/T_A) sum y` | Lines 395-396 | OK |
| For arms NOT in A_low: use Phase-1 regression estimate `p_y_given[i,j]` | Lines 397-400 | OK |
| Recommend argmax over ALL arms | Line 402 | OK |

Identity check: PASS. Minor quibble — the paper estimates `hat_mu_a` for **all** arms from phase 1 (line 7 of pseudocode), then re-estimates ONLY arms in A_low from phase 2 (line 14: "Re-estimate"). The script does the same but uses a separate dict `arm_sums/arm_counts` for phase-2 estimates — same effect.

**Causal Thompson Sampling (Bareinboim-Forney-Pearl 2015).** Compare `causal_thompson_sampling()` (lines 421-461) to the paper's Algorithm 1 (bareinboim2015mabuc.md lines 147-160):

| Paper step | Script step | Match? |
|---|---|---|
| Seed `E(Y_{X=a}\|X=x) ← P_obs(y\|x)` for **all** (a, x) by consistency | `alpha[x_obs, x_obs] += y_obs` — seeds only on-intuition (a=x_obs) | **FAIL** |
| At each round: read intuition x | `x = rng.integers(0,2)` | OK |
| Compute Q1 = E(Y_{X=x'}\|X=x), Q2 = P(y\|X=x); bias = 1-\|Q1-Q2\|; w = [1,1], w[x or x'] ← bias | **Missing — no RDC bias weighting at all** | **FAIL** |
| Pick a = argmax over arms of `β(s_a,x, f_a,x) * w[a]` | `a = 0 if theta0 ≥ theta1 else 1` from posterior, no `w` multiplier | **FAIL (partial)** |
| Update Beta posterior conditioned on (x, a) | `alpha[x,a] += y; beta[x,a] += 1-y` | OK |

The script implements **"context-conditioned Thompson sampling"** with x as state, not Bareinboim's TS_C. The two algorithmically distinct ingredients of TS_C — (i) consistency-axiom seeding on the off-intuition arm and (ii) RDC-based bias weighting — are both absent. In the greedy-casino instance, conditioning on x alone is *already* sufficient to learn the right arm per intuition (because `P(Y=1|X=x, do(X=a))` is well-defined per (x,a) cell), so the script's algorithm still achieves bounded regret. But the algorithm is **mislabeled**: this is not the algorithm of Bareinboim 2015, and the chapter text on line 226 of `rl_for_ci.tex` describes TS_C with explicit reference to seeding via consistency and the RDC, neither of which is in the code. A hostile reviewer who pulls up Bareinboim's pseudocode side-by-side with the implementation will see the discrepancy immediately.

Identity check for TS_C: **FAIL.**

## 2. Environment / MDP Fidelity

**Parallel-bandit reward model.** The tex (line 224 of rl_for_ci.tex digest, and §4a) and the paper describe a parallel graph where the reward is `E[Y|X_1,...,X_N] = sigmoid/linear function of all X_i`. The script's reward (lines 165-175) collapses this to:
```
p_Y = 0.5 + ε if X_w = 1 else 0.5 - ε
```
That is, **only one coordinate (`w`) matters for the reward**, the other N-1 parents are reward-irrelevant. This is a degenerate special case of the parallel-bandit family. The paper's Section 5 simulations use a more general construction. The tex caption on line 345 of rl_for_ci.tex describes "gap ε = 0.3" without specifying that the reward depends on a single coordinate. A reviewer reading the caption and the code would conclude they describe different objects. The setting still falls within the parallel-bandit framework (it's a valid instance), but the m(q) ↔ regret correspondence theorem of Lattimore depends on the *full* reward function, not a single-coordinate reduction.

**m(q) construction.** `make_q_with_hardness` places `m_target` arms at q=0.05 and the rest at 0.5. Verification: I_τ = {i : min(q_i,1-q_i) < 1/τ}. For balanced arms (q=0.5), min=0.5, so I_τ excludes them when 1/τ < 0.5, i.e., τ ≥ 3. For extreme arms (q=0.05), included when 1/τ > 0.05, i.e., τ ≤ 20. So I_τ = {extreme set} for τ in [3, 20], with |I_τ| = m_target. Then m(q) = min_τ max(τ, m_target) = m_target. Identity check: PASS for m_target ∈ [3, 20]. At m_target=2, I_τ=∅ at τ≥3 (since 0.05<1/3 is true but 0.5<1/3 is false), so |I_τ|=2 only when τ=2 (1/2=0.5; min(0.05,0.95)=0.05<0.5 yes, min(0.5,0.5)=0.5<0.5 no). So m(2)=max(2,2)=2. OK. At m_target=48, I_τ = extreme set of size 48 when τ in [3,20]; at τ=48: 1/48≈0.021 < 0.05 so NONE included → |I_τ|=0, max=48. So m(48)=min(48, …)=48. OK. Construction verified.

**Greedy-casino payoffs.** `GREEDY_CASINO_PAYOFFS = [[0.10, 0.50], [0.50, 0.10]]`. Bareinboim Table 1 (paper line 47): under intuition x=0, on-intuition arm (a=0) pays 0.10, counter-intuition (a=1) pays 0.50; under x=1, on-intuition (a=1) pays 0.10, counter-intuition (a=0) pays 0.50. The script's matrix indexed by `[x, a]`:
- `[0, 0] = 0.10` (x=0, a=0 = on-intuition) ✓
- `[0, 1] = 0.50` (x=0, a=1 = counter) ✓
- `[1, 0] = 0.50` (x=1, a=0 = counter) ✓
- `[1, 1] = 0.10` (x=1, a=1 = on-intuition) ✓

The COMMENT in the code (line 89) says `x=0 (drunk + non-blinking): a=0 loses, a=1 wins` — this is correct. But the in-script narrative says following intuition pays 0.10 (loses) and counter pays 0.50 (wins). The paper's payoff intuition is the opposite of what natural human gambling instinct would do in the worked example. Numerically consistent with Table 1. Identity check: PASS.

Note however that the **observational and interventional marginals** under uniform x are: `(0.10+0.50)/2 = 0.30` for both arms (paper line 47 confirms). The script does this via `x = rng.integers(0,2)` then sample p = `GREEDY_CASINO_PAYOFFS[x,a]`. Good.

## 3. Data Integrity

`compute_data` (lines 602-618) runs three experiments (`regret_vs_m`, `regret_vs_T`, `mabuc`) and caches them with config-keyed `compute_or_load`. Per-experiment configs include all relevant scalars + `GREEDY_CASINO_PAYOFFS.tolist()` so changes to payoffs invalidate the MABUC cache. The stdout shows "Cache hit:" for all three experiments, confirming the displayed numbers correspond to the configured run. Stdout numbers (e.g., 0.0237 ± 0.0018 at m=2) match what would be expected from 2000 seeds — SE/mean ratio ≈ 0.076, consistent with sqrt(2000)·SE/mean ≈ √2000·0.08 ≈ 3-4, a reasonable noise scale.

The cumulative-regret ratio `200.5 / 0.66 ≈ 305x` is reported via division in print_stdout (line 765). The tex narrative in §simstudy line 333 says "a ratio of approximately 305×" — this matches the script's stdout. Numbers in tex flow from the script.

One issue: `arm_expected_reward` (lines 268-291) is **defined but unused** (the script always calls `arm_expected_reward_exact` instead, lines 515, 521, 553, 557). Dead code, not a correctness issue.

Integrity: PASS.

## 4. Comparison Fairness

**Same horizon T**: both Successive Reject and Lattimore Alg 1 get budget T=400 (or T_GRID values). PASS.

**Same arms**: both face the same 2N+1 = 101 arms. PASS.

**Same RNG state**: For each (m_target, seed) pair both algorithms are called with the same `rng` object — but note that Successive Reject is called BEFORE Lattimore Alg 1 in `run_regret_vs_m` (line 516 then 521), so they consume *different* draws from the shared rng. This is not a fairness problem per se (each algorithm sees a fresh stochastic environment), but it does mean the comparison is not paired-sample. Standard practice; not a real concern at n=2000.

**Information asymmetry**: Successive Reject ignores realized parents X (script confirms: `successive_reject` calls `pull_arm` which discards the parent vector — `pull_arm` returns only `y`, not `(y, x)`). Lattimore Alg 1 uses the realized parents during phase 1. This asymmetry is the *point* of the comparison, not a flaw.

**MABUC: vanilla TS does not get observational seed data**, while TS_C gets 200 observational samples. Vanilla TS could in principle also have seen the observational data (and would conclude both arms ≈ 0.3 marginal, so no help). The asymmetry is mild because the obs data doesn't help unconditioned TS. But a hostile reviewer will note it. The fairer comparison would give both algorithms the same data; the difference being how each uses it.

**No-context fairness**: vanilla TS in the MABUC instance does **not** observe x at all (line 477: `x = rng.integers(0,2)` is drawn fresh, never logged), while TS_C conditions on x. This is the algorithm-distinguishing feature, not unfairness — but framing in the tex doesn't clarify whether the comparison is "ignore x" vs "use x" (basically a context-vs-no-context comparison) or "TS without causal seeding/RDC" vs "TS_C with causal seeding/RDC". The result favours the former, not the latter.

Fairness: PASS with caveats (MABUC seed-data asymmetry is real but immaterial).

## 5. Theoretical Sanity Checks

**Lattimore Theorem 1**: simple regret ≤ C·sqrt(m·log(NT)/T). With T=400, N=50, log(NT)=log(20000)≈10, gap ε=0.3:
- m=2: bound ∝ sqrt(20/400) = 0.22. Empirical regret = 0.024. Empirical is well below bound — consistent with theorem being an upper bound.
- m=8: bound ∝ sqrt(80/400) = 0.45. Empirical = 0.073. OK.
- m=24: bound ∝ sqrt(240/400) = 0.77. Empirical = 0.125. OK.
- m=48: bound ∝ sqrt(480/400) = 1.10. Empirical = 0.071.

The non-monotonicity m=24 → m=48 (regret drops from 0.125 to 0.071) is **opposite to the predicted √m trend**. The script's reward model is responsible: at m=48, 48 of 50 arms are extreme (q=0.05), so the random best-arm coordinate `w` lands in the extreme set with probability 48/50. Then `do(X_w=1)` is a high-leverage low-probability action that the algorithm aggressively phase-2-samples. With T_remain/|A_low| = 200/48 ≈ 4 pulls per unbalanced arm, the recommended arm is selected reasonably well from phase-2 data. At m=24, only 24 of 50 arms are extreme; if `w` lands among extreme arms (prob 0.48), phase 2 samples it; if w lands in balanced set (prob 0.52), the optimal arm `do(X_w=1)` is **not** in A_low (since q_w=0.5 ≥ 1/m_hat for typical m_hat) and is estimated only from the regression on phase-1 obs data, which is noisier. So mixing the two cases gives higher mean regret at m=24 than m=48.

**This is a real artifact of the specific construction**, not a contradiction of Lattimore Theorem 1 (which is a worst-case upper bound; specific instances can do better). The tex on line 333 of rl_for_ci.tex describes the regret as "rising approximately as √(m*/T) across the grid", which is **factually false at m*=48** (regret drops). A hostile reviewer will pull up Table 1 of the script output and ask why the third column doesn't grow monotonically. The chapter prose either needs to acknowledge the non-monotone behaviour (and explain it via the single-coordinate reward model) or restrict the m grid to m ∈ {2, 8, 24} where the trend holds.

**Lattimore Theorem 2 lower bound**: Ω(sqrt(m/T)). At m=2, T=400, regret should be ≥ const·sqrt(2/400) = 0.07. Empirical Lattimore Alg 1 regret = 0.024. **Below the lower bound** — but Theorem 2 is a minimax lower bound over the *worst-case* problem instance; specific instances can be easier. Not a violation.

**Successive Reject baseline**: at T=400, K=101 arms, the schedule allocates n_k ≈ 1 pull per arm per phase (as computed in section above). The Audibert-Bubeck guarantee requires T ≫ K log K. Here T/K = 4 — far too small. Expected behaviour: SR effectively guesses randomly, regret ≈ ε = 0.3. Empirical = 0.28-0.32 across m. **Matches expectation.** The √(N/T)·ε reference line on Panel (a) is √(50/400)·0.3 = 0.106 — but SR's regret is 0.28, not 0.106. The reference line is a theoretical floor, not the actual SR performance. The caption (line 345) calls it "the theoretical √(N/T)·ε floor for graph-blind algorithms", which is misleading: this is a *lower bound* on regret achievable by ANY graph-blind algorithm asymptotically, NOT the actual SR regret at this T. Reviewer will flag.

**Greedy-casino MABUC**: TS without context sees marginal `P(Y|a)=0.3` for both arms — should accumulate linear regret with slope = mean per-round regret = (mean optimal - mean played) = 0.5 - 0.3 = 0.2 per round. So cumulative regret at T=1000 ≈ 200. Empirical: 200.49. **Matches theory exactly.** Excellent.

TS_C achieves bounded cumulative regret. Empirical: 0.66 at T=1000. With perfect Bayesian convergence, regret should be O(log T) per context — bounded yes. Matches Bareinboim's Figure 1 right panel qualitatively.

Sanity: PASS with the caveat that the **monotone √m trend claimed in tex line 333 fails at m=48**, and the **√(N/T) reference line in the figure is mislabelled as the SR "floor"**.

## 6. Information Leakage

**Successive Reject**: only sees arm index and reward — no access to graph, no access to realized non-intervened parents, no access to `w`, no access to `q`. Verified in `pull_arm()` (lines 252-265): it returns only `y`, never the realized parent vector. PASS.

**Lattimore Alg 1**: sees realized parents (X) and reward (Y) during phase 1, and the arm structure (which parent each arm intervenes on). It does NOT see the true `q` (it estimates `q_hat` from phase 1 data, line 337) and does NOT see `w` (the optimal coordinate; reward function structure is unknown). Verified: `lattimore_alg1(q_true, w, T, rng)` receives `q_true` and `w` as arguments BUT only passes them along to `sample_parents()` and `reward_under_do()` which are the environment simulators, not the algorithm logic. The algorithm itself only inspects `X_obs`, `Y_obs`, and `q_hat`. PASS.

**TS_C**: sees intuition x at each step, sees reward y. Does not see the structural parameters of GREEDY_CASINO_PAYOFFS. Receives `observational_data` as input — a sample of (x, y) pairs from following intuition. PASS.

**Vanilla TS**: sees only arm chosen and reward. Does NOT condition on x. Line 477 draws x but never stores it. PASS.

No leakage. PASS.

## 7. Seed & Reproducibility

- Seeds for regret panels: 2000 (well above the 10 minimum).
- Seeds for MABUC: 500.
- Seeds set as `seed = (m * 10_007 + s) & 0xFFFFFFFF` then `rng = np.random.default_rng(seed)` — deterministic given m,s.
- Mean and SE reported (1.96·SE = 95% CI bars).
- Cache files via `compute_or_load` ensure reproducibility across re-runs.

PASS.

---

## Hostile-Reviewer Summary

The Lattimore Algorithm 1 implementation matches the paper's pseudocode line-for-line; the m(q) construction is verified; the Successive Reject baseline is honestly under-budgeted (T=400 vs 101 arms) which is the right way to expose the graph-blind cost; the MABUC instance reproduces both the linear-regret baseline (200.49 vs theoretical 200) and the bounded-regret context-aware algorithm.

Three issues a hostile reviewer will catch:

1. **TS_C is not Bareinboim's TS_C.** The implementation omits both distinguishing features of the paper's Algorithm 1: (i) consistency-axiom seeding of the *off-intuition* arm `a ≠ x` (the script seeds the on-intuition arm instead, the wrong direction), and (ii) the RDC bias-weighting step that multiplies posterior samples by `w[a] = 1-|Q1-Q2|`. What the script implements is "Thompson sampling with x as observed context" — which works in this instance because the (x,a) cells are independently learnable, but it is *not* the algorithm of Bareinboim et al. 2015. The chapter text on line 226 explicitly describes the missing components, so a reviewer reading code and tex side-by-side will see the mismatch. This is the biggest exposure: the 305× ratio is real, but the named algorithm is mislabelled.

2. **The "monotone √m" claim in the chapter text fails at m=48.** Regret at m=24 is 0.125, at m=48 is 0.071 — a clear drop, opposite to the predicted √m trend. The cause is the single-coordinate reward model: at m=48 nearly all arms are unbalanced so phase 2 covers the optimal arm with high probability, regardless of where `w` lands. The tex narrative (rl_for_ci.tex line 333) flatly states "rising approximately as √(m\*/T) across the grid" — empirically false. Either truncate the m grid to {2, 8, 24} or rewrite the sentence to acknowledge the non-monotonicity and explain it.

3. **The √(N/T)·ε reference line is mislabelled as a "floor for graph-blind algorithms".** It is a lower-bound *rate* (Audibert-Bubeck Thm 4), not an upper bound on SR performance at finite T. SR's actual regret (~0.28) sits well above the reference line (~0.106) because T=400 is much smaller than K log K ≈ 467. A reviewer will read the caption and ask why SR is above its supposed floor.

Secondary issues: the parallel-bandit reward function reduces to a single high-leverage coordinate (the tex caption does not flag this); vanilla-TS in MABUC does not receive the observational seed data while TS_C does (immaterial for this instance, but asymmetric); the chapter text on line 226 attributes RDC and consistency-axiom seeding to the simulation when neither is implemented.

Most of the substance survives: the parallel-bandit graph-aware/blind gap is real, the m(q)-driven scaling is real for m ∈ {2,8,24}, and the MABUC bounded-vs-linear story is real. But the TS_C mislabelling and the m=48 reverse-trend in conjunction with the chapter prose's "monotone √m" claim are precisely the kind of errors a paranoid Reviewer 2 catches and writes paragraphs about — "the algorithm described and the algorithm implemented are not the same algorithm" is a heavy charge to defend in a rebuttal.

**Bullshit score: 55%** — Reviewer 2 catches the TS_C identity issue and the non-monotone m=48 result that contradicts the chapter's own narrative. The substance survives a major revision, but only after renaming the MABUC algorithm to something like "context-conditional TS" (not TS_C / not Bareinboim's algorithm) and editing the chapter prose to either truncate the m grid or acknowledge the non-monotone artefact. Rounded up from 50% because the tex prose makes specific claims (RDC, consistency-axiom seeding, monotone √m) that the code does not support — a hostile reviewer will treat that as a misrepresentation, not a sloppiness.
