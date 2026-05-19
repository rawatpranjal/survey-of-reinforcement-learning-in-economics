# Audit: ch08_offline_rl/sims/offline_rl_pricing.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch08_offline_rl/tex/offline_rl.tex` (\subsection{Simulation Study: Offline RL for Dynamic Pricing}, label `sec:offline_sim`, lines 144–168; Table~\ref{tab:offline_main}; Figure~\ref{fig:offline_coverage})
**Cited paper PDFs read:** none in `ch08_offline_rl/papers/` (directory is empty — RULE C concern: chapter cites Ernst2005, Kumar2020, Kostrikov2022, Fujimoto2019, Chen2021, Emmons2022 without on-disk copies for verification; audit relies on the script's own structure plus widely-documented algorithm form).

## 1. Algorithm Identity

The script claims seven trained methods: BC, FQI, CQL, IQL, BCQ, DT, RvS. Going one by one.

**BC** (`train_bc`, lines 412–433). Plain cross-entropy MLP from state to action. Standard, no concerns.

**FQI** (`train_fqi`, lines 436–469). Iterative Bellman backup with `max_a Q(s', a)` regressed via MSE. No target network (the script's tex footnote on line 147 explicitly flags this as a deliberate pedagogical choice for the overestimation cascade). Defensible. The "200 outer iterations × 3 inner steps" budget is unusual but not wrong.

**CQL** (`train_cql`, lines 472–519). The conservative penalty IS present and correctly formed:
```
logsumexp = torch.logsumexp(q_all, dim=1)
cql_penalty = (logsumexp - q_pred).mean()
loss = bellman_loss + CQL_ALPHA * cql_penalty
```
This is the CQL(H) form from Kumar2020 §3.2 eq. (3): `α · (E_s log Σ_a exp Q(s,a) − E_{(s,a)~D} Q(s,a))`. Soft-target Q-network EMA τ=0.005 is also present. PASSES the CQL identity check.

**IQL** (`train_iql`, lines 522–565). Expectile regression IS present and correctly formed:
```
diff = q_vals - v_vals
weight = torch.where(diff > 0, IQL_TAU, 1 - IQL_TAU)
v_loss = (weight * diff ** 2).mean()
```
This is the asymmetric L2 expectile loss from Kostrikov2022 eq. (3) with τ=0.7. Q is then fit to `r + V(s')` rather than `r + max_a Q(s', a)`. PASSES the IQL identity check.

However, IQL paper specifies a **third** component — advantage-weighted regression (AWR) for the policy: `π = argmax_π E[exp(β(Q − V)) log π]`. This script instead extracts the policy by `argmax_a Q(s, a)` at deployment (line 558–564). This is a deviation: the script implements IQL's *value function* but not IQL's *policy extraction*. The deviation is not load-bearing for the reported result on this small action space (10 prices) but a hostile reviewer who runs the same evaluation on a continuous-action benchmark would catch it. Flag as a 25%-grade detail.

**BCQ** (`train_bcq`, lines 568–626). Critical issue. The chapter and the script cite **Fujimoto2019** ("Off-Policy Deep Reinforcement Learning without Exploration"), which is the **continuous-action** BCQ with VAE + perturbation network. The implementation here is the **discrete-action** BCQ-D variant from Fujimoto's *follow-up* paper (`Fujimoto et al. 2019b, "Benchmarking Batch Deep Reinforcement Learning Algorithms"`, arXiv:1910.01708), which uses the threshold-on-behavioral-probability `mask = (bc_probs >= τ · max_bc_prob)` mechanism — exactly what the script implements.

The script's BCQ implementation is itself correct as discrete BCQ. But the chapter cites the wrong paper. Both `\citep[BCQ,][]{Fujimoto2019}` in tex line 105 and the script's header comment attribute the threshold-based discrete BCQ to the 2019 ICML paper, which is the continuous version. This is a 25%-grade misattribution that a reviewer who reads Fujimoto2019 will catch immediately. The fix is to also cite `Fujimoto2019b` (or whatever bibkey the discrete benchmark paper takes) alongside `Fujimoto2019`. Currently `refs.bib` only has `Fujimoto2019`, so the bibkey needs adding.

**DT** (`DecisionTransformer`, lines 361–390; `train_dt`, lines 629–659). Fused-token form: each position embeds `(R_t, s_t, a_{t-1})` summed with positional embedding, then a causal transformer predicts `a_t`. The Chen2021 DT uses *separate* tokens for return, state, and action at distinct positions (3·T tokens per trajectory), while this implementation fuses them at a single position (T tokens). The fused-token variant is a documented simplification (it's faster and is the form RvS-T uses, per Emmons2022), but it is NOT exactly the Chen2021 DT. The chapter prose treats DT as Chen2021 throughout. A reviewer who diffs the architecture against Chen2021 figure 1 will see the discrepancy. 25%-grade.

**RvS** (`RvSNetwork`, lines 393–406; `train_rvs`, lines 662–688). Plain (state, return-to-go) → action MLP, cross-entropy on the realized action. This is the RvS-G variant from Emmons2022 §3.1 (return-conditioned, not goal-conditioned). Correct.

## 2. Environment / MDP Fidelity

Perishable inventory pricing MDP (`offline_rl_pricing.py` lines 32–49, 157–223). State `(i, d, t)` with `i ∈ {0,…,30}`, `d ∈ {0,…,3}`, `t ∈ {0,…,20}`. Action `p ∈ {1,…,10}`. Demand `Q ~ Poisson(λ_0[d]·exp(−0.15·p))`, `λ_0 = (1.5, 3.0, 5.0, 8.0)`. Reward `r = p · min(Q, i)`. Demand-regime transition matrix has diagonal 0.6 (matches tex line 147). Spoilage cost `−2.00` per unsold unit at termination. All match the tex writeup.

Two minor wrinkles:
- The script uses `inv * SALVAGE_VALUE` at the terminal step (line 167) AND adds it again at line 709 inside `evaluate_policy` (`episode_reward += inv * SALVAGE_VALUE`). I traced this: the `step` function returns the salvage value only when `time_remaining <= 0`, but `evaluate_policy`'s inner loop runs from `t=HORIZON` down to `t=1`, then exits without calling `step` at `t=0`. So the explicit `episode_reward += inv * SALVAGE_VALUE` after the loop is the SOLE salvage payment, not a duplicate. OK.
- DP oracle uses an approximation `max_q = min(i, int(rate * 5) + 10)` for the Poisson sum (line 208). For `λ_0[3]=8.0` and `p=1`, the rate is `8·exp(−0.15) ≈ 6.88`, so max_q caps at `min(30, 44) = 30`. The tail mass `P(Q > 30 | rate=6.88)` is negligible (`< 1e-15`). Fine in practice. A purist would integrate to a fixed quantile rather than `5·rate + 10`, but this is not a 25%-grade flag.

## 3. Data Integrity

Per-paradigm caching via `compute_or_load` with config hashing — config version 13 in the script (line 92). All numbers in the table and figure originate from `compute_data` returning the actual per-seed return arrays; the table writer reads `r['mean']` and `r['se']` from the result dict, not hardcoded.

**One serious flag**: the main-comparison table reports BC, BCQ, DT, RvS all at exactly `169.27 ± 0.60`. Four methods producing identical means and SEs *to 4 decimal places* is not a coincidence. Tracing the cause:

- BC's policy `argmax over MLP(s)` collapses to `p=10` for every state because the behavioral data is 85% concentrated on `p=10`, so the BC cross-entropy minimizer is constant-output `p=10`.
- BCQ's threshold (`τ=0.3` of max behavioral probability) at every state masks out everything except `p=10` because no other action gets above `0.3 · 0.85 = 0.255` empirical probability under the heavily concentrated behavioral policy.
- DT and RvS are conditioned on a target return `R* = dp_init_val ≈ 184`, well above any trajectory in the training set (typical behavioral return is much lower). The networks see this out-of-distribution target and output `argmax = p=10` because that is the only action with non-trivial training mass.

So all four methods collapse to the deterministic policy `π(s) = p=10`. Under the fixed evaluation RNG (`np.random.RandomState(seed + 10000)`), the four methods then sample IDENTICAL reward streams, yielding bit-identical per-seed means. This is technically "data integrity preserved" (the numbers are not hardcoded), but the result is empirically vacuous: the table claims to compare four offline RL methods, but four of the rows are the same policy under different names.

The chapter prose (line 156 onward) acknowledges this only for BCQ ("BCQ matches the behavioral at 88.0%; its action constraint restricts the policy to prices near 10"). The DT and RvS rows being numerically identical to BC is NOT discussed anywhere. A hostile reviewer who sees `169.27 ± 0.60` four times in a row will write "the authors apparently report the same number four times" and demand explanation. This is 50%-grade.

## 4. Comparison Fairness

- **Same offline dataset per seed across methods.** `compute_shared` generates one `offline_datasets[seed]` list, used by every method. Verified at lines 786–790 and the per-method runners at 833, 846, 859.
- **Same eval RNG per seed.** `np.random.RandomState(seed + 10000)` reused at 834, 847, 860.
- **Same training budget?** No — and this is openly documented in the audit notes file but not in the tex. Q-methods do 200 outer × 3–8 inner gradient steps. BC does 300 steps. DT does 500 steps. RvS does 500 steps. These are not equivalent. For the headline result this matters less (everything has converged) but for a coverage sweep where DT/RvS underperform pessimism methods at `ε_b=0.9`, a reviewer can argue under-training.
- **Coverage experiment uses different datasets per method per ε_b**: lines 870–923 generate fresh `offline_data` with `np.random.RandomState(seed + 20000)` for each `(method, ε_b, seed)`. Same seed across methods, so the datasets are the same — but they are NOT the same as the main-comparison datasets (which use `seed`, not `seed + 20000`). This is fine, but the script does not document why the offset differs. Minor.

## 5. Theoretical Sanity Checks

Predictions and what the data shows:

- DP Oracle at 100% by construction. ✓ (192.41).
- CQL, IQL above BC and above FQI. ✓ (91.9%, 92.0% > 88.0% > 81.2%).
- FQI below BC (overestimation cascade). ✓ (81.2% < 88.0%).
- BCQ ≈ BC under concentrated behavioral. ✓ (88.0% = 88.0%).
- Coverage sweep: FQI degrades at low coverage. ✓ at `ε_b=0.05` (54.0%). Also degrades at `ε_b=0.9` (49.1%) — the script footnote argues this is because more `max_a Q(s',a)` opportunities under broader coverage allow more overestimation. Defensible but unusual; reviewers may not buy "more data → worse" for an FQI variant without a target network. 25%-grade.
- DT/RvS in the 40–95% band: ✓ at `ε_b=0.05` and `ε_b=0.3` they sit at 87.8% (= BC), and at `ε_b=0.9` they drop to 83.5% and 82.2% — *worse than BC*. The drop at high coverage is unexplained in the tex and is the opposite of what one would expect (more coverage should help return-conditioned methods more, not less). The script's footnote about FQI does not extend to DT/RvS. Suggests these models are not really using the return-conditioning signal at all — they're just memorizing the BC mapping until coverage is wide enough that the target-return extrapolation kicks them off the BC manifold and onto a worse policy.

## 6. Information Leakage

The training functions consume only `tensors` (states, actions, rewards, next_states, dones, sa_features) derived from `offline_datasets[seed]`. No access to `dp_policy`, `dp_value`, or the env's `step` function during training. ✓

DT/RvS use `R* = dp_init_val ≈ 184` as the deployment-time target return. The audit notes file argues this is "a hyperparameter, not test-time access" because the operator could in principle pick this number. This is a defensible position only if the chapter discloses the choice and the sensitivity of the result to it. The current tex (lines 156–158) does not mention what `R*` was set to, does not say "we used the oracle value", and does not sweep `R*`. A reviewer who reads the code will notice. The audit notes file admits the sensitivity sweep is "deferred." 25%-grade for the *missing disclosure*, not for leakage per se — the leakage is real but bounded (it's a scalar, not a policy).

Evaluation uses `step` (the true simulator) for all methods. This is standard Monte-Carlo policy evaluation in a known simulator. Not leakage.

## 7. Seed & Reproducibility

- `torch.manual_seed(seed)` and `np.random.seed(seed)` set per training function. ✓
- `N_SEEDS = 20`. ✓ (CLAUDE.md requires ≥10).
- Mean and SE (`std / sqrt(N)`) reported per method. ✓
- Dataset RNG (`seed`) vs eval RNG (`seed + 10000`) separated. ✓
- Config-version field on every cached dict — bumping invalidates cache. ✓

Only nit: the script's `np.random.seed(seed)` is called inside `train_dt` and `train_rvs` but NOT inside `train_bc`, `train_fqi`, `train_cql`, `train_iql`, `train_bcq`. The Q-methods rely solely on `torch.manual_seed(seed)` for the random sampling of minibatch indices via `torch.randint`. This is reproducible (torch's CPU RNG is determined by `torch.manual_seed`), but inconsistent. Minor.

## Hostile-Reviewer Summary

The pessimism-family identities (CQL conservative penalty, IQL expectile regression) are correctly implemented and the headline result (FQI < BC < CQL ≈ IQL) cleanly demonstrates the chapter's pessimism thesis. The MDP and seeding are clean. The five-decimal coincidence of BC, BCQ, DT, and RvS at `169.27 ± 0.60` is the headline problem: four methods collapse to the same `argmax = p=10` policy under the heavily concentrated behavioral, producing bit-identical evaluation traces under the fixed eval RNG. The chapter's tex acknowledges this only for BCQ; the DT and RvS coincidence is not explained anywhere. Combined with the BCQ misattribution (Fujimoto2019 cited for the discrete-action variant that actually appears in Fujimoto2019b), the IQL policy-extraction simplification (argmax instead of AWR), and the fused-token DT (not the canonical Chen2021 3-token-per-step DT), there are three named-method-vs-implementation drifts plus a four-row collapse. The reviewer can write: "Either the supervised-conditioning methods are placeholders, or the experiment is too easy to distinguish them. The paper does not say which." That sentence is unanswerable without rerunning with a behavioral that does not concentrate on a single action — i.e., the experiment as designed cannot tell DT, RvS, BCQ, and BC apart.

**Bullshit score: 50%** — Reviewer 2 catches the four-way collapse and the BCQ misattribution; the result is not falsified but the supervised-conditioning rows of the table provide no information beyond the BC baseline, and the chapter prose does not own that fact. Major revise.
