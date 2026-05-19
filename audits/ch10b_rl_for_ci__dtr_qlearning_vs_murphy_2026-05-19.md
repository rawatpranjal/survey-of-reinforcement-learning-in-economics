# Audit: ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch10b_rl_for_ci/tex/rl_for_ci.tex` (§subsec:gmethods_bridge / subsec:murphy_watkins, lines 22–98)
**Cited paper PDFs read:** `papers/murphy2003dtr.md`, `papers/schulte2014qlearning.md` (markdown extractions). Murphy2003 PDF and Schulte2014 PDF present in `papers/`.

## 1. Algorithm Identity

The script labels its two tabular estimators "Murphy (FQI)" and "$Q$-learning". The hostile reading: "Murphy 2003" is a *family* of estimators built on regret/blip parametrisation (Murphy 2003 §3–4), not the plug-in conditional-mean computation implemented here. What the script actually computes for "Murphy" is the saturated empirical conditional mean over the discrete `(S_1, A_1, S_2, A_2)` cells, max'd over `A_2`, and recursed (lines 205–231). That is *plug-in g-computation* on a fully saturated tabular model, which in the offline-RL literature is exactly Fitted-Q-Iteration with a one-hot history feature map. The tex defends this rename: it explicitly says "the whole estimator is two regressions executed in sequence … known as Fitted $Q$-Iteration" (line 47), and the legend reads "Murphy (FQI)". So the label is honest, but a hostile DTR reviewer would flag that the simulation never engages with Murphy's *contribution* — the regret/blip reparameterisation — and so does not test "Murphy 2003 vs Watkins 1992" in the sharpest sense. It tests "plug-in g-computation vs TD".

The Q-learning implementation (lines 234–256) is one-pass stochastic-approximation TD with constant step `α=0.1`, two coupled tables `Q1[s1,a1]` and `Q2[s1,a1,s2,a2]`, target1 = `max_a Q2[s1,a1,s2,a]`. Bootstrapping order is Q1-first then Q2 within an inner loop, which is fine because Q1's bootstrap reads the previous iteration's Q2. The constant-α step means Q-learning cannot reach the empirical mean exactly — it converges to a bias-variance fixed point — which is why the panel-1 curve trails Murphy by ~0.002 at large N. This is correctly named in the tex caption.

The NN-FQI implementation (lines 423–451) fits Q2 by full-batch MSE for `n_epochs=200` Adam steps with no minibatching and no convergence check, then fits Q1 to the V2-bootstrap target the same way. DQN (lines 454–485) trains Q1 and Q2 *jointly* by minibatch TD with target detached, 8000 steps × 64-batch. Both are reasonable instantiations, but "NN-FQI = Murphy" is again a stretched label: Murphy's NN extension is unspecified in the original paper. The hostile reading: the panel-3 separation between NN-FQI and DQN is a training-procedure artifact (sequential full-batch vs joint minibatch) more than a deep statement about the FQI–Q-learning equivalence. The tex line 57 ("equivalence survives the transition to function approximation") is generous given the visible gap.

## 2. Environment / MDP Fidelity

Tabular DGP (lines 58–161): `S ∈ {1,…,5}`, binary action, logistic behavior policy on `S`, deterministic-or-stochastic transition (up with prob `P_IMPROVE` when treated and low; down with prob `P_WORSEN` when untreated and low; small random drift when high), outcome `Y = β_S·S_2 + β_A·A_2 + β_SA·1{S_2≤2}·A_2 + ε`. Match against tex line 55: "five-level ordinal status, binary action, logistic behavior policy, outcome that rewards treatment when status is low and penalises it when status is high." All four match. ✓

The treated-when-low / penalised-when-high outcome means `β_A + β_SA·1{S≤2}` is the action contrast: `-0.3` when high (untreated is better) and `+1.2` when low (treated is better). Stage-2 optimal policy is therefore "treat iff `S_2 ≤ 2`" which is exactly what `compute_oracle_tab` returns. ✓

High-dim DGP (lines 333–346): `S ∈ R^{10}` Gaussian, `S_2 = 0.5·S_1 + 0.5·a_1·e_0 + η`, outcome `Y = S_2'β + α_A·A_2 + α_SA·1{S_2[0]<0}·A_2 + ε`. Match against tex line 55: "p=10 continuous state, autoregressive transition with treatment effect on the first coordinate, same threshold interaction." ✓

Sequential ignorability holds by construction: `A_k` depends on `S_k` only via the logistic propensity, and the outcome's potential-outcome structure is `Y^*(a)|S_2 = S_2'β + α_A·a + α_SA·1{S_2[0]<thr}·a + ε` with `ε ⊥ A`. ✓

## 3. Data Integrity

`compute_data` calls `compute_or_load` for seven components (oracle tabular, Murphy sweep, qlearn_N sweep, qlearn_epochs sweep, oracle HD, FQI HD, DQN HD). Stdout shows all seven hit cache on the current run. Per-component configs (lines 78–95, 317–330) include all hyperparameters that should invalidate them. Spot-check: `MURPHY_CONFIG` includes `N_GRID` and the full DGP params; changing `BETA_SA` would invalidate Murphy and oracle (correct).

Output table (lines 681–703) is built from `m_means[-1]`, `q_means[-1]`, `f_means[-1]`, `d_means[-1]` — all computed live from cache contents. No hardcoded numbers. ✓ Stdout `_stdout.txt` lines 21–25 match the table file lines 5–10 to 4 decimals. ✓

Minor cleanliness violation: `compute_oracle_tab` and `compute_oracle_hd` print during compute (lines 183–185, 383–384). Per CLAUDE.md, `compute_data` should not produce side-effects beyond cache writes; printing belongs in `generate_outputs`. Cosmetic.

## 4. Comparison Fairness

Murphy and Q-learning use *different cohorts* at each `(N, seed)`: Murphy seed is `N*1000 + s` (line 264), Q-learning seed is `N*1000 + s + 7` (line 276). Hostile reading: the cleaner experimental design would re-use the same cohort across both estimators at each seed so the comparison is paired (within-seed). The current design measures whether the two estimators converge to the same value *in expectation*, which is what panels Q1 needs anyway, but a paired bootstrap would tighten the standard errors. At 50 seeds the unpaired SEs are small enough that this is not load-bearing.

NN-FQI and DQN at high dim use seeds `N*100+s` and `N*100+s+7` (lines 493, 505). Same unpaired-cohort caveat. At 20 seeds this matters more.

NN-FQI gets `N_FQI_EPOCHS=200` full-batch passes per stage (so 400 total Adam steps on the full data) while DQN gets `N_DQN_STEPS=8000` minibatch updates of size 64. Total gradient signal: NN-FQI sees `200·N` examples per stage × 2 stages = `400N`; DQN sees `8000·64 = 512,000` examples. At `N=5000`, NN-FQI = 2M, DQN = 0.5M. NN-FQI gets ~4× more gradient signal at the largest N. Hostile reading: the panel-3 NN-FQI > DQN gap could be a budget artifact, not an "FQI is more sample-efficient" statement. The tex line 57 attributes the gap to "trains the stage-two regression to convergence before bootstrapping the stage-one regression" which is a credible mechanism, but the budget asymmetry confounds the test. A fair-budget version (e.g., DQN with 10× more steps, or NN-FQI with 50 epochs) is not run.

## 5. Theoretical Sanity Checks

Tabular: under sequential ignorability and a saturated history-augmented Q-table, plug-in g-computation is the MLE and is consistent for `V*`. Murphy reaches `V(π̂)/V* = 1.0000 (SE 0.0000)` at `N=10000`. ✓

Tabular Q-learning with constant α=0.1 is *not* consistent for `V*` even with infinite replays at fixed N, because the constant step never decays. It converges to a fixed point biased away from the empirical mean by an amount that scales with `α·σ_Y² / (2 - α)` roughly. Panel-1 large-N value 0.9976 reflects this; panel-2 epochs sweep shows it plateaus at 0.9909, slightly *above* the Murphy reference 0.9907 at N=300 (within Monte Carlo noise). Both numbers are within 1.96·SE of each other. ✓ The tex line 57 names this the "constant-α bias", which is correct.

High-dim oracle uses the analytical optimal stage-2 rule "treat iff S2[0] < 0" derived from the outcome contrast `α_A + α_SA·1{S2[0]<0} = -0.3 + 1.5·1{S2[0]<0}`, which is positive iff `S2[0] < 0`. ✓ Stage-1 contrast then uses `E[V*_2 | S1, a1]` with the Gaussian-CDF for `P(S2[0]<0)`. Verified algebraically. V* = 0.7857 vs V(behavior) = 0.3350, so the gain from optimal policy is 0.45 in outcome units (large relative to SE 0.5). NN-FQI reaches 0.93·V* and DQN 0.91·V* at N=5000 — both well above behavior baseline 0.43. ✓

Suspicious point: the high-dim V_star MC has SE `0.0005` (line 374, returned but not printed). At M=200,000 with `σ_Y²+structural variance ≈ O(σ_β²|β|²)`, that's roughly right. Trust.

## 6. Information Leakage

Both tabular estimators receive `(S1, A1, S2, A2, Y)` only (lines 266, 278). No access to the oracle Q-table, no access to the optimal policy. ✓

NN-FQI and DQN receive the same cohort tuple (lines 495, 507). ✓

`evaluate_policy_tab` and `evaluate_policy_hd` are policy-evaluation oracles — they use the true transition kernel and outcome regression to score `(π̂_1, π̂_2)`. This is *evaluation* leakage in a benign sense: the algorithm itself does not see the oracle; only the audit/score does. Standard OPE-vs-oracle protocol. ✓

The tabular policy-evaluation function `evaluate_policy_tab` evaluates `π̂_2` at `(s1, a1, s2)` (line 199), which means it scores the *full history-conditioned* stage-2 rule that Murphy/Q-learning estimate, not the marginal `π̂_2(s2)`. This is the correct evaluation target for history-augmented DTR. ✓

## 7. Seed & Reproducibility

Tabular: `N_SEEDS = 50` ✓ (above the ≥10 floor). Seeds set deterministically per-cell via `np.random.default_rng(seed_integer)` ✓. Means and SEs reported with `±1.96·SE` confidence intervals on plots ✓.

High-dim: `N_SEEDS_HD = 20` ✓. PyTorch seeded via `torch.manual_seed(seed)` per call (line 424, 455). Note that PyTorch's CPU determinism is not fully guaranteed without `torch.use_deterministic_algorithms(True)` — but with Adam and fixed manual_seed the rerun-to-rerun variance should be near zero. Not stress-tested.

The table caption (tex line 68) reads "30 Monte Carlo seeds" for both tabular and HD settings, but the actual run is **50 (tabular) and 20 (HD)**. **Caption-vs-script mismatch.** Hostile reviewer catches this immediately. Fix the caption.

## Hostile-Reviewer Summary

The script delivers what the section advertises: empirical evidence that batch backward-regression and online TD recover the same optimal regime on a saturated tabular DTR, that Q-learning needs replay budget to catch the one-shot regression, and that neural analogues both rise toward V* in continuous state. The numerics are internally consistent (Q1 large-N Murphy hits V* exactly, Q2 plateau matches Murphy reference, Q3 NN methods both clear the behavior-policy baseline by a wide margin).

What Reviewer 2 catches: (1) the table caption says "30 Monte Carlo seeds" but the code runs 50 (tabular) and 20 (HD); (2) NN-FQI gets ~4× more total gradient signal than DQN at N=5000, confounding the "FQI is more sample-efficient" interpretation in panel 3; (3) "Murphy" is a generous label for what is actually saturated plug-in g-computation — Murphy 2003's regret/blip method is not implemented; (4) Murphy and Q-learning use different cohort seeds rather than paired draws; (5) `compute_data` has print side-effects inside the oracle component.

None of these is a substance-breaking issue. The headline equivalence claim survives, the constant-α bias story is correctly diagnosed, the analytical oracle is verified, and information leakage is clean. The caption error is the only thing the hostile reviewer would actually write a snarky comment about.

**Bullshit score: 20%** — Caption-vs-script seed count mismatch is a real Reviewer 2 catch and rounds up to 25%; the budget asymmetry in panel 3 is the next thing they'd circle. Substance, identification, and convergence story all hold. Round up to **25%**.
