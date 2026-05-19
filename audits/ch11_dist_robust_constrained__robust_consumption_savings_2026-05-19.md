# Audit: ch11_dist_robust_constrained/sims/robust_consumption_savings.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex` (§ "Simulation Study: Consumption-Savings Under Model Mismatch", lines 537–595; surrounding KL-robust theory in lines 430–489)
**Cited paper PDFs read:** `papers/hansen2001_robust_control.md`, `papers/hansen2008_robustness.md`, `papers/iyengar2005_robust_dp.md`, `papers/nilim2005_robust_mdp.md`, `papers/petersen2000_minimax_entropy.md`, `papers/panaganti2022_robust_sample.md`, `papers/barillas2009_doubts_variability.md` — all present in chapter `papers/`. (Markdown digests read, source PDFs available alongside.)

## 1. Algorithm Identity

Both robust value iteration and robust Q-learning implement the Hansen-Sargent / Petersen multiplier-preference operator with **closed-form exponential tilting** for a KL ball, exactly as described in the tex (eq. \eqref{eq:robust_consumption} and \eqref{eq:robust_q_learning}):

```python
lw = -GAMMA * cont / theta   # cont[i] = V(R(w-c)+y_i)
lw -= lw.max()                # numerical stab
wts = income_probs * np.exp(lw)
q = wts / wts.sum()           # worst-case tilted measure
val = crra(c) + GAMMA * q.dot(cont)
```

This matches `q^*(y) \propto p_0(y) \exp(-\gamma V(\cdot)/\theta)` from the tex (line 449, 554). Two minor flags:

- The tilt uses `-γV/θ` rather than the more common `-V/θ` factor seen in Iyengar 2005 / Nilim-El Ghaoui 2005. The `γ` factor is consistent with the tex's stated formula, so this is an internal convention rather than an error. Hansen-Sargent 2008 ch. 7 uses `-V/θ`, so a hostile reviewer of the dual paper convention could ding the prose for not flagging the rescaling.
- Hansen-Sargent multiplier preferences come from `min_q E_q[V] + θ KL(q || p_0)`, the *penalized* form; the KL-ball constraint form is the dual. The code/tex conflate these in shorthand. Correct, but the equivalence should be cited (Petersen2000 is, line 456 — sufficient).

Robust Q-learning (`train_robust_q_learning`) recomputes `cont` from `Q` each step at the *current* `Q`, using the nominal kernel for sampling and the same exponential tilt for the TD target. This is the generative-model (simulator) variant flagged in the tex footnote (line 561-567). Correct.

The `argmax(Q[w_next, :w_next+1])` action-masking constraint (only consume up to current wealth) is consistent across standard and robust variants. Fair.

**No placeholder code.** Both VI and Q-learning have real implementations, not stubs.

## 2. Environment / MDP Fidelity

State: integer wealth `w ∈ {0,…,30}`. Action: integer consumption `c ∈ {0,…,w}`. Reward: CRRA `u(c) = c^(1-σ)/(1-σ)`, σ=2 (matches tex line 541). Transition: `w' = min(round(R(w-c)) + y, W_MAX)` with `R=1.02`, `y ∈ {1,2,3,4,5}`. Discount `γ=0.95`.

All match the tex word-for-word. The `int(round(R * s))` quantization is a discretization artifact (necessary because the state grid is integer) but unflagged in the tex. A hostile reviewer would note that gross return on small savings often rounds to identity (e.g., `s=1` → `round(1.02)=1`), so the savings technology is essentially `R≈1` for small savings. Worth a footnote but does not break the experiment.

Nominal `p_0 = (0.05, 0.10, 0.20, 0.30, 0.35)` and perturbed `p̃ = (0.30, 0.30, 0.20, 0.10, 0.10)` match tex lines 556–559. The perturbation flips the distribution to put mass on low-income states — exactly the worst case the robust agent should hedge against. Reasonable.

W_MAX cap at 30 introduces a hard truncation at the upper boundary. Could distort policies for high-wealth states (policy table shows the cap binding: at `w=30`, Standard DP consumes 5 — same as `w=15` and `w=20`). Reviewer-2 fodder but not invalidating.

## 3. Data Integrity

Pipeline traced: `compute_data` invokes four `compute_or_load` calls (shared / Q-learning / Robust-QL-5 / Robust-QL-2). `compute_shared` solves four VI problems and runs `evaluate` for each policy. `compute_q_learning` and the two robust variants each run training then evaluate. All numbers in `_stdout.txt` correspond to lines that print computed variables — no hardcoded "expected" values.

The cache key includes per-component configs (Q-learning uses `QL_CONFIG`, robust variants extend it with `robust_theta`), so editing θ for one robust variant won't poison the other's cache. `'version': 5` in ENV_PARAMS suggests several rounds of cache invalidation already happened — good hygiene.

Table values in `robust_consumption_savings_table.tex` match the stdout numbers to two decimals (e.g., Standard DP `nom=-4.938 pert=-8.693 delta=-76.0%` → table `-4.94 / -8.69 / -76.0`). Reproducible.

## 4. Comparison Fairness

All DP methods: same VI tolerance (`1e-10`), same max_iter (5000), same wealth grid. Converged in 425–435 iterations — comparable. Same `evaluate` function, same N_EVAL=5000 episodes × EVAL_LEN=100 steps, same evaluation seed (`seed=42` for DP; `seed=seed+1000` for QL variants). The eval RNG is a fresh `RandomState(seed)`, so all policies of the same type face the same income sequences. Fair.

Q-learning vs robust Q-learning: same hyperparameters except the TD target. Same episodes (`100_000`), same horizon (`100`), same visit-count LR (`C=100`), same `ε` schedule (1.0 → 0.05, decay 0.99998), same NOMINAL_INCOME sampling. Apples-to-apples.

**One unfairness:** evaluation seeds for QL variants are `seed+1000` (= 1000), but the DP eval seed is the hardcoded default `42`. The two are evaluated on different income sequences. Magnitude effect is small (5000 episodes makes the MC error tiny), but a hostile reviewer would flag the protocol inconsistency.

## 5. Theoretical Sanity Checks

Comparative statics direction is **correct**:
- θ = ∞ would recover standard DP (no tilting). Tex line 469 confirms.
- As θ decreases (5 → 2), the agent becomes more conservative. Stdout policy table: at `w=10`, Std consumes 5, θ=5 consumes 4, θ=2 consumes 4 (tied with θ=5 at this point but lower on average across `w`). At `w=5`, Std=4, θ=5=4, θ=2=3. Monotone in the right direction.
- Perturbed-model degradation: Std=−76.0%, θ=5=−75.5%, θ=2=−71.6%. Smaller `θ` (more robustness) → smaller degradation. Direction correct.

**Magnitude flag (reviewer-2):** the robust improvement under the perturbed model is small — Std `−8.69` vs θ=2 `−8.49`, a ~2% improvement in the level of return, and the oracle achieves `−7.60`. The robust policies leave most of the value on the table. A hostile reviewer would say: "If the perturbation is this severe (modal income flipped) and KL robustness only recovers 5pp of degradation, the case for robustness is weakly made." The tex prose understates this — line 572 says "Standard DP degrades by 76% while θ=2 robust policy degrades by 72%" without acknowledging the oracle gap.

**Nominal model:** all policies cluster around `−4.94`. Robust policies sacrifice ~0.01 of nominal return for robustness. Realistic and consistent with theory: the "price of robustness" is small for moderate θ.

Oracle DP has *worse* nominal performance (−5.46) than Standard DP (−4.94) but better perturbed (−7.60 vs −8.69). This is the expected misspecified-optimal pattern. Sane.

No method beats the oracle on the perturbed model. Sane.

## 6. Information Leakage

Standard DP and robust DP both see only the nominal distribution `NOMINAL_INCOME` during planning. Oracle DP is *explicitly* allowed to see the perturbed distribution `PERTURBED_INCOME` (line 245), which is the point of the oracle.

Q-learning and robust Q-learning sample income `y` from `NOMINAL_INCOME` during training (lines 161, 194). No peek at perturbed distribution. The robust TD target uses the nominal kernel for the inner minimization (consistent with the "generative model setting" footnote at tex line 564).

Evaluation always uses the appropriate target distribution for each scenario (nominal vs perturbed). No leakage between train and eval.

Clean on this axis.

## 7. Seed & Reproducibility

**Critical failure.** `N_SEEDS = 1` (line 48). The chapter CLAUDE.md mandates "minimum 10" seeds for stochastic methods, and the audit checklist item 7 requires N≥10 with means and standard errors reported.

Consequences:
- The `_aggregate_ql` function has a `n > 1` branch for standard errors that **never executes**. Every reported QL number is a single point estimate.
- The table renders QL rows without `\pm SE` (line 449 in code: `if r['nom_se'] is not None and r['nom_se'] > 0:` — `se` is 0.0 for n=1, so the bare-number branch runs).
- The figure draws `fill_between` only if `se is not None`, but for n=1 `se_policy` is a zero array, so a zero-width band gets drawn invisibly. Misleading: visually the QL curves look like point estimates while the DP curves are also point estimates — they aren't *actually* incomparable in display, but the prose says "Q-learning and robust Q-learning policies match their DP counterparts" without any uncertainty bound to back the claim.
- "Max policy deviation from Standard DP: 2" and "Max policy deviation from Robust DP: 0" are *single-seed* findings. The "0" deviation is plausibly seed-lucky.

DP is deterministic, so DP results are reproducible. QL with one seed is reproducible but not generalizable.

**Reproducibility infrastructure is otherwise good:** explicit seeds (`np.random.RandomState(seed)` in `train_q_learning` and `evaluate`), seed-indexed eval (`seed=seed+1000`), pickled caches with config hashes. The infrastructure expects multi-seed runs; the headline parameter is just turned off.

## Hostile-Reviewer Summary

The algorithm is correctly implemented (Hansen-Sargent multiplier preferences via exponential tilting, with both VI and Q-learning variants). The environment matches the tex. Comparative statics go in the right direction. No information leakage. The pipeline is clean and cached sensibly.

Three things a hostile reviewer catches:

1. **N_SEEDS = 1.** The repository's own simulation standard demands ≥10. Single-seed QL results are presented next to DP results as if they have the same epistemic status. The aggregation code is designed for multi-seed but the parameter is set to one. This is a 25%-level flag on its own.
2. **Magnitude of robust gain is modest.** θ=2 saves ~5pp of degradation under a severe perturbation (modal income flip), while the oracle saves ~37pp. The prose doesn't acknowledge that KL robustness recovers only a small fraction of the oracle gap.
3. **Eval seed protocol inconsistency.** DP evaluated with seed=42, QL with seed=1000. With 5000 eval episodes this matters little numerically, but the protocol asymmetry is noticeable.

The substance survives — the robust policies are real, more conservative in the right direction, and recover known patterns. The single-seed issue is fixable in one config line.

**Bullshit score: 25%** — Reviewer 2 catches N_SEEDS=1 against the repo's stated standard and notes that the robust gain is small relative to the oracle, but the algorithm identity, environment fidelity, and comparative statics all hold up under scrutiny.
