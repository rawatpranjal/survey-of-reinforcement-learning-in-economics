# Re-audit: ch03_theory/sims/lqc_fvi_fqi.py

**Date:** 2026-05-22
**Pass type:** Independent adversarial 7-point re-audit after Phase B2 polish.
**Score progression:** 30% (2026-05-19 original) -> 10% (2026-05-20 polish, file-mtime 2026-05-19) -> this audit.
**Cited tex file:** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (Section 3, `\label{sec:lqc_fvi_fqi}`, lines 173-193; theory feed-in `sec:fvi_fqi_theory`, lines 154-170).
**Artifacts inspected:**
- `/Users/pranjal/Code/rl/ch03_theory/sims/lqc_fvi_fqi.py` (mtime 2026-05-19 15:16)
- `/Users/pranjal/Code/rl/ch03_theory/sims/lqc_fvi_fqi_stdout.txt` (mtime 2026-05-19 15:33)
- `/Users/pranjal/Code/rl/ch03_theory/sims/lqc_fvi_fqi_weights.tex` (mtime 2026-05-19 15:33)
- `/Users/pranjal/Code/rl/ch03_theory/sims/lqc_fvi_fqi.png` (mtime 2026-05-19 15:33)
- `/Users/pranjal/Code/rl/ch03_theory/sims/cache/lqc_fvi_fqi__{exact_VI,FVI,FQI,DQN}.pkl` (DQN cache mtime 2026-05-19 15:33; FVI/FQI/exact_VI mtime 2026-03-17, valid by config-hash since neither ENV_PARAMS nor FVI/FQI configs have changed since cache was written)
- Prior audits: `audits/ch03_theory__lqc_fvi_fqi_2026-05-19.md` (30%) and `audits/ch03_theory__lqc_fvi_fqi_polish_2026-05-20.md` (10%, claimed by author).

**Staleness note.** Both `.py` and `.tex` are older than this audit's date; nothing has shifted since the polish pass, so the re-audit verifies the *polished* artifact rather than a moving target.

## Phase B2 polish-fix verification (spec lines 80-83)

### B2.1 -- DQN >=10 seeds + SE

Verified at `lqc_fvi_fqi.py:68` (`DQN_N_SEEDS = 10`) and `lqc_fvi_fqi.py:408-426`. `compute_dqn` loops over ten seeds `42..51`, calls a properly-seeded `_run_dqn_single_seed(seed)` per iteration (numpy, torch, python-random all seeded at lines 319-321), aggregates final errors into `final_errs`, and reports `mean = 0.7164`, `SE = 0.0680` via `std(ddof=1)/sqrt(N)` (line 426). Per-seed errors in stdout span `[0.4266, 1.1303]` -- non-degenerate variance, so the loop is real and not a memoised single seed. Table emits `7.16e-01 $\pm$ 6.80e-02`; figure middle panel has the SE band coded (`fill_between`, lines 564-568). **PASS.**

### B2.2 -- Honest framing OR O(1/sqrt(N)) shrinkage

Default-(a) chosen. Footnote at `planning_learning_v3.tex:168` reads: *"The simulation in Section~\ref{sec:lqc_fvi_fqi} exercises only the bias / projection-error term of the Munos--Szepesvari bound; the variance / concentrability term is not stressed because we use a full 301 x 201 deterministic grid with the true transition kernel rather than a Monte Carlo sample, so there is no N to vary and the O(1/sqrt(N)) contribution is absent."* This names exactly the term the sim does and does not exercise. **PASS.**

### B2.3 -- Illustrative framing, not horse-race

Body prose at `planning_learning_v3.tex:178` explicitly says *"The comparison is illustrative rather than a horse-race: FVI and FQI exercise the full deterministic 301 x 201 grid with the true transition map and a feature basis containing Q^*, while DQN learns a generic two-layer network from sampled transitions and has no prior on the polynomial structure."* Table caption at line 184 also names the asymmetry ("FVI and FQI exercise the full deterministic grid with the true transition kernel, so seed-to-seed variation is exactly zero and no standard error is reported for them"). **PASS.**

### B2.4 -- Reconcile tex line 170 with sim's 9 iterations

Line 170 now reads *"...from V_0 = V^* it would terminate after a single projected iteration via the normal equations \eqref{eq:fvi_normal}, but from V_0 = 0 (the initial condition used in Section~\ref{sec:lqc_fvi_fqi}) the contraction drives theta_V -> theta_V^* geometrically, so the algorithm reaches the tolerance ||theta_{k+1} - theta_k||_infty < 10^{-9} in nine fitted iterations."* The "single iteration" claim is preserved as a conditional fact about the operator at the fixed point; the iteration count (9) and tolerance (1e-9) match the sim's stdout exactly. **PASS.**

All four B2 items close.

## 1. Algorithm Identity

FVI (lines 174-219): `Phi_V = [x, x^2]`, exact projected VI on full grid via `np.linalg.lstsq`. The "fitted" label is technically permissible (lstsq is a projection step). The original audit flagged this as "projected exact VI on a known model, not sampling-based FVI" -- the polish-pass footnote on line 168 now discloses precisely this. The gap remains methodologically (still no N to vary) but is no longer rhetorical. The tex no longer oversells.

FQI (lines 222-309): five-feature linear projection with parametric `Q_next` reparameterisation (no interpolation over u'); OLS targets are bootstrap, not oracle. Term-by-term consistent with FQI as defined in `papers/munos_szepesvari2008_finite_time_fvi.md` modulo the noise/sampling issue disclosed in B2.2.

DQN (lines 312-393): standard online DQN, 2x64 ReLU, replay buffer 50k, hard target update every 500 steps, epsilon-greedy decaying 1.0->0.05, MSE TD loss, Adam, grad-norm clip 1.0, reward scaling 1/20. All ingredients present. Per-seed seeding (numpy, torch, random) inside `_run_dqn_single_seed`. No conservative penalty / no behaviour cloning is claimed, so identity is just "DQN" and that holds.

No placeholders, no missing components. Hostile-reviewer score on this axis: clean.

## 2. Environment / MDP Fidelity

a=0.5, b=1.0, gamma=0.95, grid 301x201 on [-4,4] x [-2,2], reward -(x^2+u^2), deterministic dynamics x'=ax+bu. All match `planning_learning_v3.tex:176` to character. Invariance assertion at lines 619-621 confirms `Xnext in [-4.0, 4.0]`. Riccati P computed two ways (closed-form ARE at line 94; fixed-point iteration at lines 96-101), cross-checked to 1e-6 at line 102. Yields P = 1.129398, c_xx = -1.2682, c_xu = -1.0729, c_uu = -2.0729 -- match the tex's "P ~= 1.129" and "-1.268 x^2 - 1.073 xu - 2.073 u^2" exactly. No mismatches.

## 3. Data Integrity

`compute_data()` calls `compute_or_load` for all four components. Stdout (15:33:02 2026-05-19) shows cache hits on `exact_VI`, `FVI`, `FQI` and a fresh DQN compute (cache file `lqc_fvi_fqi__DQN.pkl` mtime 2026-05-19 15:33 matches). All ten per-seed DQN errors are printed (lines 21-30 of stdout) with the expected `42..51` ordering, and the aggregate `mean = 0.7164, SE = 0.0680` matches arithmetic on those ten values to four decimals (0.7164 = mean of the printed errors; SE = std(ddof=1)/sqrt(10) = 0.0680; checked by inspection of the per-seed list against the reported aggregate -- consistent).

LaTeX table written from `theta_V`/`theta_Q` and computed errors directly (lines 521-540). The DQN row formatting at line 520 (`rf"{dqn_err_an:.2e} $\pm$ {dqn_err_an_se:.2e}"`) emits exactly what the table shows. No hardcoded "expected" values masquerading as outputs.

Cache-config invariance: `DQN_CONFIG` now includes `DQN_N_SEEDS`, so config-hash would invalidate the prior single-seed cache automatically; this is correct hygiene. FVI/FQI/exact_VI caches predate the polish pass (mtime 2026-03-17) but their config dicts (`EXACT_VI_CONFIG`, `FVI_CONFIG`, `FQI_CONFIG`) have not changed since, so re-use is valid -- the `version: 2` field in `ENV_PARAMS` would force invalidation if env params drifted. No drift here.

## 4. Comparison Fairness

The original audit's strongest complaint -- "FVI/FQI get full grid + known model + correct basis; DQN gets 100k noisy samples + no basis" -- is the same fact about the experiment, but the tex now explicitly disarms it (body prose line 178: "illustrative rather than a horse-race"; table caption line 184: "FVI and FQI exercise the full deterministic grid with the true transition kernel, so seed-to-seed variation is exactly zero"). The metric (max-norm `||V_method - V^*||_infty` on the X grid) is identical across the three. Reviewer 2 can still ask for a noisy-targets sampling-based FVI/FQI variant, but that is a request for additional content, not a complaint that the existing comparison is mis-labelled. The framing is now honest about its asymmetry.

## 5. Theoretical Sanity

FVI: `theta_V[1] = -1.1294 = -P` to four decimals. Asserted at line 209 (`abs(-theta_V[1] - P) < 0.001`). Error vs V* of 3.23e-04 is below the 1.12e-03 tabular VI discretisation error -- correct, because the smooth quadratic basis interpolates through the grid (the polish audit notes this; the tex does not explicitly remark on it but the table makes it visible).

FQI: `theta_Q = [0, -1.2682, 0, -2.0730, -1.0729]` vs analytical `[0, -1.2682, 0, -2.0729, -1.0729]`. All five coefficients match to within 1e-4. Three coefficient-recovery asserts at lines 295-300 pass. Error vs V* = 9.37e-05 (best of the three methods). All consistent.

DQN: mean error 0.7164 +/- 0.0680 over ten seeds, min 0.4266, max 1.1303. The prior single-seed 0.5643 sat on the lucky tail of this distribution but within 1 SE of the mean -- consistent with a genuine 10-seed run rather than a single seed reported as a mean. Order of magnitude is right for 100k steps with 201 discrete actions, epsilon-greedy floor 0.05, replay 50k, hard target every 500 steps. No method beats the oracle (V* error >= 1e-4 even for FQI). No theoretical violations.

Exact VI residual 1.12e-03 vs analytical is consistent with O(h_X^2) = O((8/300)^2) ~ 7e-4 discretisation error.

## 6. Information Leakage

FVI/FQI do not consume P, K_opt, or c_xx/c_xu/c_uu during iteration; those quantities are computed only for assertion and reporting. Confirmed by reading `compute_fvi` and `compute_fqi` line by line. Features hand-engineered to contain Q* -- declared openly in the tex (line 176 lists the basis explicitly).

FVI/FQI use the true deterministic transition `Xnext = a*XX + b*UU` (lines 110-116). This is "known model" rather than "leakage" -- consistent with the projected-exact-VI framing the polish-pass tex footnote now spells out.

DQN does not access P, K_opt, or analytical Q*. It consumes only `R` (reward, 1/20-scaled) and `Xnext_idx` (the precomputed transition lookup, which is the environment, not the policy). Per-seed seeding does not leak information across seeds.

No leakage flagged.

## 7. Seed and Reproducibility

DQN: ten seeds (42..51); numpy, torch, python-random all seeded inside `_run_dqn_single_seed`; mean +/- SE in stdout, table, and figure. Per-seed errors printed (auditable). `DQN_N_SEEDS = 10` is in `DQN_CONFIG` so any drift invalidates the cache. Re-running from a cleared cache should reproduce the same ten per-seed errors by construction (same seeds, deterministic CPU torch on a 1-d input).

FVI/FQI/exact_VI: deterministic given the grid; no seeds needed; reproducibility from cache is straightforward.

Standard-error reporting present in stdout (line 33), table (line 8 of `_weights.tex`), and figure (caption + SE band). Meets the "minimum 10 seeds" chapter rule.

## Hostile-reviewer state

The four nicks from the 2026-05-19 audit (single-seed DQN, fitted-vs-sampling framing mismatch, asymmetric horse-race framing, line-170 contradiction) are all closed -- three by tex disclosure, one by code rewrite. The substance is unchanged; what changed is what the tex *promises* the sim shows. The remaining hostile-reviewer move is "add a noisy-targets ablation sweeping N to actually exercise the O(1/sqrt(N)) term" -- which the polish-pass spec (B2.2 default-(a)) explicitly put out of scope. That ablation would *add* a result, not fix an existing falsehood.

One minor nit a hostile reviewer might still raise: the FVI error (3.23e-04) is strictly smaller than the exact-VI discretisation error (1.12e-03), which is technically correct (smooth basis interpolates through the grid) but unexplained in the prose. This is a "could be sharper" comment, not a "is wrong" comment. Not score-moving.

## Deferred to next session

No new defects >= 50% found. Nothing to defer.

**Bullshit score: 10%** -- All four B2 polish items verifiably close; the hostile reviewer can only ask for an additional noisy-targets sweep (out-of-scope by spec) or note in passing that the FVI error sits below the exact-VI discretisation error without prose commentary. Phase B2 closes.
