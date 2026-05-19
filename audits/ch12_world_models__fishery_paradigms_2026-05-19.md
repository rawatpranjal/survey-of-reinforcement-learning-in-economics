# Audit: ch12_world_models/sims/fishery_paradigms.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch12_world_models/tex/s09_dual_sim.tex` (subsection §9.2, "Fishery with logistic growth", figure/table refs `figure:fc_fishery_curves`, `table:fc_fishery_results`)
**Cited paper PDFs read:** none from `papers/` directly in this audit pass; `papers/econ_adaptive_learning/ARIFOVIC_1994_GA_COBWEB.pdf`, `MARCET_SARGENT_1989.pdf` are present and referenced by the tex; classic Schaefer 1957 / Reed 1979 / Clark 1973 fishery references are NOT in `papers/`.

## 1. Algorithm Identity

Six paradigms claimed: Oracle, RLS, Model-Based LQ, Q-Learning, Naive, Arifovic GA. The tex (line 70 of s09_dual_sim.tex) describes the fishery subset of the cobweb panel.

- **Oracle.** Grid-based VI with Gauss-Hermite quadrature on a 50×25 stock/harvest grid (`fishery_env.solve_oracle_dp`). Reaches `converged=True` in 325 iterations. The deterministic optimal policy converges to a stable stock of about 4.38 (analytically below `K/2=5` because the quadratic harvest cost penalizes large harvests, shifting the optimum below MSY). This is a sensible discounted-DP solution; identity holds.
- **RLS.** Recursive least squares on `(s_t, -s_t^2)` → estimates `(r, r/K)`, recovers `K` by division, re-solves the inner DP every 25 steps with known `(p, c, sigma)`. Implementation matches the textbook RLS update (information-form: `R += xx^T`, `theta += R^{-1} x (target - x@theta)`). Matches the tex prose ("known cost parameters", "re-solving the DP every twenty-five observations"). Identity holds.
- **Model-Based LQ.** Mis-named. The tex calls this the "model-based LQ learner" and says it "estimates `(r̂, K̂, p̂, ĉ)` jointly by least squares, refits the DP every twenty-five observations" — that is accurate to the code. The label "LQ" is the cobweb-panel naming; on the fishery the planner is grid-DP (non-linear dynamics), not Riccati. A hostile reviewer will catch the name leakage from cobweb to fishery and ask why an algorithm called "LQ" runs a non-linear-dynamics DP. The substance is correct (joint LS estimation of all four parameters + DP planning); the label is sloppy. Identity holds modulo nomenclature drift.
- **Q-Learning.** Tabular ε-greedy on a 30×21 (`g_s × g_h`) bucketed grid with α=0.1 and ε decaying 0.3 → 0.01 linearly over T=500. Matches Watkins-Dayan canonical update. Identity holds.
- **Naive.** Constant `h_fixed=0.5`. The tex (line 70) calls it "a reasonable steady-state guess that lies between zero and the analytic MSY." Identity holds, but see comment in §5 — this is NOT the open-access / myopic agent the audit prompt expected. A purely myopic agent would harvest `p/c = 10` per step (unconstrained reward maximizer), collapsing the stock immediately. There is no open-access paradigm in this sim.
- **Arifovic GA.** Binary chromosomes of constant harvest rules, fitness-proportional selection on running mean realized profit, single-point crossover, bit-flip mutation. Crucially, `_evolve()` runs an **election operator** that scores child vs parent using `pi = p*h - 0.5*c*h^2` with the *true* `(p, c)` (lines 263–265). The tex says "evolves them under fitness-proportional selection and the election operator with known cost parameters" — this matches. But the election operator as implemented uses **only myopic period profit at `last_s` as the stock proxy**, not the full discounted bioeconomic objective. So children are scored on a static one-shot profit metric, ignoring stock dynamics. This is a *partial* election operator. The tex hand-waves over this; a hostile reviewer would call it a misleading description of "election operator with known cost parameters." Identity: partial.

## 2. Environment / MDP Fidelity

Schaefer surplus production: `S_{t+1} = S_t + r*S_t*(1 - S_t/K) - h_t + eps_t`, max-clipped at 0 and `s_max=1.5K`. Reward `r_t = p*h_t - (c/2)*h_t^2` (linear-quadratic harvest cost). The tex (line 67) gives parameters `r=0.4, K=10, p=2, c=0.2, sigma=0.3, gamma=0.95, T=500, s_0=K`. Script `ENV_PARAMS` (line 23): `r=0.4, K=10.0, p=2.0, c=0.2, sigma=0.3`. `SHARED_CONFIG`: `GAMMA=0.95, T_EPISODE=500`. `env.reset()` returns `K=10`. All match.

Two notes:
- The reward `p*h - (c/2)*h^2` is a *quadratic-cost harvest reward*, not the classical Schaefer/Clark "revenue minus per-unit cost" form. With `p=2, c=0.2`, the myopic max is at `h=p/c=10`. With stock-dependent reward (which Clark 1973 uses, `r(s,h) = ph - c(s)·h`) the bioeconomic equilibrium under open access differs. The tex (line 64) explicitly calls the reward "linear-quadratic", so this is documented; it is not the Schaefer-Clark formulation an econ reviewer would expect. Acceptable as a stylized choice but worth flagging.
- The hard cap `s_next = min(s_next, s_max=1.5K)` is a numerical guardrail not present in Schaefer 1957. With `sigma=0.3` and `K=10` this rarely binds. Not a substantive issue.
- The oracle DP discounts log normally over a long horizon; the deterministic steady state is `s≈4.38`, below `K/2=5`. This is mechanically correct: the quadratic harvest cost penalizes deviations from a moderate harvest rate, so the optimal policy chooses a slightly more aggressive depletion than MSY-conserving. A reviewer who expected `s* = K/2` exactly would object; the tex does not explicitly state where the optimal steady state is. Substance holds.

Fidelity: 8/10. Schaefer dynamics are faithful; reward functional form is non-standard but documented.

## 3. Data Integrity

`compute_data` follows the canonical pattern: `compute_shared` rolls out the Oracle once per seed and stores `oracle_rewards[seed]`; each paradigm's `compute_paradigm` rolls out the paradigm with the same seed and computes per-step regret as `cumsum(oracle - paradigm)`. Cache hits across all six paradigms (per `_stdout.txt`). Stdout numbers (Oracle 0.00, RLS 13.67, MB-LQ 14.69, QL 274.71, Naive 447.35, GA 706.13) match the LaTeX table (`fishery_paradigms_results.tex`) exactly. The figure caption claims twenty seeds, mean ± SE — config has `N_SEEDS=20`, results report SE.

One issue: `compute_paradigm` calls `np.random.seed(s)` at the top (line 438) but the paradigms then create their own `np.random.default_rng(seed + offset)` generators (RLS doesn't; Q-Learning, GA, MBPO do), so the `np.random.seed(s)` is a no-op for the paradigms that use default_rng but does affect anything inside `make_paradigm`/`solve_oracle_dp` that uses the legacy `np.random`. `solve_oracle_dp` is deterministic (no random calls visible), so this is harmless in practice but a sloppy pattern.

Note that the script also reports trajectories only via `mean_curve` and `final_regret` — no stock trajectories, no harvest trajectories, no collapse indicator. The audit prompt asks "compute_data runs paradigms over periods, reports stock + harvest trajectories" — **it reports rewards/regret only, not stocks or harvests**. This is a real gap if the chapter wants to argue "open-access overharvests / fishery collapses," but the tex (line 73) doesn't make that argument; it sticks to regret comparison. So integrity holds for what is reported, but the figure does not let a reader verify stock dynamics.

## 4. Comparison Fairness

Each paradigm is rolled out per seed using `FisheryEnv(seed=s)`, which seeds the env's internal rng. The same `s` is used for Oracle in `compute_shared` and for each paradigm in `compute_paradigm`. Therefore **the noise sequence is identical across paradigms within a seed**, which is the correct controlled-comparison protocol; regret is a paired comparison. Confirmed by reading `FisheryEnv.__init__` (line 24 of fishery_env.py: `self.rng = np.random.default_rng(seed)`) and `step` (line 35: `eps = self.rng.normal(0, sigma)`).

Same `T=500`, same env params, same horizon, same `N_SEEDS=20`. All paradigms share the rollout loop in `rollout()`. Each paradigm consumes the same number of environment steps. The hyperparameter budgets differ across paradigms (Q-Learning has 30×21 grid, GA has 30-pop × 10-gen, RLS has 25-step refit cadence, etc.); the tex acknowledges this as "the experiment is deliberately favorable to structured learners". A hostile reviewer would still ask why GA had a 30-population × 10-generation evolution loop tuned at exactly these values — there is no hyperparameter sweep evidence. Within stated configs, comparison is fair.

## 5. Theoretical Sanity Checks

Where the audit prompt sets the expectation:
- **Open-access (myopic) overharvests, fishery collapses.** No paradigm here implements an open-access / per-period-profit-maximizing agent. The "Naive" baseline is a fixed precautionary `h=0.5 < h_MSY=1`, which DOES NOT collapse the stock (steady state under naive ≈ 8.5, well above zero). The sim does not exhibit the textbook open-access tragedy that Clark / Reed analyze. This is a missed opportunity and a misalignment with the chapter framing if the framing were "open-access vs managed", but the tex explicitly bills Naive as a "no-learning floor" baseline, not as open-access. So no false claim, but the audit-prompt-expected sanity check is absent.
- **Optimal SDP maintains stock near MSY.** Verified directly: oracle steady-state stock ≈ 4.37 ± 0.34 over 20 seeds (under `sigma=0.3`), slightly below `K/2=5` because of the quadratic harvest cost shifting the optimum away from MSY-preservation. Reasonable; consistent with a precautionary-not-conservative optimal policy.
- **Final regret ordering.** RLS (13.7) < MB-LQ (14.7) < Q-Learning (275) < Naive (447) < GA (706). RLS beating MB-LQ on regret because RLS gets `(p, c, sigma)` for free (only 2 parameters to estimate) while MB-LQ estimates all 4. Consistent with the cobweb story and a defensible result.
- **GA worse than Naive.** This is interesting: GA finishes at 706 vs Naive at 447. The tex (line 73) explains it as "the genetic algorithm's binary-encoded constant harvest is the wrong functional form for an environment where the optimal action varies smoothly with the stock." Mechanically the GA chromosome decodes to a *single* constant `h` (10-bit value in `[0, h_max]`), same functional class as Naive, but the GA is *searching* over this class with noisy fitness, so it dithers around an inferior value and accumulates more regret than the lucky-fixed-at-0.5 Naive baseline. Plausible, but a hostile reviewer would note that this makes GA strictly worse than its own no-learning floor of the same functional class — which is a weak comparison.
- **Oracle on top.** Oracle has zero regret by construction (`oracle_rewards - oracle_rewards = 0`). Visible in the figure and table. Trivial sanity check passes.

Open issue: no Bellman-equation residual check, no comparison of the learned RLS/MB-LQ DP value function to the oracle value function, no parameter-recovery table for the fishery panel (the cobweb has one; the fishery does not). The tex for the fishery panel (lines 72–73) is one paragraph of qualitative regret narrative; there is no quantitative validation that RLS's recovered `(r̂, K̂)` actually converge to truth.

Sanity: 6/10. The two structured learners' near-oracle regret is plausible; the ordering of Naive vs GA is explained but uncomfortable; no parameter-recovery quantification.

## 6. Information Leakage

- **Oracle** legitimately knows `(r, K, p, c, sigma, gamma)`. By definition, not leakage.
- **RLS** has access to `(p, c, sigma)` as stated in the tex ("known cost parameters"). Code confirms (line 111–112). Legitimate by chapter framing, though econometrically this is a strong assumption.
- **Model-Based LQ** estimates all four parameters; no leakage of truth into the planner — the DP is re-solved with point estimates. Clean.
- **Q-Learning** sees only `(s, h, r, s_next)`. The h_max bucket bound is set from `1.5 * params['r'] * params['K'] / 4.0 = 1.5` — this leaks `r` and `K` into the action grid. A hostile reviewer would flag this: a pure model-free Q-learner shouldn't know `r*K/4` (the MSY harvest), it should learn from raw rewards. Mitigation: `h_max` is only setting the action support; the policy still has to discover what to do. But knowing that the optimal harvest is on the order of `r*K/4` is a substantive prior. Documented or not depends on whether the chapter advertises Q-Learning as having no prior — the tex (line 70) is silent on this.
- **Arifovic GA** election operator uses true `(p, c)` (lines 248, 263). Stated in tex; legitimate by framing.
- **Naive** has no information beyond the constant. Clean.

Leakage: One real issue — Q-Learning's action grid is set from true `r` and `K` (line 165: `self.h_max = 1.5 * params['r'] * params['K'] / 4.0`). The same line appears in GA (line 221) and MBPO (line 305). For GA this is the chromosome decode range; for MBPO this is the exploration noise clip. These are not "the algorithm cheats" in the strong sense (they bound the action space, not the policy), but they pre-tune the action space to the problem in a way a hostile reviewer can point at. The tex does not document this.

## 7. Seed & Reproducibility

`N_SEEDS=20` (meets the chapter's stated minimum of 10). Means and standard errors are reported across the 20 seeds. Seeds for the env are `range(20)`; paradigm-internal rngs are `seed + offset` (deterministic per paradigm). Reproducible. Cache files (`compute_or_load`) are keyed on the config dict — changing hyperparameters re-runs the right paradigm. Reproducibility 9/10.

One minor issue: `np.random.seed(s)` at line 438 sets the legacy global RNG, but most paradigms use `default_rng` and ignore it. Inconsistent but not a correctness issue.

## Hostile-Reviewer Summary

The fishery panel is the simpler sibling of the cobweb panel and inherits most of the same machinery. Substantive findings: the Schaefer dynamics and oracle DP are faithful; the regret comparison is paired across paradigms with shared noise sequences; results are deterministic, reproducible, and consistent with chapter prose. The main weaknesses are: (i) "Model-Based LQ" is mis-named on the fishery (no LQ, the planner is grid-DP); (ii) the Naive baseline is *not* an open-access / myopic agent, so the sim does not exhibit the textbook fishery-collapse tragedy a bioeconomist would look for; (iii) the GA "election operator" is a myopic-profit comparison that ignores stock dynamics, not the full Arifovic election operator; (iv) Q-Learning's action grid is pre-set from true `(r, K)`, a mild but undocumented prior; (v) no parameter-recovery quantification for the fishery panel (the cobweb has one, the fishery does not), and the figure shows only regret, not stocks or harvests; (vi) the regret-ordering Naive < GA on a one-parameter constant-rule class is awkward and a thin-evidence finding. None of these invalidates the substance; together they constitute one of those Reviewer-2 comments that demands a revision but doesn't sink the result.

**Bullshit score: 30%** — Reviewer 2 catches the "LQ" mis-label, the missing open-access agent, the partial election operator, and the Q-Learning action-grid prior; one or two of these will demand revision but the regret-comparison substance holds.
