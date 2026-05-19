# Audit: ch11_dist_robust_constrained/sims/carbon_constrained_production.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex` (lines 220–407; Section "Constrained MDPs and Lagrangian Methods" + subsubsection "Simulation Study: Carbon-Constrained Production")
**Cited paper PDFs read (chapter `papers/`):** `achiam2017_cpo.pdf`, `tessler2019_rcpo.pdf`, `paternain2019_zero_duality.pdf`, `ding2020_npg_pd.pdf`, `miryoosefi2019_convex_constraints.pdf` (all present and referenced in the chapter tex). Stooke 2020 (PID) is cited in tex but no PDF is in `papers/`. Altman 1999 (Constrained MDPs book) is the foundational reference and is not in `papers/` either, but the LP-on-occupation-measures formulation in §2 of the tex matches Altman's standard treatment.

## 1. Algorithm Identity

The script implements two algorithms:

(a) **LP Oracle on occupation measures.** Lines 180–222. Variables $\nu(s,a) \geq 0$, equality constraints $\sum_a \nu(s,a) - \gamma \sum_{s',a'} P(s|s',a')\nu(s',a') = \alpha(s)$, single inequality $\sum_{s,a} C(s,a)\nu(s,a) \leq d$, objective $\max \sum_{s,a} R(s,a)\nu(s,a)$. This is the textbook Altman 1999 LP formulation. The dual variable on the inequality is extracted from `result.ineqlin.marginals` and reported as $\lambda^*$. Match against tex Eq. (LP) and Eq. (8) lines 215–227: ✓ correct.

(b) **Lagrangian Q-learning with single-timescale dual ascent.** Lines 227–318. Inner loop runs tabular Q-learning on the Lagrangian one-step reward $r - \lambda c$ (line 263); dual update every 1000 episodes via $\lambda \gets [\lambda + \eta_\lambda (\hat{J}_C - d)]_+$ (line 279). This corresponds to Tessler 2019 RCPO with the multiplier on a slow timescale and the Q-table on a fast timescale (RCPO three-timescale scheme; this is the two-timescale version with constant $\lambda$ within a 1000-ep window). The projection `max(0, ·)` enforces $\lambda \geq 0$ correctly.

A subtle pitfall: the Q-table is NOT reset when $\lambda$ changes. The Q-table is learning $Q_\lambda$ for the current $\lambda$, but old experiences in the buffer (via TD bootstrapping) were generated under earlier $\lambda$ values. This is the usual primal-dual coupling and not a bug — Tessler 2019 explicitly relies on the fast Q-timescale to track $\lambda$ — but it is a known source of the oscillation that the tex caption refers to.

The cost-violation signal uses a truncation correction: `avg_cost_inf = avg_cost / (1 - gamma^H)` where $H=100, \gamma=0.95$. The factor is $1 - 0.95^{100} \approx 0.9941$, i.e. a 0.6% upward correction. With the true value of an infinite-horizon discounted cost lower-bounded by the truncated horizon cost, this is a small and reasonable correction; effectively negligible.

**Match to chapter claims.** The tex names three algorithm families (Lagrangian dual ascent, CPO, PPO-Lagrangian with PID dampening per Stooke 2020). The sim implements only the first. The tex (line 375) correctly states the comparison is "the constrained LP oracle, unconstrained Q-learning, and Lagrangian Q-learning with dual ascent" — it does NOT claim CPO or PID are in the sim. No false advertising.

Verdict: ✓ Lagrangian Q-learning correctly implemented; LP oracle correctly implemented; no claim of CPO or PID. Algorithm identity is honest.

## 2. Environment / MDP Fidelity

State: $(I, \text{regime})$ with $I \in \{0,\ldots,8\}$ and regime $\in\{\text{low, high}\}$ → 18 states. Action: $(a^{\text{prod}}, e) \in \{0,1,2,3\}\times\{\text{dirty, clean}\}$ → 8 actions. ✓ matches tex.

Dynamics: $I_{t+1} = (I_t + a^{\text{prod}}_t) - \min(I_t+a^{\text{prod}}_t, D_t)$, regime via 2-state Markov chain with matrix $[[0.8,0.2],[0.3,0.7]]$. Demand pmfs over $\{0,1,2,3,4\}$. The reward in code is `price·sales − prod_cost − hold_cost·inv_next` (line 111). The tex Eq. (eq:carbon_cmdp) writes the reward as $p\min(I_t+a^{\text{prod}}_t, D_t) - \kappa_{e_t} a^{\text{prod}}_t - h(I_t+a^{\text{prod}}_t-D_t)^+$. These match because $\text{inv\_next} = (I_t+a^{\text{prod}}_t - D_t)^+$. ✓

Cost: $c = \xi_{e_t} a^{\text{prod}}_t$ where $\xi \in \{3.0, 0.5\}$ for dirty/clean. ✓ matches tex.

One small discrepancy worth flagging: the tex says $I_t \in \{0,\ldots,8\}$ but the code stores `I_MAX=8` and has `n_inv = self.I_MAX + 1 = 9`. ✓ consistent.

The inventory cap at `min(inv + prod, I_MAX)` (line 106) effectively wastes production above I_MAX. This is not stated in the tex but is plausible. Minor reviewer nit.

Verdict: ✓ environment is faithful to tex.

## 3. Data Integrity

`compute_data()` (lines 339–388): builds env matrices, runs VI for unconstrained DP, computes carbon_budget = 0.3 × unconstrained cost, solves LP, runs unconstrained QL, runs Lagrangian QL. Numbers in the .tex table and stdout (`carbon_constrained_production_stdout.txt`) match: LP return 186.4, LP cost 31.35, λ* = 1.20; QL return 255.21, cost 95.83; QL-Lag return 180.27, cost 26.88, λ = 1.397.

The chapter tex paragraph at lines 376–390 says "The learned multiplier rises from zero, overshoots to $\lambda \approx 3.2$ before settling at $\lambda \approx 1.40$." The 3.2 overshoot value is NOT in `_stdout.txt` — at episode 20000 the printed λ was 1.381 and at 30000 was 1.397, neither showing a 3.2 spike. The 3.2 value is presumably read off Figure (a) lambda trajectory which is sampled every 500 episodes (60 samples over training), but the printed checkpoints (every 10k) don't show it. This isn't a data integrity bug — the trajectory could plausibly include a spike between sparse print checkpoints, especially if a single dual update with a large positive constraint violation in early training pushed λ high — but it is *unverified from stdout alone*. A hostile reviewer would ask: "Show the full trajectory; the printed values never exceed 1.4."

Cache file is keyed on CONFIG including `version: 12`, so config drift invalidates cache correctly.

Verdict: ◐ data is computed (not hardcoded), but the "$\lambda \approx 3.2$ overshoot" claim in the tex relies on figure inspection only — not corroborated by stdout. Hostile-reviewer concern.

## 4. Comparison Fairness

- Same MDP for all three methods. ✓
- Same horizon $H=100$ for QL training and eval. ✓
- Same eval protocol: deterministic-policy rollouts with `eval_rng = np.random.RandomState(99)`, 5000 episodes, discounted by $\gamma=0.95$. ✓ shared `_eval_det` function used for both QL variants and (via `eval_policy_exact`) for the DP/LP policies.
- DP/LP evaluation uses exact policy evaluation (`np.linalg.solve(I - γP_π, R_π)`), QL uses Monte Carlo. These are different estimators of the same quantity — and the MC eval has truncation bias (horizon 100 vs. infinite). With $\gamma^{100} \approx 0.006$, the bias is bounded by roughly $0.006 \cdot R_{\max}/(1-\gamma) \approx 0.006 \cdot 600 \approx 3.6$, i.e. up to 1–2% of the reported return. This favors no method systematically but the LP-oracle row uses the exact estimator (186.4) whereas the QL rows use truncated MC. ◐ Minor unfairness — the LP value is the exact infinite-horizon return, the QL value is the H=100 truncated return. The 6-unit gap between 186 and 180 is partly this truncation, not just QL suboptimality. Hostile reviewer would catch this.
- LP policy is stochastic (mixes dirty/clean at one state); QL recovers a deterministic policy. The tex acknowledges this (line 388). ✓

Verdict: ◐ comparison is mostly fair, with one defensible-but-not-mentioned truncation bias in the QL evaluators.

## 5. Theoretical Sanity Checks

Predictions and observations:

(i) **Constraint binds at the optimum?** Budget $d = 31.35$ is set to 30% of unconstrained cost (105). Unconstrained cost (96) under the unconstrained policy is way above $d$, so the constraint must bind. LP oracle cost = 31.35 = $d$ to 2 decimal places. ✓ constraint binds.

(ii) **$\lambda^* > 0$ when constraint binds.** LP returns $\lambda^* = 1.20 > 0$. ✓ correct.

(iii) **Lagrangian Q-learning approaches LP oracle.** QL-Lag return = 180.3 vs. LP = 186.4 (gap 3.3%); QL-Lag cost = 26.88 vs. budget = 31.35 (cost is below budget, i.e. policy is slightly too conservative). The final $\lambda = 1.40$ is 17% above the analytical shadow price $\lambda^* = 1.20$. The tex correctly attributes both the conservatism and the $\lambda$ overshoot to naive dual-ascent oscillation (Stooke 2020 PID damping is the standard fix). ✓ direction and magnitude reasonable.

(iv) **Unconstrained policy violates constraint.** Unconstrained cost = 96 vs. budget = 31 (3× over). Tex says "violates the carbon budget by a factor of three" — quantitatively matches. ✓

(v) **Unconstrained QL converges near DP.** DP return = 273, unconstrained QL = 255 (93%). With $\epsilon$-greedy still at 0.05 even at the end, the final policy is sub-optimal by ~6.7%. Plausible for tabular Q-learning at this exploration level. ✓ ballpark.

Verdict: ✓ all sanity checks pass.

## 6. Information Leakage

Q-learning agent sees only $(s, r, c)$ — the constraint cost $c$ at each transition. The dual update uses the known constraint bound $d$ (which is allowed and required: the budget is part of the problem specification). The agent does NOT see the LP oracle's policy, $\lambda^*$, or the unconstrained DP value during training. The LP oracle is computed in parallel for evaluation/comparison only. ✓ clean.

Q-table is initialized at zero; no warm start from the LP solution. ✓

The eval RNG seed (99) is independent of the training RNG (0), so the agent does not get to see its eval rollouts during training. ✓

Verdict: ✓ no information leakage.

## 7. Seed & Reproducibility

This is the central audit problem. **Both QL runs use a single seed (seed=0; lines 371 and 376 in `compute_data`).** CLAUDE.md mandates "Run each method across multiple seeds (minimum 10) and report means and standard errors." The script reports point estimates, not means±SE.

Single-seed reporting is a clear violation of the project's simulation standards. A hostile reviewer would not believe the reported λ trajectory or final return generalize across seeds without seeing the variance. Naive primal-dual dual ascent is notoriously seed-sensitive (the overshoot magnitude in particular).

The script's caching and config-key infrastructure trivially supports multi-seed extension (CONFIG already has the required structure), so this is a fixable defect — but it ships as-is.

Verdict: ✗ N=1 seed, no mean±SE, in a sim where the quantity of interest (the λ trajectory's overshoot magnitude) is known to be high-variance.

## Hostile-Reviewer Summary

The mechanics are correct: the LP oracle is a textbook Altman-style occupation-measure LP, the dual is extracted from HiGHS's marginal correctly, and the Lagrangian Q-learning is an honest tabular implementation of single-timescale primal-dual that matches Tessler 2019 RCPO in form. The environment matches the tex. The sanity checks all align: constraint binds, $\lambda^* > 0$, unconstrained policy violates the budget by 3×, Lagrangian Q-learning approaches the LP optimum and respects the budget. No information leakage, no algorithm-identity fraud (the tex doesn't claim CPO or PID are in the sim, and they aren't).

But three reviewer-bait issues:

1. **Single-seed reporting.** N=1 in a primal-dual sim where the λ-overshoot magnitude is the headline finding. This is a CLAUDE.md violation and the most damaging issue.
2. **The "$\lambda$ overshoots to 3.2" prose claim is figure-only, not stdout-corroborated.** Printed checkpoints (every 10k episodes) max out at 1.40. Reviewer would ask for the full trajectory.
3. **Minor truncation-bias asymmetry**: LP eval is exact infinite-horizon (186.4), QL eval is H=100 truncated MC (180.3). The 6-unit gap is partly truncation, not just suboptimality. Not mentioned.

Issue 1 is the bullshit driver. The substance survives — it would survive a revision, the algorithm is what it claims to be — but reporting N=1 as if it were a published numerical comparison is reviewer-2 catnip.

**Bullshit score: 25%** — Reviewer 2 hammers the single-seed reporting and the unverified $\lambda \approx 3.2$ figure-only claim. The substance (algorithm correct, sanity checks pass, no leakage, no algorithm-identity fraud) survives revision; the headline finding (Lagrangian Q-learning converges near the LP optimum and respects the budget) is real. Fix: re-run with $\geq 10$ seeds and report mean±SE on $\lambda$, return, and cost; verify the overshoot claim against the lambda_trajectory array.
