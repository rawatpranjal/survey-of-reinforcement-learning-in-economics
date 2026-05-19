# Cobweb Paradigms Sim — 7-Point Audit

Audit for `ch12_world_models/sims/cobweb_paradigms.py`. Seven learning paradigms on a self-referential cobweb model with adjustment cost. Updated 2026-05-18 after a stress-test pass: the original "MBPO" row was a closed-form LQ planner and now appears under its honest name (Model-Based LQ); a new MBPO row uses ensemble dynamics + branched rollouts + REINFORCE. The Arifovic GA election operator (which required true demand parameters) was removed.

## 1. Algorithm Identity

The chapter's framing is that single-agent MBRL is the latest member of a family of non-rational-expectations learning paradigms. The sim places seven exemplars side by side on the same MDP.

### Oracle (knows true parameters)

Solves the infinite-horizon discounted LQ-Bellman problem with the analytic Riccati equation. The cobweb model with adjustment cost is linear-quadratic in the state $s_t = (q_{t-1}, p_{t-1})$ and action $q_t$, after substituting the price equation into the reward and treating $\varepsilon_t$ as additive Gaussian shock. The optimal policy is affine: $q_t^\star = K_0 + K_q q_{t-1} + K_p p_{t-1}$, with $(K_0, K_q, K_p)$ recovered from the value-function quadratic form. Reference: Anderson and Moore (1989), linear-quadratic regulation; Ljungqvist and Sargent (2018), Recursive Macroeconomic Theory ch. 5.

### Naive (no learning)

Constant rule: plays a fixed quantity $q_t = 1.4$ across all periods and regimes. The value is a regime-agnostic heuristic midpoint of the three regime-optimal steady-state actions. It is not derivable from observed signals; it is just a "what if the firm guesses a plausible number" floor. The role of this baseline is to show how much the actual learners gain over a fixed, parameter-free heuristic.

### RLS adaptive learning

Marcet and Sargent (1989), the foundational paper. Agent posits a perceived linear price map $p_t = \hat a_t - \hat b_t q_t + \text{noise}$ and updates $(\hat a_t, \hat b_t)$ by recursive least squares over the past $(q_\tau, p_\tau)$ pairs. Each period the agent passes the point estimates into an LQ-Bellman planner and applies the resulting optimal policy. The headline known result is that $\hat\theta_t = (\hat a_t, \hat b_t)^\top$ converges to the rational-expectations fixed point in the *stable* region of the self-referential map and diverges in the *unstable* region. The instability criterion is the local stability of the associated ODE (E-stability of Evans and Honkapohja 2001).

Update rule, with regressor $x_t = (1, -q_t)^\top$ and outcome $p_t$:
$$\hat\theta_{t+1} = \hat\theta_t + R_{t+1}^{-1} x_t (p_t - x_t^\top \hat\theta_t), \quad R_{t+1} = R_t + x_t x_t^\top.$$
Optionally, constant-gain variant $R$ replaced by $1/\gamma_{\text{gain}}$ for tracking under drift.

### Arifovic GA (election operator removed)

Arifovic (1994), the cobweb-specific result. A population of $N_{\text{pop}}$ candidate decision rules, each encoded as an $L$-bit binary chromosome mapping to a production quantity. Selection is fitness-proportional on the chromosome's running mean of realized observed profit. Crossover is single-point with probability $p_c$. Mutation flips each bit with probability $p_m$. The top two chromosomes by realized fitness pass to the next generation intact (elitism); the remaining slots fill with crossover + mutation offspring.

Arifovic's original *election operator* compared each offspring to its parent using expected profit at the most recently observed market state — a comparison that requires the firm to know the demand intercept and slope $(a, b)$ and its own cost coefficients $(c, \phi)$. Because the chapter positions the GA as the "no parametric knowledge" paradigm, the election operator is removed: selection comes only from realized observed profit. Expect a larger regret than Arifovic's published numbers as the price of that honesty.

Default hyperparameters (Arifovic 1994 Table 1): $N_{\text{pop}} = 30$, $L = 10$, $p_c = 0.6$, $p_m = 0.0033$.

### Q-learning (tabular)

Watkins (1989), tabular off-policy temporal-difference control. Discretize state $(q_{t-1}, p_{t-1})$ to a $G_s \times G_s$ grid and action $q_t$ to a $G_a$-point grid covering the same range. Standard $\varepsilon$-greedy with decaying $\varepsilon$ schedule and learning rate $\alpha$:
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)].$$
Default $G_s = 20$, $G_a = 25$, initial $\varepsilon = 0.3$ decaying to $0.01$, $\alpha = 0.1$, $\gamma = 0.95$. This is the model-free reference point.

### Model-Based LQ Learner (closed-form Riccati on point estimates)

This is the model-based learner that exploits the linear-Gaussian structure to skip the branched-rollout machinery entirely. It fits $(\hat a, \hat b)$ by OLS on observed $(q_\tau, p_\tau)$ pairs and $(\hat c, \hat \phi)$ from $r_\tau - p_\tau q_\tau$ regressed on $(q_\tau^2 / 2, (q_\tau - q_{\tau-1})^2 / 2)$. Each step solves the LQ-Bellman with current point estimates via Riccati iteration and acts under the resulting linear policy with decaying Gaussian exploration noise. The name in the table — Model-Based LQ — is honest about the architecture; the previous label "MBPO" was wrong.

### MBPO (Janner et al. 2019): ensemble + branched rollouts + REINFORCE

Real MBPO. Maintains an ensemble of $M = 5$ linear-Gaussian demand models, each fit by bootstrap OLS on the replay buffer; the reward model is shared and fit by OLS on observed rewards. The policy is parameterized as $q = K_0 + K_q q_{t-1}$ with learnable $(K_0, K_q)$. Each step samples $N = 10$ rollouts of horizon $H = 5$ from buffer-uniform initial states under a random ensemble member, accumulates $\gamma$-discounted rollout returns from the learned reward model, and updates $(K_0, K_q)$ by REINFORCE with a moving-average baseline. Acts under the same Gaussian-stochastic policy with decaying exploration noise. This is the real branched-rollout machinery the chapter wants to compare against the closed-form LQ planner.

The role of both model-based rows is the empirical question the chapter notes flag in §9: how does the model-based family — closed-form planner with point estimates vs. REINFORCE-on-ensemble-rollouts — behave alongside RLS (which converges in this single-agent LQ setting) and the population-based GA?

## 2. Environment / MDP Fidelity

Cobweb with adjustment cost per `CHAPTER_NOTES.md` §9:

- **State.** $s_t = (q_{t-1}, p_{t-1}) \in \mathbb{R}^2$.
- **Action.** $q_t \in [q_{\min}, q_{\max}] = [0.0, 4.0]$ (bounded for numerical stability; the bound never binds in the converged regime for the parameter ranges below).
- **Price.** $p_t = a - b q_t + \varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$. Self-referential and contemporaneous.
- **Reward.** $r_t = p_t q_t - \tfrac{c}{2} q_t^2 - \tfrac{\phi}{2}(q_t - q_{t-1})^2$.
- **Discount.** $\gamma = 0.95$.
- **Episode length.** $T = 500$ steps.
- **Noise scale.** $\sigma = 0.1$.

**Stability sweep.** The classical cobweb stability condition (for the deterministic version without adjustment cost) is $b/c < 1$; supply-elasticity exceeding demand-elasticity inverts the map. We sweep:

| Regime    | $a$ | $b$ | $c$  | $\phi$ | $b/c$ | Expected RLS behavior |
| --------- | --- | --- | ---- | ------ | ----- | --------------------- |
| Stable    | 4.0 | 0.5 | 1.0  | 0.2    | 0.5   | converges             |
| Borderline| 4.0 | 1.0 | 1.0  | 0.2    | 1.0   | converges slowly      |
| Unstable  | 4.0 | 2.0 | 1.0  | 0.2    | 2.0   | diverges (Arifovic 1994 result)|

The fixed adjustment-cost coefficient $\phi = 0.2$ is held across regimes so that the cross-regime comparison cleanly isolates the cobweb-stability dimension.

## 3. Data Integrity

The chapter's results table reports a single number per (paradigm, regime) — mean cumulative regret at $T = 500$, plus standard error across seeds. The path from compute to table:

`compute_data()` → per-regime, per-seed, per-paradigm: `rollout_paradigm(env, paradigm, seed)` → `regret_curve` (length $T$ array of cumulative regret) → aggregated `regret_at_T = regret_curve[-1]` → table writer reads `data['results'][algo][regime]['regret_at_T_mean']`, `regret_at_T_se`. No hardcoded numbers in the table or figure writers. Output stdout will print the per-(seed, regime, paradigm) regret-at-T before aggregation, so the table is auditable line-by-line against the cache.

## 4. Comparison Fairness

- All six paradigms see the same `np.random.seed(seed)` for the env noise sequence per episode.
- Each paradigm has $T = 500$ environment steps per seed; no paradigm gets extra interaction budget.
- Q-learning's discrete action grid is sized to match the precision the continuous methods can realize given finite-sample noise; $G_a = 25$ over a $[0, 4]$ range gives action resolution $\Delta q = 0.16$, smaller than $\sigma = 0.1$ implies for the policy noise floor.
- Regret is computed per seed against the oracle's *realized return on the same noise sequence*, not the oracle's expected return. This is the within-seed regret definition, which removes the noise variance from the regret metric.
- All paradigms are evaluated on the cumulative undiscounted return over the episode for the regret calculation, even though the planners optimize the discounted return. The undiscounted convention follows Janner (2019) and is the cleaner economic comparison.

## 5. Theoretical Sanity Checks

Before declaring the sim trustworthy, the following must hold:

1. **Oracle = Riccati fixed point.** The closed-form LQ gain matches a numerical iteration of the Bellman operator on a fine grid to within $10^{-4}$. (Verified in `cobweb_env.py` smoke test: max diff $\approx 4 \times 10^{-5}$ across 5 test states for the stable regime.)
2. **Oracle dominates in expectation.** In every regime, oracle cumulative regret is exactly zero, and every other paradigm's mean regret is positive across the 20 seeds. (Per-seed regret may be slightly negative for high-variance paradigms like Q-Learning, which is acceptable so long as the mean is positive.)
3. **RLS converges across all regimes.** RLS with the correct functional form (known cost structure) is expected to converge in this single-agent monopoly setting, regardless of $b/c$. The Marcet-Sargent divergence-in-unstable-region story applies to the multi-agent expectational cobweb, *not* to the single-agent LQ monopoly with adjustment cost that this sim uses. The chapter's prose must reflect this distinction. RLS regret should be small (single digits) across all three regimes.
4. **Sample-efficiency ordering.** The expected ordering by cumulative regret at $T = 500$, from best to worst, is Oracle, RLS (correct functional form + known costs), Model-Based LQ (learns demand and cost; plans via closed-form Riccati), Arifovic GA (gradient-free, learns no model), Naive (fixed-action floor, regime-dependent), MBPO (ensemble + REINFORCE; comparable to or worse than Model-Based LQ because REINFORCE has higher variance than analytical planning when both apply), Q-Learning (tabular model-free, worst). Naive is regime-dependent and should not be confused with the learning paradigms.
5. **Q-Learning is bounded above by Naive only by accident.** Q-Learning's tabular model-free nature makes it sample-inefficient for continuous control; it is expected to be the worst learner across regimes. Whether Q-Learning beats Naive depends on whether the Naive constant happens to land near the regime's optimal action.

If any of these fail, the sim is not trustworthy and this audit is updated. Deviation from the original chapter notes (the planned "RLS divergence in unstable region" finding) is intentional and documented; see also the §9 prose.

## 6. No Information Leakage

- Oracle agent receives the full parameter vector $(a, b, c, \phi, \sigma, \gamma)$ at construction time; this is by definition.
- Naive, RLS, Q-learning, Arifovic GA, MBPO-style: receive only the action bounds $[q_{\min}, q_{\max}]$, the state dimension, the discount factor $\gamma$, the episode length $T$, and the observed sequence $(q_\tau, p_\tau, r_\tau)$. They do *not* observe $\varepsilon_\tau$ separately, the true coefficients, or the oracle's policy.
- The reward function shape (quadratic in $q$ with adjustment-cost penalty) is *known* to all learners because it is the economic primitive; what is unknown is the price-map parameters $(a, b)$ and the cost parameters $(c, \phi)$. To respect this:
  - RLS estimates $(\hat a, \hat b)$ from price observations and assumes the cost shape is known with $(c, \phi)$ known. This is the most generous version of the RLS paradigm and is the version studied in Marcet-Sargent 1989.
  - Model-Based LQ and MBPO both estimate the price map and the cost coefficients from observed rewards via least squares; neither reads the true parameters.
  - Q-learning sees only observed $(s, a, r, s')$ tuples; it knows nothing parametric.
  - Arifovic GA selects on running mean of realized observed profit; no parametric knowledge. The Arifovic 1994 election operator (which would require $(a, b, c, \phi)$ to score hypothetical offspring) is omitted.

**Cost-recovery caveat.** Both Model-Based LQ and MBPO recover $(\hat c, \hat \phi)$ to machine precision in the stdout summary. This is an artifact of noiseless reward: the equation $r_t = p_t q_t - (c/2) q_t^2 - (\phi/2)(q_t - q_{t-1})^2$ is exactly solvable for $(c, \phi)$ from any two observed $(r, p, q, q_{t-1})$ tuples once they span the cost feature space. With additive reward noise $\sigma_r > 0$, recovery would degrade at rate $O(\sigma_r / \sqrt{N})$.

## 7. Seed and Reproducibility

- $N_{\text{seeds}} = 20$, above the project minimum of 10. Each seed sets `np.random.seed(seed)` at the top of the per-seed rollout.
- The environment uses an internal `np.random.Generator(np.random.PCG64(seed))` so noise draws are independent of any paradigm-internal randomness.
- Paradigm-internal randomness (Q-learning $\varepsilon$-greedy choice, GA mutation, MBPO model-fit batch shuffle) uses a separate `np.random.Generator(np.random.PCG64(seed + 1000))` so that paradigm choices do not perturb the env noise sequence.
- All results reported as mean ± standard error of the mean (SE) across the 20 seeds.

## Hyperparameter Reference

| Item            | Value     | Source / rationale                                |
| --------------- | --------- | ------------------------------------------------- |
| $T$             | 500       | Long enough to see RLS divergence in unstable     |
| $\gamma$        | 0.95      | Standard infinite-horizon proxy                   |
| $\sigma$        | 0.1       | Small relative to action range                    |
| $N_{\text{seeds}}$| 20      | Above project minimum                             |
| RLS gain        | RLS       | Decreasing $1/t$; classical Marcet-Sargent        |
| GA pop          | 30        | Arifovic 1994 Table 1                             |
| GA chromosome   | 10 bits   | Arifovic 1994                                     |
| GA $p_c, p_m$   | 0.6, 0.0033| Arifovic 1994                                    |
| Q-learning grid | 20 × 20 × 25 | Matches noise floor                            |
| Q-learning $\alpha$ | 0.1   | Watkins-Sutton standard                           |
| Q-learning $\varepsilon$| 0.3 → 0.01 | Linear decay                                |
| Model-Based LQ explore $\sigma$ | 0.15 | Gaussian exploration on closed-form planner |
| MBPO ensemble   | 5         | Bootstrap OLS on the replay buffer                |
| MBPO rollout $H$| 5         | Branched-rollout horizon (Janner et al. 2019)     |
| MBPO $N$ rollouts | 10      | Trajectories per policy update                    |
| MBPO policy lr  | 0.005     | REINFORCE step size on $(K_0, K_q)$              |

## Sign-off

This audit was written before the implementation. If the implementation deviates in any item, the audit is updated to match, and the deviation is noted in the chapter prose.
