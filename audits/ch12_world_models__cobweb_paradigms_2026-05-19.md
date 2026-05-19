# Audit: ch12_world_models/sims/cobweb_paradigms.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch12_world_models/tex/s09_dual_sim.tex` (§Cobweb with adjustment cost)
**Cited paper PDFs read (in `papers/`):**
- `ARIFOVIC_1994_GA_COBWEB.pdf` (presence verified; election operator deliberately omitted)
- `MARCET_SARGENT_1989.pdf` (presence verified; RLS adaptive learning)
- `JANNER2019_MBPO.pdf` (presence verified; branched-rollout MBPO)
- `BROCK_HOMMES_1997_RATIONAL_RANDOMNESS.pdf` (presence verified; not directly used — see §1)
- Tests: `tests/test_cobweb_ga_no_param_leak.py` enforces the no-leak claim about Arifovic GA.

Reviewer's framing: this is the cobweb panel of the chapter's dual simulation. The chapter does **not** sell this as "naive vs. rational vs. adaptive in the classical Ezekiel/Muth/Brock-Hommes sense." It sells it as "seven learning paradigms on a single-agent monopoly cobweb with quadratic adjustment cost," explicitly noting (line 9 of `s09_dual_sim.tex`) that this is not the multi-agent expectational cobweb of Marcet-Sargent. That reframing matters for grading: I cannot ding "no REE jump" or "no naive cobweb damped oscillation" if the simulation isn't claiming to produce them. I grade against what the script and tex actually claim.

---

## 1. Algorithm Identity

The audit prompt lists the canonical paradigms as Naive (myopic), Adaptive (linear update), Rational (REE fixed point), RL/learned. The script's actual seven paradigms differ in name and in mathematical content. Comparing the implementations to the tex prose:

**Oracle (`OraclePolicy`).** Solves the LQ-Bellman by fixed-point iteration on the Riccati recursion in `cobweb_env.solve_oracle_lq`, yielding $q_t^\star = K_0 + K_q q_{t-1}$. The Riccati derivation in the docstring matches the algebra for a discounted infinite-horizon LQ problem with one state variable. The `cobweb_env.py` smoke test cross-validates against a 401-point grid Bellman iteration. **Pass.** This is the "rational-with-true-parameters" benchmark; it is not Muth-style REE under uncertainty because there is no agent-side filtering problem. The tex labels it correctly as "knows the true parameters."

**Naive (`NaivePolicy`).** Returns a constant $q_t = 1.4$ every step. This is **not** the classical cobweb "naive expectations" of $p_{t+1}^e = p_t$ followed by myopic supply response. It is a fixed-action no-learning floor. The script's docstring is honest ("True no-learning baseline... chosen as the midpoint of the optimal steady-state actions across the three regimes"). The tex (line 15) is also honest: "The constant-rule baseline plays $q_t = 1.4$ at every step regardless of state." A hostile reviewer can still object that calling it "Naive" in the legend invites confusion with the textbook cobweb naive-expectations agent, but the prose does not make the textbook claim. Minor labeling friction; not a substantive identity error.

**RLS (`RLSPolicy`).** Recursive least squares on $(\hat a, \hat b)$ via the standard information-matrix update $R \leftarrow R + xx^\top$, $\theta \leftarrow \theta + R^{-1}x(p - x^\top\theta)$. Cost parameters $(c, \phi)$ are assumed known. The update rule matches Marcet-Sargent 1989's RLS adaptive-learning template. Replanning each period by re-solving the LQ-Bellman with point estimates is "anticipated-utility / certainty-equivalence" planning in the Cogley-Sargent / Kreps sense. The tex (line 15) correctly says "estimates $(\hat a, \hat b)$ from observed prices, treats $(c, \phi)$ as known, and re-solves the linear-quadratic planner." **Pass.**

**Q-Learning (`QLearningPolicy`).** Tabular Q on a $20 \times 20$ state grid and $25$-point action grid, $\varepsilon$-greedy with linear $\varepsilon$ decay from 0.3 to 0.01, learning rate $\alpha = 0.1$. The TD update $Q \leftarrow Q + \alpha (r + \gamma \max_{a'} Q' - Q)$ is the standard Watkins-Dayan 1992 form. **Pass.**

**Arifovic GA (`ArifovicGAPolicy`).** Population of 30 binary chromosomes (10 bits), fitness-proportional selection on **realized** profit running mean, single-point crossover (prob 0.6), bit-flip mutation (prob 0.0033), 2-elite. **Crucially, the election operator is deliberately omitted** because it requires the true demand/cost parameters to score hypothetical offspring. The tex (line 15) is explicit about this departure from Arifovic 1994. The test file `test_cobweb_ga_no_param_leak.py` enforces this by monkey-patching `expected_reward` to raise on call inside the GA path. This is intellectually honest — the audit explicitly distinguishes "Arifovic GA without election" from the original — and the test is a real guardrail. **Pass with credit.**

**Model-Based LQ (`ParametricLQLearner`).** Estimates $(\hat a, \hat b, \hat c, \hat\phi)$ jointly by least squares, plans via the closed-form LQ-Bellman with point estimates, acts with decaying Gaussian exploration. The reward-side fit regresses $r - pq$ on $(-0.5q^2, -0.5(q-q_{\text{prev}})^2)$, which is the correct decomposition given the reward functional form. The agent is using the *correct functional form*; this is anticipated-utility planning with broader parameter learning than RLS. **Pass.**

**MBPO (`MBPOPolicy`).** Bootstrap ensemble of 5 linear-Gaussian demand models, linear policy $q = K_0 + K_q q_{\text{prev}}$, branched rollouts of horizon 5 from buffer-uniform initial states, REINFORCE updates with moving-average baseline. The score-function term `(a_unclipped - mean) / sigma^2` is the correct Gaussian-policy gradient. This is a simplified MBPO that drops the SAC actor-critic for a one-parameter-pair REINFORCE, but the chapter labels it "MBPO of Janner et al. 2019" without qualifying that the actor is REINFORCE rather than SAC. **A hostile reviewer would flag this**: in Janner 2019, the policy optimizer is SAC, not REINFORCE, and the dropout-style ensemble disagreement weighting is not implemented here either. The tex says "a linear policy trained by REINFORCE" (line 4) and "REINFORCE with a moving-average baseline" (line 15), which is honest about the actor choice. But the name "MBPO" is doing work that the implementation does not. Reviewer 2 would write a comment: "this is not MBPO; this is a linear-Gaussian model-based REINFORCE with bootstrap ensemble."

**Identity verdict.** Six of seven are faithful to their cited papers or honestly relabel known departures (Arifovic without election, Naive as fixed-action). The MBPO name overshoots its implementation: SAC → REINFORCE and the rollout-disagreement weighting is absent. The tex partially admits this in the description but uses the name "MBPO" three times without "-style" or "simplified" qualifiers.

The prompt's expected paradigm list (Naive/Adaptive/Rational/RL) is not what the script implements. The script implements a different taxonomy — oracle, fixed-rule, parametric adaptive, Q-learning, evolutionary, model-based parametric, model-based ensemble + RL — and the tex frames it correctly as an "inductive-bias frontier." So I do not penalize the absence of REE-jump or naive-cobweb-oscillation, because they are not claimed.

## 2. Environment / MDP Fidelity

The cobweb environment (`cobweb_env.CobwebEnv`):
- State $s_t = (q_{t-1}, p_{t-1}) \in \mathbb{R}^2$.
- Action $q_t \in [0, 4]$, clipped on entry.
- Dynamics $p_t = a - b q_t + \varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$.
- Reward $r_t = p_t q_t - (c/2) q_t^2 - (\phi/2)(q_t - q_{t-1})^2$.
- Reset: $q_{\text{prev}} = a / (2(b + c/2))$ (the static-optimum quantity, ignoring adjustment cost), $p_{\text{prev}} = a - b q_{\text{prev}}$.

This is a self-referential monopoly cobweb with adjustment cost. The tex (line 12) matches: "$s_t = (q_{t-1}, p_{t-1}) \in \mathbb{R}^2$ and the action is $q_t \in [0, 4]$. The price clears contemporaneously as $p_t = a - b q_t + \varepsilon_t$..."

**Hostile-reviewer observation.** This is not the *classical Ezekiel cobweb* (producers form $p_{t+1}^e = p_t$, then $q_{t+1}^S = S(p_{t+1}^e)$ clears against demand $D(p_{t+1}) = c - d p_{t+1}$). It is a monopolist with linear inverse demand and quadratic costs choosing quantity. The standard "stability" criterion in the cobweb literature is $|S'(p)/D'(p)| < 1$ or equivalently $|b_{\text{supply}}/d_{\text{demand}}| < 1$. The script's "stability regimes" are parametrized by $b/c$, which controls value-function curvature in the LQ planner, not classical cobweb stability. The tex (line 9) flags this: "varies across regimes to modulate the curvature of the value function rather than to drive the kind of E-stability divergence \citet{MarcetSargent1989} report for the multi-agent expectational cobweb, which does not arise in this single-agent monopoly variant." This is the right disclaimer; without it the reviewer would torch the paper for misusing the word "cobweb stability."

So the environment is *internally* consistent with the LQ Bellman it solves; it is *not* the textbook cobweb the prompt's checklist implicitly references. The script and tex are aligned. **Pass with the caveat that a reader expecting Ezekiel-style supply-and-demand dynamics will be surprised.**

The smoke test in `cobweb_env.py` validates the closed-form Riccati against a 401-point grid Bellman iteration — concrete evidence the oracle is correct.

## 3. Data Integrity

- `compute_data` is split into `compute_shared` (oracle rollouts and reference states per regime) and `compute_paradigm` (each learner across $N_\text{seeds} \times T$ steps). Per-component caching via `compute_or_load` keyed on the per-paradigm config dict. Hyperparameter changes invalidate the right cache; configuration drift would be loud.
- The reported numbers in the stdout file (e.g., RLS at 5.89 in stable, MBPO at 656.60 in stable) trace directly to the `final_mean` / `final_se` fields of the `compute_paradigm` output. No hardcoded values.
- The cache hits in the stdout file mean the results were not re-computed in this run, but the cache write earlier would have come from the same code path. There is no smoking-gun discrepancy.
- The figures and tables are produced by `_plot_*` and `_write_table_*` from the `data` dict only — no leakage of true parameters into outputs (true values appear as `axhline` references in the param-recovery plot, which is correct usage).

**Pass.** I see no fabrication.

## 4. Comparison Fairness

- Each paradigm is reset under the same `regime_params` and seeded identically (`seed=s`) for $s = 0..N-1$. The environment instantiates with `seed=s` so the noise sequence is identical across paradigms within a seed.
- Episode length $T = 500$ for all paradigms.
- `cumulative_regret` uses the **same seed's oracle reward sequence** as the reference (`oracle_rewards_all[s]` vs `res['rewards']`), so the comparison is on identical noise realizations. **Good.**
- Per-paradigm hyperparameters vary, of course, but each paradigm's config is a documented hyperparameter choice rather than a fairness violation.

**One concern:** the warmup mechanism. `ParametricLQLearner` and `MBPO` use a `warmup=5` of uniform random exploration. The other learners (RLS, Q-Learning, GA) do not have an explicit warmup. For seeds where the first five steps happen to score poorly (which they will under uniform random), the model-based learners get a five-step regret head start versus RLS and Q-learning. Inspecting the stdout regret values, this matters most in the stable regime — MBPO with 656.60 regret in stable is bad by every measure, so the warmup is not protecting it. RLS at 5.89 vs Model-Based LQ at 11.65 in stable: a ~6-unit gap is roughly the size of 5 warmup steps of bad rewards, so the head start is non-negligible for the small differences but irrelevant for the order-of-magnitude conclusions. A reviewer would point this out and probably ask for an apples-to-apples 5-step warmup applied to RLS and Q-Learning too. **Minor unfairness; results are robust to it.**

**A second concern:** RLS is told $(c, \phi)$ a priori; Model-Based LQ has to learn them. The tex (line 15) is upfront about this asymmetry: "treats $(c, \phi)$ as known." The conclusion that "RLS wins regret because it has correct functional form and known cost parameters" is therefore *honest*, but a reviewer could argue the panel mixes "amount of structural prior" with "estimation procedure" in a way that confounds the ranking. The tex acknowledges this in line 24 ("recursive least squares paying a higher per-step penalty as long as its cost-parameter prior is correct"), so the framing is defensible but the experimental design intentionally bakes in the asymmetry. Reviewer 2 would still want this in big bold letters.

## 5. Theoretical Sanity Checks

The script's regret numbers (stable / borderline / unstable):
- Oracle: 0 / 0 / 0 — by construction.
- RLS: 5.89 / 4.38 / 5.87 — small, ~5 units, roughly invariant across regimes. Consistent with the theoretical prediction that anticipated-utility planning with the correct functional form should pay an $O(\log T)$ regret penalty.
- Model-Based LQ: 11.65 / 18.96 / 42.90 — rises with $b/c$. The tex (line 18) explains: "reflecting the larger curvature it must resolve in the unstable region and the additional cost coefficients it must learn." Plausible but not theoretically tight.
- Arifovic GA: 92.89 / 133.43 / 308.88 — rises with $b/c$. Population search without functional-form prior pays a much larger regret. Consistent with theory.
- Naive (fixed action 1.4): 179.73 / 3.37 / 450.34 — non-monotone. Because the optimal $q^\star$ varies with regime and 1.4 happens to be close to the borderline regime's optimum. This is the expected behavior of a constant rule and the tex acknowledges it ("a value that depends on how far the fixed action $q_{\text{fixed}} = 1.4$ lies from the regime's optimum"). **Consistent.**
- MBPO: 656.60 / 112.06 / 48.87 — *decreases* with $b/c$. Tex (line 18) explains: "REINFORCE's sensitivity to the curvature of the return surface, which is flat in the stable regime where small differences in $(K_0, K_q)$ matter little for return and sharp in the unstable regime where the policy gradient carries usable signal." This is a real and known issue with REINFORCE under flat reward surfaces; the explanation is theoretically defensible. But the variance is enormous (185.41 SE on the stable mean of 656.60), suggesting that some seeds completely fail to converge. A reviewer might ask whether longer training would close the gap; the current panel does not show it.
- Q-Learning: 953.69 / 841.50 / 991.15 — near-flat large regret. Consistent with the "tabular Q-learning is bad in continuous problems under tight environment budget" sanity check.

**Parameter recovery (final $|\hat\theta - \theta|$).** All three model-based learners recover $(\hat a, \hat b)$ to within ~4% of the truth, and the two estimating $(\hat c, \hat\phi)$ recover them to 0.000 ± 0.000. The tex (line 20, footnote) explains this exact recovery as a consequence of noiseless reward in the simulation: given observed $(p_t, q_t, q_{t-1})$, the cost coefficients are exactly identified from any two distinct $(q^2, (q - q_{\text{prev}})^2)$ tuples. **Mathematically correct and well-flagged.**

**Stable cobweb prediction check.** The audit prompt asks: "Stable cobweb: prices converge to equilibrium. Naive: damped oscillation. Rational: jumps to equilibrium." These predictions are from the textbook *expectational* cobweb. They do not apply here because the script's "Naive" is a constant action, not a naive-expectations agent, and the "Rational" benchmark is the LQ oracle, not a rational-expectations price-jumping solution. So this check does not bind on the script's claims. A reviewer who reads the audit prompt's framing literally would be confused; a reviewer reading the tex would not.

## 6. Information Leakage

The tex makes specific claims about what each paradigm sees:
- Oracle: knows $(a, b, c, \phi)$. ✓ verified — `OraclePolicy.reset` stores `regime_params['a'], ['b'], ['c'], ['phi']` and uses them in `solve_oracle_lq`.
- Naive: state-independent constant. ✓ verified — `NaivePolicy.act` returns `1.4` regardless.
- RLS: estimates $(a, b)$, treats $(c, \phi)$ as known. ✓ verified — `RLSPolicy.reset` stores `self.c = regime_params['c']` and `self.phi = regime_params['phi']`, but `self.theta` is initialized at `[1.0, 0.5]` (not at the true values). The known-cost assumption is intentional and disclosed.
- Q-Learning: only sees $(q_{\text{prev}}, p_{\text{prev}})$ via `_bucket`. ✓ verified.
- Arifovic GA: realized observed profit only, no $(a, b, c, \phi)$. ✓ verified by `_evolve` (uses `self.fitness`, which is the realized-reward running mean) **and** by `tests/test_cobweb_ga_no_param_leak.py`, which monkey-patches `expected_reward` to raise on call. This is the **strongest part of the audit**: a real test that fails loudly if the leak is re-introduced. Credit.
- Model-Based LQ: estimates all four from data. ✓ verified — `_refit` fits $(a, b)$ on $(1, q)$ and then $(c, \phi)$ on $(-0.5q^2, -0.5(q - q_{\text{prev}})^2)$.
- MBPO: estimates from buffer. ✓ verified — `_fit_ensemble` uses only `self.buffer` contents, no `regime_params`.

The reward function is **not stochastic in the simulation** beyond the price noise $\varepsilon_t$. Reward observations are $r_t = p_t q_t - (c/2) q_t^2 - (\phi/2)(q_t - q_{\text{prev}})^2$ where $p_t$ contains noise. After subtracting $p_t q_t$ (observed), the residual is deterministic in $q_t, q_{\text{prev}}$, which is why the model-based learners recover $(c, \phi)$ to machine precision. This is "leakage" only in the sense that noiseless reward overstates achievable parameter recovery — but the tex explicitly footnotes this in line 20 and gives the $O(\sigma_r / \sqrt{N})$ degradation rate. **Honest disclosure of a clean-room assumption.**

**No information leakage in the standard sense.** The model-based learners' machine-precision $(c, \phi)$ recovery is a feature of the noiseless reward, not a coding bug; flagged in tex.

## 7. Seed and Reproducibility

- `np.random.seed(s)` at the top of each per-seed loop in `compute_paradigm`. ✓
- Each paradigm class also seeds its own `np.random.default_rng(seed + offset)` with a distinct offset per paradigm (`+12345` for QL, `+54321` for GA, `+98765` for ParametricLQ, `+13579` for MBPO). The environment uses `seed=s`. Different offsets across paradigms ensure that within a seed, paradigms do not share the same exploration noise — but they do share the same environment noise. **Good practice.**
- $N_{\text{seeds}} = 20$, exceeds the $\geq 10$ floor.
- Standard errors reported as $\sigma / \sqrt{N}$ in tables, stdout, and figure bands. ✓
- The cache files are config-keyed, so reproducibility from cache is deterministic; reproducibility from scratch would require the same `numpy` PRNG behavior (which is stable across modern versions for `default_rng`).

**Pass.**

---

## Hostile-Reviewer Summary

**What is honest and well-done.** The Riccati solve has an independent grid-Bellman cross-check. Per-paradigm caching is config-keyed. The GA's election-operator omission is enforced by a real test that monkey-patches `expected_reward`. The model-based learners' machine-precision $(c, \phi)$ recovery is flagged as a noiseless-reward artifact with the right asymptotic rate. The tex distinguishes "regime curvature $b/c$" from "classical cobweb E-stability" upfront, defusing the most obvious reviewer objection.

**What a hostile reviewer will write.** Two things.
1. **"MBPO" oversells.** The implementation is bootstrap-ensemble + linear-Gaussian dynamics + linear policy + REINFORCE with moving-average baseline. Janner 2019's MBPO uses SAC, neural dynamics, and dropout-based disagreement. The tex says "MBPO with branched rollouts and REINFORCE" three times without "-style" or "simplified-version-of"; a reviewer would ask the chapter to either rename it (e.g., "MBPO-style branched REINFORCE") or implement actual SAC. Given the chapter's stated goal is to locate methods on an inductive-bias frontier, not to benchmark Janner's algorithm, the simplification is intellectually defensible — but the label is doing more work than the implementation supports.
2. **Asymmetric structural priors mix with estimation procedures.** RLS gets $(c, \phi)$ for free; Model-Based LQ has to learn them. The chapter draws the conclusion "RLS wins regret because correct functional form + known cost parameters" honestly, but the experimental design entangles two factors. A reviewer would ask for a fourth panel — RLS without known $(c, \phi)$ — to disentangle.

Both objections are real but neither overturns the chapter's qualitative ordering. The Bullshit Score must reflect "a hostile reviewer would write a snarky comment, but the substance survives revision."

**Bullshit score: 25%** — Reviewer 2 catches the "MBPO" naming overshoot and the asymmetric-prior confound, demands a rename and a fourth panel, but the regret ordering, parameter-recovery numbers, and policy-distance interpretation all survive minor revision. The Arifovic-GA no-leak test is the kind of evidence trail that makes the audit much harder to attack on integrity grounds.
