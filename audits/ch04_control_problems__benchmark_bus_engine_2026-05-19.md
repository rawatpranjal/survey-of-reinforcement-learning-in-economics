# Audit: ch04_control_problems/sims/benchmark_bus_engine.py

**Date:** 2026-05-19
**Diagram-only:** no (full Monte Carlo benchmark, 6 fleet sizes x 3 seeds, VI + DQN + 2 heuristics)
**Cited tex file(s):** `ch04_control_problems/tex/applications.tex` (Section "Simulation Study: Bus Engine Replacement", lines 242-258; Figure `fig:bus_engine_scaling`)
**Cited paper PDFs read:** `ch04_control_problems/papers/rust1987_bus.md` (markdown digest of Rust 1987 Econometrica). No PDF of Rust 1987 itself in `papers/` — only the digest.
**Output artifacts present:** `bus_engine_scaling.png` (239 KB), `bus_engine_results.tex` (587 B), `benchmark_bus_engine_stdout.txt`, cache pickle.

---

## 1. Algorithm Identity

Two methods compared: tabular value iteration (VI) on the Bellman optimality operator, and vanilla DQN (Mnih et al. 2015 style).

**VI** (`run_value_iteration`, econ_benchmark.py:165-221). Standard synchronous Bellman backup:
`V_new[s] = max_a [r(s,a) + gamma * sum_{s'} P(s'|s,a) V[s']]`. Uses `env.expected_reward` and `env.transition_distribution`. Tol=1e-8, max_iter=1000. Identity is clean — this is textbook VI.

**DQN** (`run_dqn`, econ_benchmark.py:228-347). Target network refresh every 50 episodes, MSE Bellman-error loss, epsilon-greedy with linear decay (start 1.0, end 0.01, decay_frac 0.5-0.6 of training), Adam optimizer, replay buffer. Loss: `(Q(s,a) - (r + gamma * max_a' Q_target(s',a')))^2`. This is vanilla DQN; not Double-DQN, not Rainbow, not Dueling. No PER. Identity matches what the tex calls "DQN" — clean.

**Heuristics:** threshold (replace engines with mileage >= 3, up to capacity) and never-replace. Implementations match descriptions.

No misnaming. Algorithm identity check: clean.

---

## 2. Environment / MDP Fidelity

The implemented MDP departs from Rust 1987 in several ways. The tex footnote (line 247) partially acknowledges this. Compare against the Rust1987 digest:

| Component | Rust 1987 | This sim | Documented in tex? |
|---|---|---|---|
| Mileage bins | 90 | 6 | yes (footnote, line 247) |
| Transitions | Stochastic, $\xi \in \{0,1,2,3\}$ with estimated probs | Deterministic +1 (or 0 if replaced) | yes (footnote) |
| Cost | $\theta_{11} x + RC \cdot a$ | $\alpha \sum_i m_i + \beta |a|$ | yes (footnote — fleet ext.) |
| Logit shock | Type-I EV per action | None | NOT mentioned |
| Discount | $\beta = 0.9999$ monthly | $\gamma = 0.95$ | NOT mentioned |
| Horizon | Infinite | Train 50, eval 100 (finite, discounted) | NOT mentioned |
| Fleet size | 1 (per-bus, independent) | 1..6 with capacity constraint | yes (new extension) |
| Action space | {keep, replace} | Subsets of engines, |a| <= 3 | yes |

The deterministic-transitions choice strips the *random mileage increments* that are core to Rust's setup — but the tex footnote explicitly says this is intentional "to isolate the combinatorial scaling challenge." Acceptable as an extension if the prose doesn't claim to replicate Rust's empirical findings (it doesn't).

The missing logit shock matters less for an RL-vs-DP comparison than for an econometric estimation pass, so it's fine to drop.

The discount-factor drop from 0.9999 to 0.95 is a real change in the optimization landscape (effective horizon ~20 periods vs ~10,000 periods); the tex does not mention this. Hostile reviewer point but not a fatal one.

The N=5 footnote claim "$M = 6$ mileage bins, the state space is $6^N$: 1,296 states at $N = 4$, 7,776 at $N = 5$, 46,656 at $N = 6$. Value iteration is feasible for $N \leq 5$." matches the script's `MILEAGE_STATES = 6` and the observed state counts in stdout.

Fidelity: defensible as an extension, with one undocumented parameter swap (discount factor).

---

## 3. Data Integrity

`compute_data()` invokes `run_scaling_sweep()` which calls `run_single_complexity(N, SEEDS)` for each N. Each call runs VI fresh, then trains DQN with three seeds, then evaluates the two heuristics. Numbers in the LaTeX table and stdout are pulled from the `results` dict populated by these calls — no hardcoded values.

The stdout shows fresh VI runs at each N with reported iterations, residuals, and wall times that scale plausibly: 382 → 396 → 404 → 409 → 414 iter (small growth because horizon is dominated by discount), wall time 0.06 → 0.06 → 0.59 → 6.93 → 81.79s (roughly |S|^2 scaling because backup is O(|S| * |A|) per iter * |S| sweep). DQN wall times also grow with N as expected (replay buffer, hidden width, episodes all scale up).

The cache pickle exists; the CONFIG dict includes a `version: 1` field, so config drift would invalidate.

Data integrity: clean.

---

## 4. Comparison Fairness

Several issues, ranked by severity:

(a) **Different `env.reset()` initial states across methods.** `evaluate_dp_policy`, `evaluate_dqn_policy`, and `evaluate_heuristic` are called in sequence on the same `env` instance; each calls `env.reset()` which draws a fresh random initial mileage tuple. So DP, DQN, and the two heuristics are evaluated on *different* episode sequences. With 200 episodes per evaluation and a deterministic transition kernel, the noise is small (stdout DQN std is 0.07-1.09 absolute on returns of -55 to -331), but the comparison is not on identical trajectories. A clean fix would pre-sample initial states once and replay them through each policy. Hostile reviewer would flag this.

(b) **DQN trained at horizon 50, evaluated at horizon 100.** `episode_horizon=TRAIN_HORIZON=50` for training (line 245), but `evaluate_dqn_policy(env, q_net, ..., horizon=EVAL_HORIZON=100, discount=GAMMA)` evaluates at 100. With $\gamma=0.95$, the tail past step 50 contributes weight $\gamma^{50}/(1-\gamma) \approx 1.6$ relative to the per-step cost ~5, so the discounted-tail bias is small but nonzero. VI computes the infinite-horizon $V$, but is evaluated at horizon 100 too — so the apples-to-apples comparison is between two finite-horizon truncations of policies trained for two different effective horizons. Reviewer-2 nit.

(c) **DQN evaluated with greedy argmax on a network trained against a noisy MSE target.** Standard practice — fine.

(d) **Three seeds.** CLAUDE.md "Simulation Standards" says "Run each method across multiple seeds (minimum 10) and report means and standard errors." This sim runs 3 seeds. DP is deterministic so 1 seed suffices, but DQN should have N>=10. The reported `DQN Return -55.10 +/- 0.10` is std-dev over n=3 — the standard error is half that, but it's reported as std not SE. Reviewer would catch this.

(e) **Heuristics use single deterministic policy.** Fine; no seed dependence.

(f) **Compute budget asymmetry.** DP at N=5 takes 82s; DQN at N=5 takes ~320s per seed (~960s total). DQN gets ~12x the compute. For an "RL matches DP" claim, this works because the claim is that RL *can* match DP given enough compute, not that it does so efficiently. But the left-panel "computation time" figure plots wall-clock and shows DQN time growing more slowly than DP's at large N (the scaling story). The plot does not normalize for hyperparameter-search cost or for the fact that DP's wall time grows because of `|S|^2 |A|` sweeps in plain Python, not because VI is fundamentally costly — a vectorized VI would be 10-100x faster. Reviewer 2 would push back on the absolute time comparison.

Comparison fairness: meaningful flaws (3-seed report and trajectory mismatch are the clearest), but not falsifying.

---

## 5. Theoretical Sanity Checks

- VI converged at all N with final residuals all ~1e-8 (below tol). Good.
- DQN tracks DP within 0.0% - 0.4% on discounted return across N=1..5. This is the correct theoretical prediction (DQN should approach $V^*$ as compute grows, modulo function-approximation error).
- The threshold(3) heuristic loses 1-1.5% to DP — sensible because mileage 3 is roughly the right threshold but ignores capacity timing.
- The never-replace heuristic loses ~70% — sensible because cumulative mileage cost blows up.
- Q-error and policy-agreement: agreement falls from 100% (N=1) to 86.7% (N=5) but reward gap stays under 0.5%. This is consistent with the well-known phenomenon that *policy-relevant* states form a small subset of the joint state space; the DQN can disagree with DP on rarely-visited states without losing return. Good — sanity is intact.
- One small worry: at N=1 the Q-error is *higher* than at N=4 (2.569 vs 0.212). This is suspicious if it were the same scale, but at N=1 returns are -55 and Q is per-state so absolute Q-error of 2.5 on Q magnitudes of order 55 is ~4.5%; at N=4 Q-error of 0.21 on Q magnitudes of -220 is ~0.1%. So in *relative* terms, DQN actually fits N=4 *better*. Not flagged as wrong, but the Q-error table without normalization to magnitude is mildly misleading. Not in the tex, only stdout — non-fatal.

Sanity: clean. The headline claim "DQN matches DP within 1%" is supported by the stdout numbers.

---

## 6. Information Leakage

- VI is fully model-based (uses `env.transition_distribution` and `env.expected_reward`) — this is by design and not "cheating" — it's the oracle.
- DQN is fully model-free: in `run_dqn`, the only env interaction is `env.reset()`, `env.step(a)`, and `env.state_to_features(s)`. No call to `transition_distribution` or `expected_reward`. The replay buffer stores only (s, a, r, s', done) tuples. Clean.
- DQN evaluation calls `evaluate_dqn_policy` which only uses `env.reset/step/state_to_features` — no peeking.
- Heuristic evaluation: same — clean.
- Initial state is randomized by `env.reset()`, not biased toward any DP-favored region.

Information leakage: clean.

---

## 7. Seed & Reproducibility

- Seeds set at top of __main__: `np.random.seed(42)`, `random.seed(42)`, `torch.manual_seed(42)`. Inside `run_dqn`, seeds are re-set per-seed (`np.random.seed(seed)` etc.). Good.
- Three DQN seeds (42, 123, 7). CLAUDE.md requires >=10. **Violation.**
- The "+/-" in the LaTeX table is std (`np.std(dqn_rewards)`), not SE. With n=3, the std/sqrt(3) correction matters. Tex caption and prose do not specify which. Reviewer 2 catches this.
- VI is deterministic so 1 "seed" is fine for it.
- Cache invalidation tied to CONFIG hash — version field present. Reproducible across runs.
- Note: VI uses `env.expected_reward` but the running evaluation uses `env.step` which calls `np.random.randint` inside `reset` for the initial state. So the evaluation seeds are not fully controlled across the three calls (DP eval, DQN eval, heuristic eval) — they share the same global RNG stream that is being chewed through. Hostile reviewer flag.

Seed and reproducibility: 3 seeds (below the in-house minimum of 10) and std-vs-SE ambiguity are the substantive issues.

---

## Hostile-Reviewer Summary

The sim is technically clean in the places that matter most: VI is the correct oracle, DQN is vanilla-DQN-as-named, no information leakage, no hardcoded results, the headline "DQN matches DP within ~1%" is supported by the numbers. The MDP departs from Rust 1987 in documented ways (deterministic transitions, 6 bins) plus one undocumented way (discount factor 0.95 vs Rust's 0.9999).

The two things a real reviewer flags hardest:

1. **Three DQN seeds.** The project's own standards require ten. With n=3 the "+/- 0.1" error bars are not informative and the reported quantity is std not SE.
2. **Initial-state mismatch across methods.** DP, DQN, and heuristics are not evaluated on the same trajectories — they share an env whose `reset` keeps drawing random states. With 200 episodes the noise is small, but a careful reviewer wants paired evaluation.

Neither falsifies the main claim. The substance survives revision.

**Bullshit score: 25%** — Reviewer 2 catches the 3-seed report, the std-vs-SE label, and the undocumented discount-factor change from Rust's 0.9999 to 0.95; writes a snarky comment about the "matches DP within 1%" claim being computed against an extension that strips Rust's stochastic dynamics. The substance (DQN tracks the DP oracle on a combinatorial fleet problem) survives.
