# Audit: ch06_games/sims/cournot_bertrand_marl.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch06_games/tex/rl_in_games.tex` (§ "Simulation Study: Cournot and Bertrand Duopoly", lines 79-102)
**Cited paper PDFs read:** none — Calvano2020 / Klein2021 / AskerFershtmanPakes are absent from `ch06_games/papers/`; the only MARL primary read for context was Hu2003_nash_q_learning.md (filename only, not contents). The chapter section relies on BowlingVeloso2002 (WoLF-PHC), Hu2003 (Nash-Q), and Tan1993 (IQL), all present.

---

## 1. Algorithm Identity

**IQL (lines 78–110).** Standard tabular Q-learning over own actions only, with Boltzmann selection on Q-values divided by temperature τ. τ initialized at 5.0 and decayed multiplicatively by 0.9995 per step (capped at 0.01). The Q update `Q[i][a_i] += α (r − Q[i][a_i])` is the classic stateless TD update with no discount and no joint action. This is faithful to Tan1993 / classic IQL. No bug.

**Nash-Q (lines 180–225).** Stated as Hu-Wellman 2003 Nash-Q. The Q-table is over joint actions `Q_i(a_0, a_1)`. The stateless backup is `Q[i][a0,a1] += α (r − Q[i][a0,a1])` — for a one-shot repeated game with no transitions, this is fine. However:
- The original Hu-Wellman Nash-Q backs up via `α (r + γ NashV(s') − Q)` where `NashV` is one specific equilibrium (typically the global optimum or a saddle-point) selected by an exogenous rule.
- This implementation's `solve_nash_2player` selects the NE that **maximizes the sum of payoffs** (line 134, 165: `if val > best_value`). That is not in the Hu-Wellman paper — it is an equilibrium-selection rule that biases toward joint-payoff-maximizing (collusive) equilibria. In a pure-NE game like discretized Bertrand this happens to coincide with the unique NE, but in Cournot the discretized grid has *three* pure NE (see point 5 below), and the code's selection rule would pick whichever has the largest summed payoff.
- The Nash policy is mixed with ε-greedy exploration over uniform actions; the original Nash-Q does not specify this. Minor deviation, not fatal.

**WoLF-PHC (lines 231–286).** Bowling-Veloso 2002. Q-update is over own actions only (line 257). Policy hill-climbing toward `argmax Q[i]` with `delta_w=0.002` when winning and `delta_l=0.02` when losing. Average policy `pi_bar` computed from action-count empirical frequency, and winning/losing is judged by `pi @ Q vs pi_bar @ Q`. This matches the spec in §IV of BowlingVeloso2002. However, there is a coding issue: the policy is moved by `±delta` and then projected via `np.clip(...) / sum(...)`. The standard PHC update should move probability mass *from* `a_star` to non-best actions proportionally, or use a softer projection. With the current code, if `pi[i, a]` drops to 0 it gets clipped and the renormalization makes the policy collapse onto a single action quickly. The deterministic-policy collapse explains why WoLF-PHC reports `0.00 ± 0.00` action and exactly `9.0 ± 0.0` profit in Cournot — essentially zero stochasticity by the tail window. Acceptable for Cournot but means the algorithm is no longer fundamentally different from greedy IQL in this experiment.

**Verdict:** All three algorithms are recognizable. Nash-Q's "max-sum NE selection" is a non-standard equilibrium-selection rule that should be flagged in prose (it is not) and conflates Nash-Q with a quasi-correlated-equilibrium variant. IQL and WoLF-PHC are reasonable implementations of their canonical forms.

## 2. Environment / MDP Fidelity

**Cournot setup (lines 34–50).** Inverse demand `P = max(10 − Q, 0)`, marginal cost `c = 1`, integer quantities `{0, …, 9}`. Payoffs computed exactly as `q_i (P − c)`. **Faithful to the standard linear-Cournot textbook model.** The analytical continuous Nash `q* = (a−c)/3 = 3` is correctly computed.

**Bertrand setup (lines 52–72).** Differentiated demand `d_i = max(10 − 2p_i + p_j, 0)`, marginal cost `c = 1`, integer prices `{0, …, 9}`. Payoff matrix correctly built. **BUT the analytical Nash formula on line 69 is wrong:**

```python
self.nash_action = (a + b * c + e * c) / (2 * b - e)
```

Solving the FOC `∂π_i/∂p_i = (a − b p_i + e p_j) + (p_i − c)(−b) = 0` gives the symmetric NE `p* = (a + b c) / (2b − e)`. There is **no `e·c`** term. With `a=10, b=2, e=1, c=1`, the correct continuous NE is `p* = (10 + 2)/(4 − 1) = 4.00`, not `13/3 ≈ 4.333`.

I verified this two ways: (i) numeric BR enumeration on the integer grid: `argmax_p (p−1)(14 − 2p) = 4` for `p_j = 4`; (ii) symbolic FOC. The unique pure-strategy NE on the discretized grid is exactly `(p_0, p_1) = (4, 4)` with profit `18` each. This is also the value the agents converge to.

Consequence: the table column "$|a - a^*|$" reports `0.33` for IQL and Nash-Q in Bertrand, but the agents are actually sitting **on the true Nash**; the gap is an artifact of the wrong reference value. The tex (line 82) propagates the same bug: "the continuous Nash equilibrium is $p^* \approx 4.33$, which discretizes to $p^* = 4$." Both the formula and the cover story are wrong; the continuous NE *is* 4 exactly.

This is a real bug, not a rounding issue. The stated Nash profit `18.89` (line 9 of stdout) is also wrong; true Nash profit is `(4−1)(10 − 8 + 4) = 3·6 = 18` per firm.

## 3. Data Integrity

`compute_data` (lines 424–498) actually trains: each algorithm is called per-seed via `run_experiment`, results are pickled, and `compute_stats` walks the cached histories to derive table numbers. Stdout shows training timings consistent with real Q-learning runs (~17s IQL Cournot, ~100s Nash-Q Cournot for 20 seeds), so the numbers are computed, not hardcoded.

One soft concern: `compute_stats` (line 327) takes `int(round(tail_actions[seed_idx, 0]))` as the "representative" action for profit lookup. This rounds away any mixed-strategy behavior — if seed 1 ended at `mean_action = 3.7`, profit is read off at `(4, 4)` joint cell, not as expectation. For the tight pure-NE convergence here, this is harmless, but it would mask mixed-strategy outcomes if they existed.

## 4. Comparison Fairness

Same demand parameters, same 50,000-iter budget, same 20 seeds per algorithm. Same `alpha0 = 0.5` and same `1 + 0.0001 t` decay schedule across IQL, Nash-Q, WoLF-PHC. Exploration schedules differ by design (Boltzmann vs ε-greedy vs PHC), which is the right kind of asymmetry — each algorithm's exploration is canonical to it. **Apples-to-apples on the inputs.** No leakage of analytical Nash into training (the `nash_action` attribute is used only at evaluation/plotting time, never inside the training loops; verified by grep).

## 5. Theoretical Sanity Checks

**Cournot continuous NE.** `q* = (a−c)/3 = 3.0`, profit `9.0`. Matches code (line 49-50) and the converged values (`IQL 2.95, Nash-Q 2.89, WoLF-PHC 3.00`). ✓

**Cournot discretized NE multiplicity.** I enumerated all pure NE on the `{0,...,9}` grid. There are **three**: `(2,4)`, `(3,3)`, `(4,2)`. The asymmetric ones give one firm profit `12` and the other `6`. The tex (line 82) asserts "Both games have unique Nash equilibria in pure strategies" — **false for Cournot on this discretization.** The symmetric `(3,3)` happens to coincide with the continuous NE, which is why agents converge there, but the uniqueness claim is wrong.

**Joint monopoly (Cournot).** `max_{q0,q1}(q0+q1)(10 − q0 − q1 − 1) = 20.25` at total `Q = 4.5`. On the integer grid: `(q0=0, q1=4)` gives total profit `20`. So the cartel gain over Cournot total `18` is only `2` — and any symmetric cartel requires `q_total = 4 or 5` with asymmetric allocation. **No collusion finding is claimed**, but if one were, this grid is too coarse for meaningful collusion analysis.

**Bertrand continuous NE.** Formula bug (point 2). True `p* = 4`, profit `18`. Agents converge to `(4, 4)` and report distance `0.33` from a fictitious target of `4.33`. **Distance-to-Nash column is misleading.** The agents *did* find the true Nash.

**Bertrand discretized NE.** Verified unique pure NE at `(4,4)`. Joint monopoly is `(5,5)` with profit `20` each (vs Nash `18`). The fact that all three algorithms cluster at `(4,4)` rather than `(5,5)` is consistent with non-collusive Q-learning in a one-shot/stateless game — there is no memory of past prices, no shared state, no trigger strategy. **The result is *expected* (NE under stateless learning); the prose does not claim otherwise, which is correct.**

**Calvano-style framing absent.** The canonical AI-pricing-collusion result requires a state defined by past prices and a memory-dependent strategy; this implementation has no state. So we should *not* expect collusion, and the sim correctly does not find any. But the chapter does not cite Calvano2020 / Klein2021 / AskerFershtmanPakes despite their being in `refs.bib` — a missed connection given how prominent that literature is for "MARL in IO duopoly."

**Convergence iteration is a floor, not a measurement.** The code (lines 339–344) initializes `conv_iter = n_iter` and then for `i in range(len(smoothed))` if smoothness enters a 0.5-band, sets `conv_iter = i + 1000`. With `i=0` after a 1000-window smoothing pass, this floors at `1000`. **Every entry in the table reports `1,000`** — that is the smoothing-window offset, not a convergence iteration. The table column is essentially meaningless. The tex (line 95) builds on this: "All three algorithms converge to Nash in both games within the first 5,000 iterations." The actual data does not support a per-algorithm convergence-speed comparison.

## 6. Information Leakage

`run_iql`: each agent only observes its own reward (line 103). ✓
`run_nash_q`: each agent observes both rewards (Nash-Q requires this per Hu-Wellman). ✓
`run_wolf_phc`: each agent observes only own reward (line 257). ✓

Analytical Nash is computed in `__init__` (lines 49, 69) and stored as `game.nash_action`, but `grep nash_action` in the training functions returns no matches — it is read only by `compute_stats` and `make_figure`. **No leakage of the analytical equilibrium into training.** Good.

## 7. Seed & Reproducibility

20 seeds. Seeded with `np.random.RandomState(42 + s)` per run (line 309). Mean and SE reported. Reproducible.

Standard errors look implausibly small for some entries: `IQL Cournot 2.95 ± 0.05`, `WoLF-PHC Cournot 3.00 ± 0.00`, `IQL Bertrand 4.00 ± 0.00`. The `0.00` SE arises because (i) Boltzmann τ has decayed below 0.01, so the tail-window policy is effectively deterministic, and (ii) all 20 seeds lock onto the same integer action with no variance. Not a fraud signal — it's a consequence of using a coarse integer action space plus aggressive exploration decay; the population of policies has collapsed. But it does mean the SE column is uninformative and the "convergence" is more a statement about the discretization than the learner.

---

## Hostile-Reviewer Summary

The Bertrand Nash formula is wrong (extraneous `e·c` term added to numerator); the true continuous and integer-grid NE is `p* = 4`, not `4.33`. The agents converge to the *true* Nash; the table's "distance from Nash" column of `0.33` measures distance from a fictitious target. The tex propagates this with "continuous Nash equilibrium is `p* ≈ 4.33`, which discretizes to `p* = 4`," which sounds like a discretization story but is actually a propagated algebra mistake. The reported Bertrand Nash profit (`18.89`) is similarly off; the correct value is `18`.

The chapter also claims pure-strategy NE uniqueness in *both* games. Cournot on the integer grid has *three* pure NE (`(2,4), (3,3), (4,2)`); only the symmetric one coincides with the continuous formula. The asymmetric NE are not pathological — they are an artifact of the coarse discretization the author chose. The "uniqueness" claim is wrong as stated.

The "Conv. iter" column reports `1,000` for every algorithm in every game, because the convergence-detection code floors at the smoothing-window length. The column is not a measurement; it is a constant. Reviewer 2 will flag this immediately.

The Nash-Q implementation silently picks the joint-payoff-maximizing equilibrium when multiple NE exist (lines 134, 165), which is a non-standard equilibrium-selection rule. In Cournot this matters because there are three NE. The selection rule is not mentioned in tex or in inline comments — a reader assumes canonical Hu-Wellman.

Calvano2020 / Klein2021 / AskerFershtmanPakes2021 — the canonical Q-learning-IO-pricing literature — are absent from `papers/`, and Calvano is in `refs.bib` but uncited in this section. The sim does not claim collusion (correctly, given the stateless design), but the framing "MARL in IO duopoly" without engaging that literature is a hostile-reviewer red flag for a chapter on RL in games.

What survives: the Cournot continuous-NE comparison is fine, the IQL/WoLF-PHC implementations are recognizable, no analytical-equilibrium leakage into training, seeds are real. The qualitative point — "stateless MARL converges to one-shot Nash in well-posed games" — is supported by the data. But the specific Bertrand number is the kind of error a reader catches on first read of the equations.

**Bullshit score: 50%** — The Bertrand Nash formula bug (with the wrong reference value embedded in both the code and the prose), the false uniqueness claim for Cournot, the `Conv iter = 1000` constant masquerading as a measurement, and the silent Nash-Q equilibrium-selection deviation combine to a "the methods do not say what they appear to say" situation. The hostile reviewer concludes the comparison is muddied by basic algebra mistakes that should have been caught by inspecting any of the numerical outputs. Major revise: fix the Bertrand formula, recompute the distance column, fix the Cournot uniqueness claim, fix or remove the convergence-iteration column, and either follow standard Hu-Wellman Nash-Q or label the variant.
