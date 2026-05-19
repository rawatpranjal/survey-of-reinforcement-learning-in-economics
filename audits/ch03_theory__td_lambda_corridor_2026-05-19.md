# Audit: ch03_theory/sims/td_lambda_corridor.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch03_theory/tex/planning_learning_v3.tex` (subsection "Simulation Study: Credit Assignment in a Corridor", lines 138-152; theory section "sec:td_lambda" lines 113-136)
**Cited paper PDFs read:** none directly opened during this audit; relevant references in `ch03_theory/papers/`: `tsitsiklis1997_td_learning_function_approximation.pdf` (linear TD(λ) convergence — cited in the surrounding theory section), `bhandari2021_td_linear.pdf` (finite-time TD(0)), `mitra2024_finite_time_td_learning.pdf` (finite-time TD). No Sutton 1988 or Singh–Sutton 1996 PDFs are present in `papers/`; the classical bias-variance / random-walk source is uncited at the file level. Sutton & Barto 2nd ed. is available as `RLCOURSECOMPLETE 2ndEDITION.pdf` — not opened in this pass.

---

## 1. Algorithm Identity

The update inside `run_td_lambda` is the textbook backward-view, **accumulating-trace** TD(λ):

```
e[s] += 1.0
V[:n_states - 1] += alpha * delta * e[:n_states - 1]
e *= gamma * lam
```

with TD error δ_t = r + γ V(s_{t+1}) − V(s_t). This matches Equation (eligibility_trace) in the tex (line 130):
`e_t(s) = γλ e_{t-1}(s) + 1{s = S_t}`, `V(s) ← V(s) + α δ_t e_t(s)`.

Two minor points:
- The trace increment happens *before* the value update, and the decay happens *after*, which makes the trace update look like `e_t = γλ e_{t-1} + 1{s=S_t}` in standard form (the +1 is applied to the previous-step decayed trace at the current state). That is correct.
- Trace decay multiplies the *full* trace vector (including the terminal index), but `V[:n_states - 1]` is what gets updated — terminal value is fixed at 0. This is fine; the terminal trace component never feeds into a target since the episode ends.
- Replacing traces are *not* implemented (the tex footnote line 134 mentions replacing/Dutch traces but the implementation choice "accumulating" is not stated in the tex). Minor disclosure gap.

**Verdict:** algorithm identity is correct (accumulating-trace backward TD(λ)). The tex does not name the trace variant; for full rigor it should say "accumulating traces."

---

## 2. Environment / MDP Fidelity

The tex (line 141) says: "A 20-state deterministic corridor ($s \in \{0, \ldots, 19\}$, action: move right, reward $+1$ only at the terminal state $s = 19$, $\gamma = 0.99$) isolates the credit-assignment mechanism. The true value function is $V^*(s) = \gamma^{19-s}$."

Code matches the *MDP description*: 20 states, deterministic right-walk, reward 1 only at the 18→19 transition, γ=0.99, terminal at s=19.

**Bug: the closed-form $V^*(s) = γ^{19-s}$ is wrong by one power of γ.** Deriving from the implemented Bellman recursion:

- The code sets `r = 1` when `s_next == n_states - 1` (the 18→19 transition); `V_next = 0` at terminal.
- Bellman: V(s) = r(s, s+1) + γ V(s+1), with V(19) = 0.
- V(18) = 1 + γ·0 = 1
- V(17) = 0 + γ·1 = γ
- V(s) = γ^(18−s) for s ∈ {0,...,18}

The script (`true_values`, line 50–52) returns `γ^(n_states - 1 - s) = γ^(19−s)`, i.e. one extra factor of γ at every state. The tex carries the same off-by-one error in its `V^*(s) = γ^{19-s}` claim.

Quantitative consequence: the spurious offset is `γ^(19−s) − γ^(18−s) = γ^(18−s)(γ − 1) = −0.01·γ^(18−s)`. The RMS of this offset over s ∈ {0,...,18} is

`0.01 · sqrt(sum_{k=0}^{18} γ^(2k) / 19) ≈ 0.01 · sqrt(0.838) ≈ 0.0092`.

The reported "final RMSVE" for λ=1.0 is **0.0091 ± 0.0000** — i.e. essentially the bias floor created by the wrong reference, not the actual approximation error. For an exactly-on-policy deterministic MC estimator with 200 episodes, the true RMSVE against the actual fixed point should be much closer to machine precision; the 0.0091 number is the off-by-one artefact almost to the third decimal.

This is a real bug. It does not change the qualitative ordering (λ=1 still reaches RMSVE < 0.05 fastest), but it makes the numbers in the table *misleading*: a reader sees "MC reaches RMSVE 0.0091" and infers near-convergence, when it is in fact the algorithm correctly converging to a *wrong target*.

---

## 3. Data Integrity

- `compute_data` is wired through `compute_or_load` against `CONFIG` (with a `version` key bumped to 3), so cache invalidation on config change is correct.
- The reported `final_mean`, `final_se`, and `eps_to_thresh_*` are computed from `all_rmsve` (one row per seed), with proper per-seed bookkeeping.
- Stdout shows a cache hit on the most recent run; no evidence of stale or hardcoded values being printed.
- Numbers in the rendered `td_lambda_corridor.tex` match what `_run_td_lambda_experiment` would produce given the (buggy) `true_values`.

Data pipeline is internally consistent. The bug in Section 2 propagates faithfully through every reported number.

---

## 4. Comparison Fairness

- Same seeds (`42 + seed_idx` for `seed_idx ∈ {0,...,19}`) used across all λ values.
- Same α=0.05 for all λ. **No per-λ step-size tuning is disclosed or performed.** The classical bias-variance demo (Sutton & Barto Fig 7.6) tunes α per λ and reports a U-shape over a grid; here only one α is used. Given the deterministic environment this isn't fatal (variance is zero anyway), but the surrounding theory section in `planning_learning_v3.tex` invokes the bias-variance language (line 127), and the simulation does not demonstrate it — it only demonstrates credit-propagation speed. The reviewer-charitable read is that the section title is "Credit Assignment in a Corridor," not "Bias-Variance," so this is consistent. The less charitable read is that the surrounding prose primes the reader for a bias-variance experiment that the sim cannot deliver.
- Same number of episodes (200) per λ. Fair.

Apples-to-apples on what is actually being compared (credit assignment under fixed α). Not what a reader who arrives via the bias-variance theory paragraph would expect, but defensible.

---

## 5. Theoretical Sanity Checks

The audit prompt cites the standard test: RMS error vs λ should be U-shaped (Sutton-Barto Ex 7.10 / Fig 7.6) with minimum at intermediate λ. **That experiment is on a stochastic random walk**; this environment is deterministic, so λ=1 (MC) has zero variance and the U-shape collapses to a monotone improvement in λ. The figure correctly shows monotone improvement.

Other sanity checks:

- **MC (λ=1) on deterministic MDP**: every episode produces the *same* return path, so MC value estimate should converge to the true fixed point in *one* episode per state. The figure shows λ=1 drops sharply within the first 50 episodes, consistent with this. The final value plateaus at 0.0091 — the bias floor from Section 2, **not** convergence to V*.
- **TD(0) on a 20-state chain with α=0.05**: one-step bootstrapping needs O(n_states / α) = O(20/0.05) = O(400) episodes to back-propagate fully; the table reports λ=0 not crossing the 0.05 threshold in 200 episodes, which is qualitatively correct.
- **Closed-form match**: the tex claim `V^*(s) = γ^{19-s}` does *not* match the MDP as implemented (Section 2). The correct closed-form is `γ^{18-s}`.
- **Linear-TD(λ) bound** (line 136 of tex): `(1 - λγ)/sqrt(1-γ²)` times best-in-class. With tabular features (best-in-class error = 0), all λ should converge to V* exactly. The reported λ=1 floor of 0.0091 is not the algorithm failing this bound; it is the *reference* being wrong.

Theory passes for what is actually being measured, but the headline numbers do not represent convergence to the actual V*.

---

## 6. Information Leakage

- The agent does not see V_true during learning; `V_true` is only used post-hoc to compute RMSVE. Clean.
- The "policy" is degenerate (one action always available, move right); the tex calls it a "deterministic corridor" with "action: move right." There is no separate behaviour vs target policy issue because there is no choice. On-policy by construction.
- Eligibility traces are updated on the realised trajectory only, with no peeking at future states beyond the standard one-step bootstrap. Clean.
- The terminal value V[n_states - 1] is hard-pinned to 0 at the start of each episode (line 60). This is the standard convention for absorbing states and is not leakage.

No leakage.

---

## 7. Seed & Reproducibility

- Random seeds fixed at `42 + seed_idx` for `seed_idx ∈ {0,...,19}` → 20 seeds (meets the ≥10 minimum).
- The only stochastic input is the initial random value vector `V = rng.uniform(0, 0.5, n_states)`; the trajectory itself is deterministic, so seeds vary only the initialization. This is unusual but documented in code.
- Mean and SE reported in both the figure (shaded SE band) and table (`±` notation). The SE values for λ=0.8 and λ=1.0 are vanishingly small (0.0001, 0.0000) because the MDP is deterministic and the only variation is initialization, which decays out within ~10 episodes for high λ. **The reported `48 ± 0` for "Episodes to threshold" at λ=1.0 means every seed reached threshold at episode 48** — i.e. the standard error is genuinely zero on this metric, which is a tell that the experiment has almost no real stochasticity left to average over.

Reproducible. The "low variance" observation is a feature of the deterministic environment, not a seed-handling bug — but it is also why this sim cannot illustrate the bias-variance tradeoff the surrounding theory paragraph invokes.

---

## Hostile-Reviewer Summary

The sim is a clean accumulating-trace TD(λ) on a clean deterministic chain, and the algorithm is implemented faithfully. The qualitative story (higher λ propagates sparse rewards backward faster) is correct and matches the figure.

But the closed-form $V^*(s) = γ^{19-s}$ stated in both the script and the tex is **wrong by one power of γ** relative to the MDP as defined (correct form: $γ^{18-s}$). This is a transparent algebra error in a textbook setup. Every reported RMSVE number sits on top of a `~0.009` bias floor created by comparing against the wrong target, and the table's flagship number (λ=1 → final RMSVE = 0.0091) is *exactly* that floor. A reviewer who derives the Bellman recursion by hand — which the chapter audience definitely will — will catch this in under a minute and conclude either (a) the authors did not verify their closed form or (b) the sim has a subtle off-by-one in reward timing. Either reading is reputationally bad in a *theory* chapter.

Adjacent weaker points: (i) the deterministic environment cannot demonstrate the bias-variance tradeoff that the theory paragraph immediately above invokes, so the sim under-delivers on its own framing (the section title "Credit Assignment" is honest but the reader arrives primed for variance); (ii) trace variant (accumulating vs replacing) is undocumented in the tex; (iii) the SE = 0 entries in the table are a tell that the only stochasticity is the initial value vector, which a sharp reviewer will notice.

The closed-form error is genuinely material because it lands in a chapter titled "The Theory of Reinforcement Learning." In a methods chapter, a reviewer might wave it off; in a theory chapter, getting the closed-form wrong for the worked example is the kind of thing that prompts "did the authors check their algebra?" The fix is one line in `true_values` plus one symbol in the tex.

**Bullshit score: 50%** — The closed-form $V^*(s) = γ^{19-s}$ is off by one factor of γ; the headline RMSVE numbers are the resulting bias floor, not convergence. A theory-chapter reviewer will derive V* by hand, catch it immediately, and argue the result is suspect because the reference is wrong. Major-revise: fix `true_values` to `γ^(18-s)`, update the tex statement, re-run, and re-verify the table. The qualitative ordering survives, but the quantitative numbers as printed do not.
