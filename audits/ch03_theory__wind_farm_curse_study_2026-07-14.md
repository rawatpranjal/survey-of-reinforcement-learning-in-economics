# Simulation Audit: Wind Farm Curse-of-Dimensionality Study

**Date:** 2026-07-14
**Auditor:** hostile-reviewer pass (7-point checklist, `/Users/pranjal/Code/rl/CLAUDE.md`)
**Artifacts:**
- Script: `/Users/pranjal/Code/rl-wind-farm/ch03_theory/sims/wind_farm_curse_study.py`
- Stdout: `/Users/pranjal/Code/rl-wind-farm/ch03_theory/sims/wind_farm_curse_study_stdout.txt`
- Table: `/Users/pranjal/Code/rl-wind-farm/ch03_theory/sims/wind_farm_curse_study_results.tex`
- Figure: `/Users/pranjal/Code/rl-wind-farm/ch03_theory/sims/wind_farm_curse_study_times.png`
- Prose: `/Users/pranjal/Code/rl-wind-farm/ch03_theory/tex/curse_of_dimensionality.tex` (Section 3.3, lines 123-153)
- Cache: 16 pickles in `/Users/pranjal/Code/rl-wind-farm/ch03_theory/sims/cache/`
- Reference: `.../papers/curse_of_dimensionality/lu2025_approximate_factorization.md` (+ du2021, liu2022)

Verification method: loaded all 16 cache components via the script's own `load_component` with the
script's current configs, recomputed every table/stdout cell, cross-read the tex against code line by
line, and confirmed the environment against the Lu 2025 source text.

---

## 1. Algorithm Identity Check — PASS

The tex never claims these are the cited algorithms; it repeatedly and explicitly labels them
"illustrative analogies to the three pathways ... not implementations of the cited algorithms"
(line 137). This disclaimer is load-bearing and neutralizes the usual identity risk.

- **Factored RL** (`FactoredQL`): one Q-table per state dimension, summed additively for `Q(s,a)`.
  This is a legitimate factored-value representation. The reward-relevant dims (first 3) receive the
  TD error at weight `1/reward_dims`; auxiliary dims get zero weight and stay at zero. The tex
  discloses that it "is given the identity of the three reward-relevant dimensions in advance."
- **DQN** (`DQNAgent`): standard DQN, experience replay + target network, `target = r + gamma *
  max_a target_net(s')`. Correct single-network target (not double DQN, none claimed). Legit.
- **Linear AC** (`LinearAC`): actor-critic, linear V and Gaussian policy mean in polynomial features
  (1 + 2d + d(d-1)/2). TD critic, score-function actor update. Legit actor-critic pathway.
  Minor nit: the score function uses `action_raw` (the raw normalized Gaussian sample) while the
  environment executes a scaled/clipped/discretized action, a slight self-inconsistency, but harmless
  for an explicitly illustrative method.

No placeholder stubs, no always-zero penalties, no argmax-masquerading-as-expectile. The methods are
what they are named (generic RL baselines), and the tex names them as such.

## 2. Environment / MDP Fidelity — PASS

Every transition, the reward, demand, bounds, horizon, and discount in the tex (lines 126-137) match
the code (`ExtendedWindFarmEnv.step`, and the DP's `_sample_next_state`/`_compute_reward` mirror it
exactly):

| Quantity | tex | code | match |
|---|---|---|---|
| `w_{t+1}` | 0.7 w + N(30, var 25) | `0.7*w + normal(30, std 5)` | yes |
| `p_{t+1}` | 0.6 p + 0.05(w/100) + N(0.4, var 0.01) | `0.6*p + 0.05*(w/100) + normal(0.4, std 0.1)` | yes |
| `c_{t+1}` | c + 0.9 a | `c + 0.9*a` | yes |
| `x^(i)_{t+1}` | 0.8 x + 0.01(w/100 - 0.5) + 0.1 + N(0, var 0.0025) | same, `normal(0, std 0.05)` | yes |
| reward | `p·min(w+a,D) - 0.01c - 5·max(0,D-w-a)` | identical | yes |
| demand | Poisson(50 + 10 sin(2πt/24)) | identical | yes |
| a ∈ [-20,20], H=24, γ=0.95 | | | yes |

The tex's variance notation (e.g. N(30,25)) correctly maps to the code's std-dev arguments.

**Lu 2025 provenance (checked against the source, not the abstract).** Lu 2025 §3.3, §7.2, and
Appendix F contain a genuine wind-farm-equipped storage control problem with state
`s_t = (w_t, p_t, c_t)` = wind power, price, state-of-charge, and a three-component factorization
(wind / price / storage). The sim's base state and its three-way factored structure faithfully mirror
Lu's model. Differences (γ=0.95 vs Lu's 0.9; 11-point vs 3-point action grid; a revenue-minus-penalty
reward vs Lu's mismatch-penalty cost) are consistent with the tex's word "adapted," not "replicated."
The "adapted from the scaling experiment in Lu et al. (2025)" claim is truthful.

## 3. Data Integrity — PASS

- `compute_data` genuinely runs `compute_dp` / `compute_rl`; no hardcoded returns anywhere.
- **All 16 cache components load with current configs (hash MATCH, none stale/missing).**
- Every reported number reproduces exactly from cache:
  - DP row: d3=1110.2, d4=1113.2 (table 1110/1113); d5,d6 = None → TIMEOUT. Reproduced.
  - DQN: 1107.6±0.4, 1107.4±0.7, 1106.5±0.2, 1100.1±0.4 → table 1108/1107/1107/1100. Reproduced
    (1106.51 correctly rounds to 1107).
  - Factored: 1090.7/1082.8/1093.7/1083.6 with SE 4.2/4.1/1.8/4.9. Reproduced.
  - Linear AC: 1001.6/1016.0/1023.0/1008.9 with SE 29.4/33.9/25.4/30.7. Reproduced.
  - DP extrapolation: growth 2.157/dim (×8.6), d5=45.7 min, d6=6.59 h. Reproduced.
- SE uses `ddof=1`, n=10 per cell, all 10 per-seed values distinct in stdout (not 1-seed-as-10).
- `_format_se` asserts `se > 0`, a live guard against a zero-SE (identical-seed) cell.

## 4. Comparison Fairness — PASS (one disclosed asymmetry)

- Shared 11-point action grid, shared horizon, shared γ, all RL trained 3,000 episodes × 10 seeds.
- **Evaluation is a genuine paired comparison.** The environment's random draws per step (Poisson
  demand, `eps_w`, `eps_p`, aux `eps_i`) and per reset are all action-independent: no branch depends
  on the action, so for a fixed seed the exogenous shock sequence (w, p, demand) is identical across
  policies; only the endogenous SoC path differs. DP and every RL method evaluate on the same
  `EVAL_SEED=99` environment over 50 episodes, so the tex's "same shock sequences" claim is exact.
- Training seeds (0-9) are disjoint from the eval seed (99): no evaluate-on-training-data.
- **Disclosed asymmetry:** Factored RL is handed the reward-relevant dimension set that DQN and
  Linear AC must infer. This is the known-structure premise of factored MDPs and is stated in the tex,
  but it is still privileged information the other two do not receive.

## 5. Theoretical Sanity Checks — PASS

- DP cost is exponential: 36.7s (d3) → 317.5s (d4) → times out (d5, d6). Directly observed
  intractability, not merely extrapolated.
- RL returns are near-flat across d (DQN 1108→1100, Factored 1091→1084, Linear AC ~1002-1023),
  expected because the auxiliary dims are payoff-irrelevant by construction and each RL method can
  ignore them.
- DP (1110, 1113) sits marginally above the best RL (DQN ~1107). Consistent with a model-based planner
  slightly beating model-free RL; DP is explicitly a coarse benchmark, not an oracle, so no
  "beats-the-oracle" violation.
- **Weak spot:** the exponential fit rests on only TWO completed points (d3, d4), yet the tex reports
  "45.7 minutes at d=5 and 6.6 hours at d=6" to false precision. The growth factor 8.6×/dim also
  exceeds the ~7×/dim expected from pure state-count scaling (7^d states, constant per-state work);
  the excess is unexplained (likely dict overhead). Both are Reviewer-2 snark targets. Mitigant: the
  extrapolation is decorative; the load-bearing claim (DP times out, RL does not) is directly
  measured.

## 6. Information Leakage — PASS

- RL methods observe only `(next_state, reward, done)` from `env.step`; no peeking at the true model,
  reward function, or optimal policy.
- DP is model-based by design (it replicates the transition/reward to plan) — the legitimate
  "knows-the-model" benchmark, not cheating.
- Factored RL's knowledge of the reward-relevant dims is the factored-MDP structural assumption,
  disclosed in the tex (audited under fairness above).
- Greedy evaluation on a held-out seed; no train/test leakage.

## 7. Seed & Reproducibility — PASS

- RL: 10 seeds, `np.random.seed(seed)` + `torch.manual_seed(seed)` per unit, means and SEs reported.
- Env uses a separate `default_rng(seed)`; eval on fixed `EVAL_SEED=99`; DP integration on fixed
  `DP_SEED=42` (deterministic, single value reported — appropriate).
- Cache config hashing (`config_hash` over the per-component config incl. `DIM`) invalidates cleanly;
  confirmed all 16 hashes still match the live configs.

**Note on Linear AC's repeated exact returns** (e.g. d3: `915.3` on seeds 0/3/9, `1107.4` on seeds
4-7): not fabrication. Degenerate/near-constant greedy policies produce a deterministic return on the
fixed eval seed, so training seeds that collapse to the same policy return byte-identical values. The
`1107.4` value also appears in DQN's per-seed list, i.e. it is the good simple policy's return on seed
99 — an internal-consistency signal, not a red flag. The bimodality is disclosed in the tex ("attains
DP-level returns on some seeds while collapsing ... on others").

---

## Summary

A carefully built, honestly labeled illustrative simulation. The environment is faithful to both the
tex and Lu 2025's actual wind-farm storage model; all 16 cache components hash-match and every reported
number reproduces exactly; the paired-shock evaluation is genuinely well-designed; and the one real
integrity risk — presenting three generic RL baselines as the three theoretical pathways — is defused
by explicit, repeated disclaimers that they are analogies, not the cited algorithms. The residual
hostile-reviewer catches are all disclosed or explainable and none touch the substance:

1. Two-point exponential extrapolation reported to 0.1-min / 0.1-hour precision ("45.7 minutes",
   "6.6 hours"). Thin basis, over-precise; but decorative, since DP timeout is directly observed.
2. The curse is demonstrated with payoff-irrelevant-by-construction auxiliary dims, so methods that
   ignore noise trivially don't degrade — a gentle, mildly circular illustration (disclosed).
3. Growth factor 8.6×/dim vs the ~7×/dim expected from state-count scaling — minor, unexplained.
4. Factored RL receives privileged reward-relevant-dim information the other methods don't (disclosed).
5. Linear AC's exact-duplicate per-seed returns look odd at a glance (explainable; disclosed bimodality).

If the tex had claimed these were implementations of Lu 2025 / Liu 2022 / Du 2021, this would be a
50-75% audit. The disclaimers are present and thorough, so the substance holds.

**Bullshit score: 25%** — Reviewer 2 snarks at the two-point "6.6 hours at d=6" extrapolation and the payoff-irrelevant-by-construction curse demo, but the environment is faithful, the numbers reproduce exactly, and the illustrative-analogy disclaimers keep the substance intact.
