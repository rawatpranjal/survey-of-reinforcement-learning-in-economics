# Simulation Audit: Numbers in the Compiled Paper

**Date:** 2026-03-07
**Scope:** Only simulations whose `.tex` tables or `.png` figures are `\input`/`\includegraphics` in active (non-archive) chapter tex files.

---

## 1. ch03a — Illustrated Gridworld (13 algorithms)

**Files:** `gridworld_illustrated.py`, `gridworld_study_results.tex`, `illustrated_example.tex`

### ISSUE: "Ep to 99%" column measures RMSE, but prose discusses return convergence

The table column "Ep to 99%" is defined in the caption (line 18) as "first checkpoint at which RMSE drops below 1% of V\*\_max" (value function accuracy). But the prose tier descriptions (lines 46–54) discuss "episodes to reach 99% of the optimal return" (policy quality). These are fundamentally different metrics.

The code confirms this: `episodes_to_99` (return-based, line 352) and `episodes_to_value_conv` (RMSE-based, line 365) are separate variables, and the table uses `episodes_to_value_conv` (line 1033).

Consequences:

| Algorithm | Prose claim (return-based) | Table "Ep to 99%" (RMSE-based) |
|-----------|---------------------------|-------------------------------|
| NPG | "100 episodes to reach 99% of optimal return" (line 40) | `---` (RMSE = 3.58, never < 0.10) |
| Q(λ) | "fast tier, 200 episodes" (line 48) | 1500 |
| Q-learning | "middle tier, 400 episodes" (line 50) | `---` (RMSE = 0.33) |
| SARSA | "slow tier, 800 episodes" (line 54) | `---` (RMSE = 3.73) |

A reader comparing prose to table will find contradictions. NPG is called the "fastest" learner but shows `---` in the table. Q(λ) is in the "fast tier (100–200)" but shows 1500 in the table.

**Severity: HIGH.** Same phrase "99%" used for two different metrics. Prose and table appear to contradict each other.

---

## 2. ch03_theory — Curse of Dimensionality (wind farm)

**Files:** `wind_farm_curse_study.py`, stdout, `results.tex`, `curse_of_dimensionality.tex`

### Minor discrepancies

| Item | Stdout | Table | Prose |
|------|--------|-------|-------|
| DP d=4 time | 327.7s | 332s | "328s" (line 147) |
| Factored RL d=4 | 1069.8 | 1072 | "1070–1104" |

None individually alarming. The Factored RL d=4 gap (1069.8 → 1072) is not simple rounding.

All RL methods show ±0 standard error (single seed only). Not a numerical error, but weakens claims.

**Severity: LOW.** Numbers are close. Single-seed results flagged but acceptable for a demonstration.

---

## 3. ch04 — Bus Engine Replacement (fleet scaling)

**Files:** `benchmark_bus_engine.py`, `econ_benchmark.py`, `bus_engine_results.tex`, `applications.tex`

### ISSUE: DP Reward and DQN Reward use different metrics

Table values for N=1:
- DP Reward: −12.72
- DQN Reward: −65.01 ± 0.03

The ratio DP/DQN ≈ 0.196 is consistent across N=1–3 (~5.1×). Both `evaluate_dp_policy` and `evaluate_dqn_policy` in `econ_benchmark.py` (lines 354–390) use identical undiscounted episode sums over the same horizon. However, the `bus_engine_results.tex` table format (using `\hline`, different column order) does NOT match what either `benchmark_bus_engine.py:make_latex_table()` (line 315, uses `\toprule` and includes heuristics) or `econ_benchmark.py:make_scaling_table()` (line 637, uses `\toprule` and has Agreement column) would generate. The table was produced by a different code path or older script version where "DP Reward" was likely V(s₀) from VI (discounted infinite-horizon value) rather than simulation-based evaluation return.

The prose (line 205) claims "DQN converges to the DP optimal policy where DP is computable" — but the table shows a 5× difference in reward. This claim is not supported by the numbers shown.

### Missing heuristic baselines

Prose (line 205) claims DQN "outperforms simple heuristics (threshold replacement, never replace)." The table has no heuristic columns. The code computes them (`h_threshold_reward`, `h_never_reward` in `benchmark_bus_engine.py`) but the current results table omits them.

### N=8 Bellman residual: 1.58 ± 1.56

Standard error nearly equal to the mean — some seeds did not converge. The prose doesn't acknowledge this.

### No stdout file

Cannot independently verify the table numbers.

**Severity: HIGH.** DP vs DQN reward comparison is likely apples-to-oranges (discounted V(s₀) vs undiscounted episode return). Prose claims convergence not supported by table. Missing heuristic baselines.

---

## 4. ch06_games — Durable Goods Monopoly (Coase conjecture)

**Files:** `durable_goods_monopoly.py`, `durable_goods_stdout.txt`, `durable_goods_results.tex`, `rl_in_games.tex`

### Status: CLEAN

- Stdout matches `.tex` table exactly (verified row by row for π-sweep)
- Phase transition at π\* = 0.5 matches theory
- Screening price P\*(δ) = 200 − 100δ verified with max error = 0.000000
- NashConv values are high (5–30) but explained by buyer indifference at P\*
- Prose claims match table numbers

**Severity: NONE.**

---

## 5. ch07_bandits — Knowledge Ladder (dynamic pricing)

**Files:** `knowledge_ladder.py`, `knowledge_ladder_stdout.txt`, `knowledge_ladder_results.tex`, `dynamic_pricing.tex`

### Stdout-to-table: MATCH (verified all 24 cells)

### ISSUE: UCB-PI (untuned) does not achieve claimed O(log T) regret

The stdout rate diagnostics (lines 34–41) show R/log(T) growing across checkpoints:

| T | R/log(T) for UCB-PI |
|---|---------------------|
| 10,000 | 85.7 |
| 50,000 | 209.5 |
| 100,000 | 281.7 |
| 200,000 | 368.9 |

A true O(log T) algorithm would have R/log(T) stabilize. The ratio is growing, consistent with O(√T) behavior (R/√T: 7.2 → 11.7 → 13.5 → 15.1, roughly stabilizing). UCB-PI-tuned correctly achieves O(log T): R/log(T) = 36.7 → 50.0 → 55.6 → 63.9 (slower growth, plausibly converging).

The prose (line 141) classifies UCB-PI as achieving O(log T) through WARP-based dominance elimination. The data does not support this at T = 200,000.

### Minor prose issue

Line 141: "UCB-PI's regret of 4,503 is comparable to UCB1" — UCB1 has 6,734. UCB-PI is 33% better. "Comparable" understates the difference.

### Percentage calculations verified

- "31% reduction over Thompson": (1136−780)/1136 = 31.3% ✓
- "83% reduction over UCB-PI": (4503−780)/4503 = 82.7% ≈ 83% ✓

**Severity: MODERATE.** The theoretical rate claim for UCB-PI (untuned) is not supported by the data. All numbers are internally consistent.

---

## 6. ch08_rlhf — RLHF Gridworld

**Files:** `gridworld_rlhf.py`, `gridworld_rlhf_results.tex`, `gridworld_hypothesis_tests.tex`, `rlhf.tex`

### CRITICAL: Script has been rewritten; tables are from old version

The current `gridworld_rlhf.py` (10×10 grid, 6 methods, hazards/traps, 30 seeds) does NOT match the paper's description or results:

| Item | Current script | Paper |
|------|---------------|-------|
| Grid | 10×10 | 5×5 (line 51) |
| Methods | 6 (NN, correct, misspec, DPO, Q-learn, DP) | 3 (RLHF, DPO, Q-learn) |
| Params | 4 structural params | 2 structural params |
| Seeds | 30 | 50 (caption, line 55) |
| RLHF model | Neural net (~4800 params) | 2-parameter structural |
| DP return | 6.524 (from stdout) | 8.50 (from table) |

The `.tex` result files still contain the OLD 5×5 results (dated Mar 7, 13:09). Re-running the current script will overwrite them with completely different numbers.

A second hypothesis file `gridworld_rlhf_hypothesis.tex` (dated Jan 30, 20:40) exists from the new script, showing H1–H4 all "NO" — contradicting the paper's Table 2 which shows H1–H3 "Yes".

### The old results (what's in the paper) are internally consistent

| Claim | Verification |
|-------|-------------|
| RLHF K=50 → 8.50 ± 0.00, matches DP | Table line 10: 8.50 ± 0.00 ✓ |
| "1,981× fewer queries": 99,030/50 | = 1,980.6 ≈ 1,981 ✓ |
| "400× more for DPO": 20,000/50 | = 400 ✓ |
| DPO K=10 = −45.4% of DP: −3.86/8.50 | = −45.4% ✓ |
| DPO K=20,000 = 96.7%: 8.22/8.50 | = 96.7% ✓ |
| H4 p=0.0814 | Table: 0.0814 ✓ |
| H5 p=0.5523 | Table: 0.5523 ✓ |

The arithmetic within the paper is internally consistent. The problem is reproducibility: no script exists that generates the published numbers.

### The new script's results (from stdout) tell a different story

From `gridworld_rlhf_stdout.txt` (new script, incomplete run):
- DP return: 6.524 (vs 8.50 in paper)
- NN RLHF at K=1000: 6.454 ± 0.042 (98.9% of DP — takes 20× more data)
- Correct structural at K=5000: still has high variance
- DPO: negative returns through K=5000

**Severity: CRITICAL.** The script no longer generates the published results. No old script exists for reproduction. The newer hypothesis file contradicts the paper.

---

## 7. ch09_causal — Confounded OPE

**Files:** `confounded_ope.py`, `confounded_ope_stdout.txt`, `confounded_ope_results.tex`, `causal_rl.tex`

### Status: CLEAN

- All stdout values match `.tex` table (verified all cells, rounding to 3 decimal places)
- Oracle constant at −6.9437 across all ρ values ✓
- Naive OLS bias monotonically increasing with ρ ✓
- IV(CF) < Backdoor < DR < Naive OLS ranking stable across all ρ ✓
- Prose claims match numbers exactly

**Severity: NONE.**

---

## Summary

| # | Chapter | Simulation | Severity | Core Issue |
|---|---------|-----------|----------|------------|
| 1 | ch03a | Gridworld (13 algos) | **HIGH** | Table "Ep to 99%" measures RMSE; prose discusses return convergence. Same name, different metrics. |
| 2 | ch03_theory | Wind farm curse | LOW | Minor rounding discrepancies, single-seed results |
| 3 | ch04 | Bus engine fleet | **HIGH** | DP and DQN rewards likely use different metrics (5× gap). Prose claims not supported by table. Missing heuristics. |
| 4 | ch06_games | Durable goods | CLEAN | All verified |
| 5 | ch07_bandits | Knowledge ladder | MODERATE | UCB-PI (untuned) doesn't achieve claimed O(log T) rate empirically |
| 6 | ch08_rlhf | RLHF gridworld | **CRITICAL** | Script rewritten; paper shows old results. No reproduction path. New hypothesis file contradicts paper. |
| 7 | ch09_causal | Confounded OPE | CLEAN | All verified |
