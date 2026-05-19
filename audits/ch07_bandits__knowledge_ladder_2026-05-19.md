# Audit: ch07_bandits/sims/knowledge_ladder.py

**Date:** 2026-05-19
**Diagram-only:** no (Monte Carlo over 10 seeds × T=200,000 × 6 algorithms)
**Cited tex file(s):** `ch07_bandits/tex/dynamic_pricing.tex` (lines 158-172, `\subsection{Simulation Study: Structural Knowledge and Curve Learning}`)
**Cited paper PDFs read:** `papers/Misra-DynamicOnlinePricing-2019.md` (Sections 3.1-3.2, 4.1-4.2), `papers/auer2002_finite_time_multiarmed_bandit.md` (UCB1 definition, regret bound). Thompson (1933) listed in refs.bib (`Thompson1933`) — original-likelihood paper, not the modern TS analysis, but used as legacy citation. No Lai-Robbins (1985), Bubeck-Cesa-Bianchi (2012), or Russo-Van Roy (2014) PDF in `papers/`.

---

## 1. Algorithm Identity

Six algorithms; verify each against the tex/paper.

- **EpsilonGreedy (Level 0)** — Fixed ε=0.1, random tie-breaking. Standard. Matches the "no adaptive exploration" claim in tex line 163. Identity OK.
- **LearnThenEarn (Level 1)** — Uniform random for first τ·T = 10,000 rounds, then commits to empirical-best arm. Matches the 5% LTE described in the tex and in Misra (2019) Section 4.1 (a learn-then-earn comparator). Identity OK.
- **UCB1 (Level 2)** — Index = `mean + sqrt(2·log(t+1) / counts)`. Matches Auer et al. (2002) Eq. (1). Bonus is unscaled by reward range (in [0,1] rewards bonus is fine; here rewards are in [0, max(price)] = [0, 1] so the assumption holds). Identity OK.
- **ThompsonSampling (Level 3)** — Beta-Bernoulli posterior on per-arm *purchase rate*, then `argmax p_k · θ_k`. This is a *price-scaled* Thompson sampler (a la Ganti 2018 / Misra ZipRecruiter setup). It is *not* a posterior on the profit reward directly, which avoids the bounded-Bernoulli mismatch. Identity OK as "TS on the purchase rate, profit-aware selection".
- **UCB_PI (Level 4)** — Per Misra (2019) §3.1, Eq. (7):
  - Maintains per-segment `(p_min, p_max)` from observed purchases via WARP.
  - Estimates `δ_hat` from "crossed" segments (with both an LB and UB strictly inside [V_L, V_H]) using `max + (max - mean) = 2·max - mean`, an ad-hoc spec — *Misra's paper uses an estimator of δ from the data; this concrete formula `delta_max + (delta_max - mean_delta)` is not the textbook one*. Hostile reviewer would flag this as a homegrown estimator dressed as a paper implementation.
  - Builds demand UB/LB from `v_hat ± δ_hat` and computes profit UB/LB for each price. Dominance test: `ub_profits ≤ max_lb` (matches paper's dominance criterion).
  - Index: `mean + p_k · sqrt(2·log(t+1)/counts)` for non-dominated arms. Matches Eq. (7) of Misra exactly (the price-scaled bonus).
  - **Subtle deviation:** the script uses `V_L` and `V_H` (the *true* bounds) as the prior `p_min/p_max` for each segment. The paper assumes the firm knows the valuation support, so this is consistent.
- **UCB_PI_Tuned (Level 5)** — Adds variance term per Misra §3.2, modeled on Auer's UCB-tuned. The exploration term in the code is `2·p_k·δ_hat·sqrt(log(t+1)/counts · min(0.25, V_kt))`. Misra Eq. (8) is `2·p_k·δ_hat·sqrt(log(n+S)/N_kt · min(0.25, V_kt))` where `V_kt = empirical_variance + sqrt(2·log(n+S)/N_kt)`. Two issues:
  1. The script uses `log(t+1)` not `log(n + S)`. Minor since n+S ≈ t for large t with S=1000, but not identical at small t.
  2. The factor of `2·p_k·δ_hat` is correct.

Overall: implementations are recognizable instantiations of their named algorithms. The δ-hat estimator inside UCB-PI is the most idiosyncratic piece; the script does not cite a formula. A hostile reviewer would ask for a derivation or paper reference.

## 2. Environment / MDP Fidelity

Misra (2019) §4 specifies: K=100 prices on $0.01-$1.00 grid, S=1,000 segments, δ=0.1 true heterogeneity, T=200,000, segment midpoints `v_s ~ Uniform[V_L+δ, V_H-δ]`, within-segment `v_i ~ Uniform(v_s-δ, v_s+δ)`, consumer buys iff `v_i ≥ p`. The script matches all of these exactly (lines 32-38, 67-72). The "buy iff v_i ≥ p" WARP rule is correctly implemented (line 361).

Mismatch: the tex line 163 says "segment midpoints v_s ~ Uniform(0.1, 0.9)" which corresponds to `Uniform(V_L+δ, V_H-δ)` with δ=0.1 — consistent.

Fidelity: OK.

## 3. Data Integrity

- `compute_data()` runs `run_experiment(seed)` for `seed in range(10)`, real loop, no hardcoded numbers. Each seed loops T=200,000 rounds with all 6 algorithms. Stdout reports 35:50 wall time for the 10-seed loop, plausible for the work.
- Cumulative regret aggregated at sample_interval=100, then averaged across seeds. Tables in stdout are computed from these arrays (verified by reading code paths).
- Cache mechanism: `load_results(...)`/`save_results(...)`. If config changes (e.g., K, T, N_SEEDS, δ), cache invalidates. CONFIG dict captures these.
- Numbers in stdout (`regret_at_T200K`: eps=2,263, LTE=1,771, UCB1=6,734, TS=1,136, UCB-PI=4,503, UCB-PI-tuned=780) are not hardcoded; they come from `regret_arrays[name][:, -1].mean()`.

Integrity: OK.

## 4. Comparison Fairness

This is the cleanest part of the script. In `run_experiment(seed)`:
- A single `segment_ids` sequence of length T and a single `valuation_offsets` sequence (line 320-321) are sampled once from the seed's RNG.
- The inner loop iterates `for name in algorithms:` and gives every algorithm the *same* `(segment_id, v_i)` realization at time t. Each picks its arm `arm = alg.select_arm(t, alg_rng)`, then the reward is computed from the *common* `v_i ≥ price` test.
- Each algorithm has its own per-algorithm RNG (offsets +1000, +1500, ...), so internal randomization (ε-greedy choice, TS posterior draws, LTE uniform sampling) doesn't share entropy.

This is approximately optimal common-random-numbers (CRN) coupling: all algorithms see the same customer stream and same valuation noise, differing only in arm choice. **However**, all algorithms receive `segment_id` as an argument to `update()`. ε-greedy, LTE, UCB1, and TS *ignore* the segment_id (only UCB-PI uses it). This is consistent with their declared identities — the simpler algorithms truly don't condition on segments.

Fairness: OK. The 5% LTE explore fraction is arbitrary and is one of many possible LTE parameters (Misra tests 0.1%, 1%, 5%, 10%, 25%); using only 5% is a reasonable single-point choice but a hostile reviewer would ask for the LTE-vs-fraction sweep.

## 5. Theoretical Sanity Checks

The stdout's rate-diagnostic table is the most damning piece:

| T       | ε-greedy R/T | LTE R/T^⅔ | UCB1 R/√T | TS R/√T | UCB-PI R/log T | UCB-PI-tuned R/log T |
|--------:|-------------:|----------:|----------:|--------:|---------------:|---------------------:|
| 10K     | 0.0160       | 2.2       | 7.2       | 2.9     | 85.7           | 36.7                 |
| 50K     | 0.0125       | 0.9       | 11.7      | 2.8     | 209.5          | 50.0                 |
| 100K    | 0.0116       | 0.6       | 13.5      | 2.7     | 281.7          | 55.6                 |
| 200K    | 0.0113       | 0.5       | 15.1      | 2.5     | 368.9          | 63.9                 |

Predicted-rate stabilization (small/constant column over T) is the test. Observations:

- **ε-greedy: R/T not stable** (0.0160 → 0.0113, declining). This would suggest sub-linear regret, but in fact for a fixed ε with a non-stochastic-optimal-action problem the algorithm's regret IS Θ(T) only asymptotically with a non-zero rate floor; the early phase exhibits transient improvement. Hostile reviewer would say: "your column doesn't stabilize, so calling this Θ(T) on the basis of 200K rounds is premature." Predicted column flagged.
- **LTE: R/T^⅔ DROPS** (2.2 → 0.5) — not stabilizing, declining. After commit at t=10K, LTE just earns the empirical-best arm forever; regret grows linearly *at a small rate*, not as T^⅔. R/T^⅔ would only stabilize if LTE were genuinely T^⅔-regret, which it isn't if exploration length is fixed (it's Θ(T) with a small constant). Predicted column wrong.
- **UCB1: R/√T grows** (7.2 → 15.1) — not stabilizing, growing. UCB1 should be O(√(KT log T)) by Auer 2002 Thm 1. The growth here suggests the bound is slack or the script is in a pre-asymptotic regime; with K=100 and Δ_min tiny, UCB1 may not yet be in its asymptotic √T regime. Tex line 165 itself acknowledges "UCB1 is worse" but doesn't tie this to the rate diagnostic.
- **TS: R/√T DECLINING** (2.9 → 2.5) — closer to stable but trending down. Consistent with TS doing better than its worst-case √T bound, which is typical empirically.
- **UCB-PI: R/log T GROWING** (85.7 → 368.9) — *strongly* growing. The tex line 165 itself notes this: "R/log T for plain UCB-PI keeps rising over the plotted checkpoints, so the simulated finite-sample behavior is not visibly logarithmic." This is honest, but it is a meaningful failure of the theoretical claim on this T range — the regret bound is not visibly tight.
- **UCB-PI-tuned: R/log T GROWING** (36.7 → 63.9) — also rising, less severely. Same caveat.

The result is the inverse of what the "ladder" framing wants to show: UCB-PI is *worse* than TS and even LTE on this run; only UCB-PI-tuned beats TS. This is a known empirical issue in Misra (2019) too — the *tuned* variant is the one that performs well at the modest T tested. The tex now explicitly acknowledges this ("good finite-sample performance depends on the variance-tuned implementation"), which is appropriate.

What's missing: there's no analytical reference value (no DP-style oracle gap or Lai-Robbins lower bound shown on the figure as a reference line). The figure does show Θ(T), O(√T), O(log T) reference *lines* fit to the final regret, so the slope-comparison story is shown — but this is a *visual* claim only, not a quantitative rate fit.

Hostile reviewer: "The ε-greedy reference line is fit to make ε-greedy look linear, but ε-greedy itself isn't on a log-log straight line at this T. The whole point of the figure is to teach the rate hierarchy, but most of the rate columns don't visibly stabilize at the chosen T."

## 6. Information Leakage

- All algorithms receive `(arm, reward, segment_id, purchased)` in `update()`. ε-greedy, LTE, UCB1, TS *use only* `arm` and `reward` (or `purchased` for TS).
- UCB-PI and UCB-PI-tuned use `segment_id` and `purchased`. **This is consistent with the model**: the paper assumes the firm observes segment membership at the time of pricing (e.g., from cookies, demographics). The simulation gives this info to the methods that need it; not leakage in the technical sense.
- No algorithm reads `optimal_arm`, `OPTIMAL_PRICE`, or the true demand model during selection. UCB-PI/tuned uses `V_L`, `V_H` (true valuation bounds), which the paper assumes are known to the firm.
- One subtle issue: UCB-PI uses `segment_weights` as a known input (line 195). In Misra (2019), segment weights are also known by assumption (segment shares are observable). OK.
- The "true" `δ_TRUE = 0.1` is not directly used by any algorithm; UCB-PI estimates `δ_hat`. Good.

Leakage: OK, consistent with paper assumptions.

## 7. Seed & Reproducibility

- `np.random.seed(42)` at module load.
- N_SEEDS = 10 (meets the "minimum 10" floor in CLAUDE.md).
- Per-seed `np.random.RandomState(seed)` for demand model and customer arrival; separate RNGs per algorithm for arm selection (seed+1000, +1500, ...).
- Standard errors computed as `std/sqrt(N)` and displayed both in stdout and in the figure as `±2·SE` shaded bands.
- Shaded SE bands in the figure look very narrow given N=10, but cumulative regret variance is genuinely low here (large T, lots of averaging within each seed), so this is plausible.

Reproducibility: OK. N=10 is the floor — for a chapter "ladder" figure, more seeds would tighten the picture (especially LTE has SE=242 at T=200K vs mean 1,771, a 14% relative SE, suggesting more seeds would help here).

---

## Hostile-Reviewer Summary

The script implements six recognizable bandit algorithms on a faithful Misra (2019) demand environment and uses tight common-random-numbers coupling across algorithms. The bullet points the tex makes are honest about the limits of the run, and the diagnostic stdout table is the kind of thing a careful reader actually wants to see.

Three weaknesses a hostile reviewer would seize on:
1. The δ-hat estimator inside UCB-PI (`delta_max + (delta_max - mean_delta)`) is homegrown — not a derivation from the paper, no citation. It works, but it's the kind of detail a referee will demand justification for.
2. The rate-diagnostic columns mostly *do not stabilize* at T=200K, and the figure is built around an asymptotic-rate-hierarchy narrative. The tex acknowledges this for UCB-PI but doesn't flag it for ε-greedy and LTE, where the diagnostic is just as broken.
3. UCB-PI in this run is *worse* than Thompson Sampling and LTE — the "ladder" story collapses if you read across the third row of the regret table. The tex (line 165) is straightforward about this, which insulates it somewhat, but a hostile reviewer would still call the figure misleading: the legend orders things by *claimed* rate (Θ(T) → O(log T)) while finite-sample performance permutes this order completely.

The substance is fine; this is real working code, the numbers came out of a real 36-minute run, and the limitations are acknowledged in the tex. It's also already accompanied by a sister curve-learning sim that softens the chapter's claims.

**Bullshit score: 25%** — Reviewer 2 catches (a) the homegrown δ-hat estimator, (b) the rate-diagnostic table failing to stabilize for half the methods, and (c) the awkward fact that the "knowledge ladder" finite-sample ordering is non-monotone. The tex line 165 disclaimer keeps the substance defensible.
