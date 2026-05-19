# Audit: ch06_games/sims/durable_goods_monopoly.py

**Date:** 2026-05-19
**Diagram-only:** no (CFR is actually trained; sweep results consumed by tex table)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch06_games/tex/rl_in_games.tex` §"The Coase Conjecture", lines 148–186; `\input` of `durable_goods_results.tex`.
**Cited paper PDFs read:** none of Coase 1972, Gul-Sonnenschein-Wilson 1986, Stokey 1981, Bulow 1982, or Ausubel-Cramton-Deneckere appear under `ch06_games/papers/`. The chapter `papers/` directory holds CFR/MARL papers (Brown 2019 Deep CFR, Zinkevich 2007 CFR, etc.) but no durable-goods reference PDF. The script's docstring claims it follows "Section 3.1.1 of Ausubel, Cramton, Deneckere" — that PDF is not in the chapter's papers directory.

## 1. Algorithm Identity

The script implements vanilla CFR (counterfactual regret minimization) over an extensive-form bargaining tree with private buyer types. The implementation looks like a textbook CFR loop: per-info-set regret matching, strategy-sum accumulation, recursive node traversal, time-averaged strategy. That is a faithful CFR.

However, the cited paper trio (Coase 1972, GSW 1986, Stokey 1981) is not an "algorithm identity" question — it is an *equilibrium-concept* question. CFR converges to a coarse-correlated equilibrium / Nash in two-player zero-sum, but the durable-goods game is two-player *non-zero-sum* with private information, where CFR's convergence guarantees are far weaker. The script never acknowledges this. The huge exploitability values (5–30) in the table are themselves evidence that CFR's standard guarantee is not biting here.

Score contribution: 25–30%.

## 2. Environment / MDP Fidelity

The script implements a **2-period** game with a **2-element price set** $\{P_L=100, P^*(\delta)\}$. The tex (line 156) admits the truncation: "$T = 2$ periods." The Coase conjecture, however, is fundamentally an asymptotic statement: as the inter-offer interval shrinks (equivalently, $T \to \infty$ with appropriate $\delta$), the price collapses to marginal cost. A 2-period model with only 2 admissible prices is a *screening-vs-pooling* exercise (à la Sobel-Takahashi / Fudenberg-Levine-Tirole 1985 / Ausubel-Deneckere), not a demonstration of the Coase limit.

The tex section is titled "The Coase Conjecture" and says "as the inter-offer interval shrinks ($\delta \to 1$), price collapses to marginal cost" but the simulation cannot exhibit that limit because the seller's price action set is hard-wired to $\{100, 200 - 100\delta\}$. The "screening price" is computed analytically and then plugged in as the only non-pooling action. The CFR is choosing between *two* analytical pre-computed prices, not discovering Coase-style price collapse.

This is the most serious fidelity problem: the environment is constructed in a way that bakes the answer into the action set and then "validates" the analytical threshold rather than the conjecture in the section title.

Score contribution: 40–50%.

## 3. Data Integrity

`compute_data()` does call `run_pi_sweep_experiment` and `run_delta_sweep_experiment`, which actually train CFR per `(π, δ)` and report `avg_strategy[seller_R1_info_set][1]` as P(Screen). The `durable_goods_results.tex` file contains numbers that match the cache (verified by unpickling `cache/durable_goods_monopoly.pkl`). Reported P(Screen) and NashConv values are real CFR outputs, not hardcoded.

What is hardcoded: the entire `compute_analytical_equilibrium` step-function (Theory column), and the "Eq. Type" column. These are not validation outputs; they are restatements of the model's closed-form solution.

Score contribution: 10–15%.

## 4. Comparison Fairness

The only comparison is CFR vs analytical step function. There is no commitment-vs-no-commitment baseline (which would be the natural Coase demonstration), no comparison to backward induction, no benchmark RL method. The "validation" reduces to "did CFR find a known equilibrium?" — which is fine as a sanity check, but is not what a chapter titled "The Coase Conjecture" promises.

The `Status` column is suspicious. In the π-sweep table (`durable_goods_results.tex`):
- π=0.50: P(Screen)=0.001 vs Theory=0.5 → `\checkmark` (mis-marked; CFR pooled at the indifference point)
- π=0.55: P(Screen)=0.002 vs Theory=1.0 → `\checkmark` (mis-marked; CFR clearly *pooled*, theory said *screen*)
- π=0.60: P(Screen)=0.506 vs Theory=1.0 → `\checkmark` (mis-marked; CFR is mixing, theory pure-screening)

The code excuses these via a hard-coded "near threshold (0.45–0.60), strategies can mix during transition" carve-out at line 936–938. That carve-out is post-hoc, undocumented in the tex, and lets π=0.55 (where CFR overwhelmingly pools and theory predicts pure screening) silently pass.

Score contribution: 30–40%.

## 5. Theoretical Sanity Checks

This is where the audit gets damning. Run `python3 -c "import pickle; data=pickle.load(open('ch06_games/sims/cache/durable_goods_monopoly.pkl','rb')); ..."`:

δ-sweep at π=0.7 (analytical theory: screen for ALL δ since π > 0.5):
```
delta=0.50  P(Screen_CFR)=1.000  Theory=1.0  expl=22.503
delta=0.65  P(Screen_CFR)=0.999  Theory=1.0  expl=20.257
delta=0.70  P(Screen_CFR)=0.991  Theory=1.0  expl=19.515
delta=0.75  P(Screen_CFR)=0.013  Theory=1.0  expl=17.531
delta=0.80  P(Screen_CFR)=0.002  Theory=1.0  expl=14.021
delta=0.85  P(Screen_CFR)=0.001  Theory=1.0  expl=10.519
delta=0.90  P(Screen_CFR)=0.000  Theory=1.0  expl=7.019
```

CFR collapses to pooling at δ ≥ 0.75 even though the script's own analytical solver (with π=0.7, threshold π*=0.5) says theory predicts pure screening throughout. The tex paragraph at line 185 reinterprets this divergence as Coase-consistent: "at $\delta \approx 0.75$, the seller switches to pooling as patient buyers erode the screening premium, consistent with the Coase conjecture."

This is a *theory-stretching narrative*. The script's own analytical solver disagrees with that reading: `prob_high_analytical = 1.0` at every δ in this sweep. The actual cause is more mundane — at high δ, the screening price $P^*(\delta) = 200 - 100\delta$ shrinks toward $v_L=100$, and the screening profit $\pi P^* + (1-\pi)\delta v_L$ approaches the pooling profit, so the seller becomes nearly indifferent and CFR's average strategy drifts to the corner that any small regret push selects. The script never reports this. The tex paper's claim that the δ=0.75 switch is "consistent with the Coase conjecture" is not what the analytical column says, and the analytical column is what the script uses to grade itself.

NashConv (exploitability) values of 5–30 in a game whose total achievable seller payoff is bounded by 200 (a buyer's max valuation) are catastrophic. The π-sweep table shows max NashConv = 30 — equivalent to roughly 15% of the maximum possible utility. CFR has not converged to ε-Nash for any reasonable ε. The script reports this as "convergence to Nash equilibrium" in the title of `plot_exploitability_convergence`. The convergence plot is a log-scale plot, which makes the 20-unit gap look like progress; in linear scale it would look like nothing.

The tex paragraph at line 185 also claims "CFR recovers the sharp phase transition at $\pi^* = 0.5$." It does for π ≤ 0.50 and π ≥ 0.65, but at π=0.55 (clearly above the analytical threshold) CFR pools. So the transition CFR finds is not at π=0.5 — it is somewhere in [0.55, 0.65].

The "Coase conjecture" claim has no $T \to \infty$ behavior, no $\delta \to 1$ Coase limit price → marginal cost demonstration, no continuous-time-limit analysis. The simulation cannot, by construction, exhibit Coase-style price collapse: it has a 2-element action set.

Score contribution: 60–70%.

## 6. Information Leakage

Buyers' info sets are indexed by their type (`BL:` vs `BH:`); the seller's info sets are indexed only by history, not by buyer type. The seller does not observe the buyer's valuation directly, only the acceptance/rejection. That looks correct. No leakage.

Score contribution: 0%.

## 7. Seed & Reproducibility

`np.random.seed(42)` is set at module load. CFR is deterministic given starting regrets (all zeros), so there is no Monte-Carlo seed sweep here — every run produces the same CFR trajectory. The chapter standard requires "minimum 10 seeds" with means and standard errors. This is N=1 dressed up as N≥10 because CFR is internally deterministic; but the section reports single-seed point estimates as if they were equilibrium predictions. A π-sweep with stochastic CFR variants or random initialization would have been the right move.

The `pi_sweep_iterations: 5000` is far too few for this game to drive exploitability anywhere near zero (current max 30). The Brown 2019 Deep CFR paper sitting in `papers/` runs millions of iterations on much smaller games.

Score contribution: 15–20%.

## Hostile-Reviewer Summary

The script implements CFR correctly on a 2-period, 2-action bargaining game and produces results that mostly match the closed-form screening/pooling threshold the script also computes analytically. Then it labels this exercise "The Coase Conjecture" and writes a tex paragraph reinterpreting a CFR convergence failure at δ ≥ 0.75 (caused by near-indifference and finite iterations) as a Coase-consistent regime switch — even though the script's own analytical solver predicts screening throughout that region. The π=0.55 row is marked `\checkmark` despite CFR pooling where theory says screen, via an undocumented "near threshold" carve-out in the validation function. NashConv values of 5–30 in a game with payoffs bounded by 200 are reported as "convergence to Nash equilibrium" because the plot is on a log scale. The chapter's papers directory contains zero durable-goods references; the docstring cites Ausubel-Cramton-Deneckere but the paper is not in `papers/`. Most importantly, a 2-period game with a hard-coded 2-element price set cannot demonstrate the Coase conjecture — Coase is an $T \to \infty$ / $\delta \to 1$ price-collapse statement, and this artifact pre-supplies the screening price $P^*(\delta)$ analytically as one of the only two actions the seller can take.

A hostile reviewer reads §"The Coase Conjecture", sees a 2-period game with a 2-element price set, sees the screening price computed analytically and handed to the algorithm, and sees the "Coase limit" claim at δ=0.75 contradicted by the script's own theory column. They conclude that the section's title does not match the artifact and that the validation logic was tuned post-hoc to checkmark everything. They are particularly hostile about the NashConv-as-Nash-convergence framing.

**Bullshit score: 65%** — Reviewer 2 lands on the title/artifact mismatch (Coase conjecture is asymptotic; sim is 2-period with hard-coded screening price), then notices the π=0.55 row is incorrectly checkmarked, then catches the δ=0.75 narrative reinterpreting a CFR collapse as a Coase regime switch against the script's own theory column. Reputational territory, not just methodological — the section makes a famous theorem from economics into the headline result and delivers a screening-vs-pooling phase diagram instead.
