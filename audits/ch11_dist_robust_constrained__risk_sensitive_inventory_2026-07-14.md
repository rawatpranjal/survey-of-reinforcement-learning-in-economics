# Simulation Audit — Risk-Sensitive Inventory Management via IQN

- **Sim:** `ch11_dist_robust_constrained/sims/risk_sensitive_inventory.py`
- **Date:** 2026-07-14
- **Type:** FULL (never previously judge-audited)
- **Auditor mode:** fresh independent hostile referee, read-only, no retrain permitted (DP oracle re-verified cheaply)

**Files read (end to end):**
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/risk_sensitive_inventory.py`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/risk_sensitive_inventory_stdout.txt`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/risk_sensitive_inventory_table.tex`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/risk_sensitive_inventory_policy.png` (viewed)
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/risk_sensitive_inventory_returns.png` (viewed)
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/tex/dist_robust_constrained.tex` (lines 40–196, the distributional-RL prose + the sim subsubsection)
- `/Users/pranjal/Code/rl/sims/sim_cache.py` (cache-key semantics)
- git log / mtimes for all five artifacts + cache pickle

**Outputs the script writes (enumerated from source):** `risk_sensitive_inventory_table.tex` (3 rows), `risk_sensitive_inventory_returns.png` (return CDFs), `risk_sensitive_inventory_policy.png` (order-by-inventory), plus console diagnostics captured to `_stdout.txt`. Cache: single monolithic `risk_sensitive_inventory.pkl`.

**Independent re-verification performed:** reimplemented the newsvendor MDP + `dp_solve` from scratch with `/usr/local/bin/python3`. DP average initial value reproduces at **40.44** (stdout line 2, table 40.4). DP `t=0` policy = `[5,5,5,5,5,5,5,4,3,2,1,0,0,0,0,0]` (order 5 at low stock, 0 at high stock), which matches the dashed oracle curve in the policy figure. Demand mixture mean = 4.4 = 0.8·3 + 0.2·10.

---

## Step 3 — What claim is this sim evidence for?

The surrounding prose (lines 86–148) develops the IQN → distortion-risk bridge: a single IQN learns the continuous quantile function $F_Z^{-1}(\tau;s,a)$ (Dabney 2018b), and any distortion risk measure (mean, CVaR$_\alpha$, CPT) is obtained by changing only the $\tau$-sampling distribution at decision time, with the network unchanged. CVaR$_\alpha$ corresponds to $\tau \sim U([0,\alpha])$.

The sim (lines 151–196) is the mechanism demonstration: **one** IQN is trained per seed with $\tau\sim U([0,1])$, then the *same* net is evaluated under two decision-time samplings, $U([0,1])$ (neutral) and $U([0,0.05])$ (CVaR$_{95}$), against the exact DP oracle. The lesson the sim must sell: re-sampling $\tau$ at decision time alone shifts behavior and improves the left tail without retraining.

---

## Criteria verdicts

### (a) Algorithm identity — PASS
The IQN loss is the Dabney 2018b quantile-Huber loss, correct term-by-term:
- **Cosine $\tau$ embedding** (lines 148–166): $\phi_j(\tau)=\mathrm{ReLU}(\sum_i \cos(\pi i\,\tau)w_{ij}+b_j)$ merged multiplicatively with the state embedding $\psi(s)\odot\phi(\tau)$. Standard IQN architecture (uses $i=1..64$ rather than $0..63$, a trivial variant).
- **Pairwise TD** (line 270): `td[b,i,j] = target_j − current_i`, shape (batch, $K$, $K'$). Correct $\delta_{ij}$.
- **Quantile weight** (lines 276–279): `|tau_i − 1{δ_ij<0}| · huber(δ_ij)`, with `tau_vals` broadcast over the target index $j$ so the *current* quantile's $\tau_i$ weights the check function. Correct $\rho^\kappa_{\tau_i}$.
- **Huber** $\kappa=1$ (lines 271–275): exact. Division by $\kappa$ omitted but $\kappa=1$ makes it a no-op.
- **Reduction** (line 280): `sum(dim=j).mean(dim=i).mean()` $= \frac1K\sum_i\sum_j\rho$. With $K=K'=8$ this equals the paper's $\frac1{K'}\sum_i\sum_j$ exactly. (The tex eq. 99 writes $\frac1{KK'}$; the extra constant is a loss scale absorbed by Adam — immaterial.)
- **Bellman target** (lines 256–265): mean-over-$\tau$ greedy next action, Polyak target ($\tau_{\text{target}}=0.01$), $r+\gamma(1-d)Z'$. Risk-neutral bootstrap — matches "trained with $U([0,1])$."
- **Risk selection** (lines 309–324): CVaR `taus = rand·alpha` = $U([0,\alpha])$ ✓; CPT uses the Tversky–Kahneman weighting $w(u)=u^\gamma/(u^\gamma+(1-u)^\gamma)^{1/\gamma}$, $\gamma=0.71$ ✓. No placeholder, no always-zero term.

### (b) Environment / MDP fidelity — PASS with one footnote imprecision
State $s\in\{0,..,15\}$, action $a\in\{0,..,5\}$, horizon 10, $\gamma=0.99$, demand $0.8\,\text{Poisson}(3)+0.2\,\text{Poisson}(10)$ — all match tex lines 154–158 and the config. Demand pmf precomputed as the exact mixture (truncated at 40, renormalized; tail mass negligible), used identically by DP and by sampling. The reward code (lines 107–112) matches the tex footnote (line 161) **on the support that is actually visited**: `inv = min(s+a, 15)`, `5·min(inv,D) − 2a − 1·(inv−D)⁺ − 8·(D−inv)⁺`. See Finding 1 for the `min(·,15)` cap the footnote omits.

### (c) Data integrity — PASS
`compute_data` genuinely trains (10 seeds × 50k episodes) and evaluates (5 configs × 50k episodes) — no hardcoded returns. Every published number reconciles: stdout RESULTS = table.tex exactly; prose 86% = 34.9/40.4, 90% = 36.4/40.4, "−55.3 against −57.5", "36.4 against 34.9", "2.99 and 3.21 against 2.56" all trace to the table. Artifact mtimes (stdout/table/pngs all 08:13:09, cache 08:13:08) postdate the last `.py` edit (06:19:51), so the committed outputs are a fresh regeneration, not a stale-cache republish. Cache key is `CONFIG` (env+iqn+eval+version=4) hashed — invalidates on any hyperparameter change (Finding 4 notes the no-code-hash limitation).

### (d) Comparison fairness — PASS
Within each seed, every policy is evaluated with the **same** RNG seed (`seed+10000`), so all policies see identical initial states and demand sequences; the DP oracle and all IQN variants share one trained-per-seed environment realization. Same 50k episodes, same 10 seeds, held-out eval RNG distinct from training RNG. Apples-to-apples.

### (e) Theoretical sanity — PASS
DP oracle dominates on mean (40.4 > 34.9, 36.4, and the unreported CVaR99/CPT policies which fall to 18–35). No policy beats the oracle. The risk-neutral DP oracle also has the best *tail* (−51.5) because it is simply the better policy overall; the tail claim is therefore correctly restricted to *within* the IQN family (CVaR$_{95}$ eval vs neutral eval), which is the only fair risk contrast. My paired-by-seed recomputation of the tail effect: mean paired CVaR$_{95}$ improvement 2.22, paired SE 1.11, $t\approx2.0$ — a ~2σ effect, consistent with the tex's hedged "improves the average left tail … gap of under two standard errors."

### (f) Information leakage — PASS
IQN sees only `(s,a,r,s',done)` from `env.step`; never the demand pmf, the DP value, or the DP policy. The oracle is computed separately and used only as a baseline row. Model-free throughout; no test-time label access.

### (g) Seed / reproducibility — PASS
N_SEEDS=10 (meets ≥10). `RandomState(seed)`+`torch.manual_seed(seed)` for training, `seed+10000` for eval, `seed+20000` for CPT action noise — all fixed and distinct. SE = std/√10 across seed means, reported in the table (0.0 / 0.8 / 0.4). DP SE rounds to 0.0 (actual ≈0.04 across seeds).

---

## 7-point checklist

1. **Algorithm identity** — PASS. IQN cosine-embedding + quantile-Huber loss correct term-by-term; CVaR = $U([0,\alpha])$, CPT = TK weighting.
2. **Environment fidelity** — PASS (footnote omits the `min(s+a,15)` cap; non-binding on-policy — Finding 1).
3. **Data integrity** — PASS. Real train/eval, all numbers reconcile, artifacts postdate code edit.
4. **Comparison fairness** — PASS. Same eval seed/episodes/env realization across all policies per seed.
5. **Theoretical sanity** — PASS. Oracle dominates; tail claim correctly confined to the IQN family; effect ~2σ.
6. **Information leakage** — PASS. Model-free; no oracle/pmf/label access during learning.
7. **Seed/reproducibility** — PASS. 10 seeds, fixed distinct seeds, SEs reported.

**Diagram-only cap:** does NOT apply. Genuine Monte Carlo experiment.

---

## Findings (severity-ordered)

1. **[Low-Med] Reward footnote ≠ implemented reward for $s+a>15$.** tex line 161 states the reward on the uncapped $(s+a)$, but the code caps `inv = min(s+a, S_MAX=15)` before computing sales/holding/stockout while still charging `2a` on the full order. Under the DP oracle the cap never binds (max on-policy $s+a=11$), and IQN reaches high inventory rarely, so no reported number changes. Still, the stated reward and the coded reward diverge above $s+a=15$. Fix: add `min(s+a,15)` to the footnote, or note the clip.

2. **[Low-Med] "Avg Order … units per period" mislabels the metric.** `avg_orders` (lines 413–422) is a **state-uniform** average of per-state mean orders, with 0-fill for unvisited inventory levels — not a visitation-weighted per-period average. The comparative direction (both IQN > oracle: 2.99, 3.21 vs 2.56) is likely robust because it is computed identically for all three, but the absolute "units per period" reading (tex line 186) is loose; a visitation-weighted per-period figure would differ.

3. **[Low-Med, conceptual] The demonstrated risk effect is weak and slightly muddy.** The CVaR$_{95}$ eval improves BOTH the mean (36.4 vs 34.9) and the tail (−55.3 vs −57.5) over neutral — so the figure is not a clean risk-return tradeoff; it partly reflects the risk-neutral IQN being sub-optimal (an imperfect net whose mean-greedy action is beatable). The footnote's a-priori "risk-averse agents should carry more safety stock" is not clearly borne out: CVaR$_{95}$ orders *less* on average (2.99 vs 3.21) and differs from neutral only at *high* inventory (ordering less, tracking the oracle's 0). The tex prose is carefully hedged and does not overclaim, but an adversarial reviewer would note the effect is ~2σ and the safety-stock mechanism set up in the footnote is not the mechanism the figure shows. The eval is also a one-step distortion on a risk-neutrally-trained net, not a certified CVaR-optimal policy (the tex itself cites Hau2023 that such decompositions are suboptimal).

4. **[Low] Config-only cache key (no code hash).** `CONFIG` hash invalidates on hyperparameter change but not on a logic edit that leaves `CONFIG` fixed; a `--plots-only` regen could then serve stale results. Mtimes here confirm no actual staleness. Repo-wide limitation, not specific to this sim. Fix: bump `version` on any algorithm edit.

5. **[Low] Dead / unreported computation.** `compute_data` evaluates IQN-CVaR99 ($\alpha=0.01$) and IQN-CPT every run (10 seeds × 50k episodes each) but `generate_outputs` reports only DP / Neutral / CVaR95. The two dropped policies are the worst-behaved (CVaR99 mean collapses to 18–35 across seeds). The tex never mentions a CVaR99-policy or CPT-policy row (Prospect Theory at lines 123/147 is general concept prose, not a sim claim), so **there is no overclaim in the compiled tex** — but silently dropping the ugliest configs is a mild cherry-pick smell, and the wasted compute plus dead `sample_cpt_taus` / unused `eval_rng` (line 385) are maintenance smells. Note the docstring says "four risk-sensitive policies," compute evaluates five configs (incl. DP), outputs report three; the tex reports exactly the three, which is the honest set.

**Context points evaluated (as requested):**
(a) Confirmed — tex lines 164–167 describe exactly "one IQN trained with $U([0,1])$, then evaluated under two $\tau$-samplings"; the code trains one net per seed and varies only decision-time $\tau$. No mismatch.
(b) Confirmed — docstring "four" vs five evaluated configs vs three reported is real, but the compiled tex does **not** overclaim: it reports only DP / IQN-Neutral / IQN-CVaR95, and the CVaR99 column in the table is a *metric* on those three policies, not the dropped CVaR99-sampling policy.

---
**Bullshit score: 25%** — Reviewer 2 catches that the reward footnote omits the inventory cap the code applies, that "units per period" mislabels a state-uniform average, and that the CVaR$_{95}$ eval improving both mean and tail (a ~2σ effect, with the risk-averse policy ordering *less* not more) makes the "safety stock" framing awkward; but the IQN loss is exactly Dabney 2018b, the DP oracle re-verifies at 40.44, every reported number reconciles from fresh artifacts, the comparison is fair, and the compiled tex overclaims nothing, so the substance survives revision.
