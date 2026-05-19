# Paper Coherence Audit — 2026-05-20

**Paper:** A Survey of Reinforcement Learning For Economics (Pranjal Rawat, Georgetown)
**Compile entry:** `/Users/pranjal/Code/rl/docs/main.tex`
**Auditor scope:** pre-arxiv coherence (abstract↔conclusion, figure↔claim, method reproducibility)
**Prior audit:** `2026-05-19-paper-coherence.md` (post-Phase-A/B/C cleanup re-audit)
**Out of scope:** `journals/`, `thesis/`, `thesis_v2/`, `ORE_main/`, `archive/`, `tex/backups/`, `tex/v*_archived/`.

Verdict summary:
- Section 1 (Abstract ↔ Conclusion): **PARTIAL** (improved from prior; intro roadmap now fixed; world models now covered)
- Section 2 (Figure ↔ Claim): **MISALIGNED** (one critical regression in `ch08_offline_rl` — table fully stale relative to current caches)
- Section 3 (Method reproducibility): **PARTIAL** (most fixes from prior audit landed; one new critical mismatch in ch08 prose vs code)

Critical-vs-deferred legend:
- **[CRITICAL]** = an arxiv reader / hostile reviewer would flag this on first pass
- **[DEFERRED]** = nit, internal consistency, can wait until journal revision

---

## Changes Since 2026-05-19 — Verification

| Cycle change | Verified? | Notes |
|---|---|---|
| ch99 new World Models paragraph (conclusion §11, paragraph 5) | yes | conclusion.tex:21 explicitly refs Section~\ref{section:world_models} and the cobweb/fishery sim |
| ch08 BC/BCQ→BCQ-D, IQL→IQL-argmax labels | yes | offline_rl.tex:85, 105, 147, 156, 165, 169 all use new labels |
| ch08 four-way identity collapse paragraph | yes (prose), **stale (table)** | offline_rl.tex:158 introduces the paragraph; **but** see CRITICAL issue C2 below |
| ch06_games durable_goods retitled + two subsections | yes | rl_in_games.tex:148 (\subsec{The Coase Conjecture...}), :206 (\subsec{Screening versus Pooling...}); new asymptotic sim with backward induction at rl_in_games.tex:157-204 |
| ch06_games Bertrand FOC corrected | yes | `cournot_bertrand_marl.py:68` reads `(a-bp+ep)+(p-c)(-b)=0 → p* = (a+bc)/(2b-e)`; tex:82 matches the formula and produces p*=4 |
| ch06_games three pure NE on Cournot integer grid named | yes | rl_in_games.tex:82 names $(2,4), (3,3), (4,2)$ |
| ch10b causal_thompson_sampling → context_conditional_thompson_sampling | yes (mostly) | rl_for_ci.tex:226, :244, :291, :329, :333, :345 use CCTS / context-conditional; one residual TS_C reference remains and is intentional (the *other* algorithm, by Bareinboim) |
| ch03_theory td_lambda corridor exponent V*(s) = γ^(18-s) | yes | planning_learning_v3.tex:141 matches td_lambda_corridor_stdout.txt:8 |
| ch05_econ_models nfxp_ccp_td PMLE Theorem 5 footnote | yes | rl_in_se.tex:197 footnote discloses omission of "the locally robust PMLE correction (Theorem~5 of that paper)" |
| ch05 10 seeds + SE columns | yes | rl_in_se.tex:197 ("Each configuration is replicated across 10 seeds"); table has SE columns for RC Bias / RC RMSE / θ1 / θ2 |
| refs.bib trimmed | partial | refs.bib now 469 entries (close to the cited 433); refs_extended.bib confirmed deleted |
| Intro chapter map updated (prior CRITICAL I11) | yes | intro.tex:11 now lists all 17 sections including world_models / dist_robust_constrained / rl_for_ci |
| Stale path `ch11_rl_for_ci` (prior CRITICAL) | yes (fixed) | rl_for_ci.tex:304, :333 now read `ch10b_rl_for_ci` |
| Caption seed count (prior CRITICAL on dtr_qlearning_vs_murphy) | yes (fixed) | rl_for_ci.tex:68 reads "50 Monte Carlo seeds" (tabular) and "20 seeds" (high-dim), matches script |

Three of the five prior CRITICAL items from `2026-05-19-paper-coherence.md` are resolved. The fourth (`tab:offline_main` identity collapse) was attempted but **regressed** — see C2 below. The fifth (Conclusion silent on World Models) is resolved by the new conclusion.tex:21 paragraph.

---

## 1. Abstract ↔ Conclusion alignment

The abstract is unchanged (`ch00_introduction/tex/abstract.tex:1`, one paragraph). The intro chapter map (`ch00_introduction/tex/intro.tex:11`) is now current. The conclusion (`ch99_conclusion/tex/conclusion.tex`) has four subsections: Domain Structure → How RL Advances Applied Modeling (now 5 paragraphs including World Models) → Open Challenges → Conclusion.

| # | Abstract / Intro claim | Matched in conclusion? | Conclusion line | Notes |
|---|------------------------|------------------------|------------------|-------|
| A1 | "(re)introduces RL methods..." | partial | §29-33 | Closing paragraph is synthesis-style, not re-introduction. Same as 2026-05-19. |
| A2 | "curse of dimensionality limits exact DP" | yes | :13 | Tight match. |
| A3 | "RL extends tractability to high-dim states / continuous actions / strategic interactions" | yes | :13, :17 | Strategic interaction covered by independent Q-learning to Nash (Section~\ref{section:rl_games}). |
| A4 | "review the theory connecting classical planning to modern learning" | partial | :33 | Conclusion mentions "shared mathematical foundations (Section subsec:structural_equivalences)" without synthesis. Same as prior. |
| A5 | "simulated examples in pricing, inventory, strategic games, preference elicitation" | partial | :7, :15, :17, :9 | Same as prior; abstract does not enumerate causal / macro / bandits / offline as sim domains. |
| A6 | "brittleness, sample inefficiency, hyperparameter sensitivity, no global convergence" | yes | :25 | Deadly-triad paragraph covers the list. |
| A7 | "reliance on accurate simulators" | yes | :5 | "Most applied domains lack this ingredient." |
| A8 | "when guided by economic structure, RL provides a flexible framework" | yes | :7, §3.1 title | Subsec title "How Domain Structure Improves RL" answers this. |
| A9 | "Companion survey (Rust and Rawat, 2026b)" | no | — | Not restated in conclusion. **[DEFERRED]** — fine to omit. |
| A10 | "All simulation code is publicly available" | no | — | Not standard to restate. **[DEFERRED]** |
| I11 (intro:11) | Chapter roadmap | n/a | — | **(Prior CRITICAL fixed)** Now lists all 17 sections; matches main.tex:146-218 exactly. |

### Unmatched conclusion claims

| Conclusion claim | Where | Coverage in abstract? |
|------------------|-------|------------------------|
| Algorithmic collusion (`Rawat2026collusion`) | :19 footnote | Not in abstract; mentioned only in intro:9. **[DEFERRED]** |
| Multi-agent RL hardness (PPAD-complete Nash) | :27 | Not in abstract. **[DEFERRED]** |
| Lucas critique / causal simulators | :5 | Not in abstract. **[DEFERRED]** |
| World models / cobweb-fishery sims | :21 | **New paragraph this cycle.** Not announced in abstract but the substantive claim ("small learned dynamics model paired with planning can outperform model-free baselines on economic environments...self-referential or exogenous-stochastic") is faithful to the ch12 chapter's cobweb (RLS 5.89, LQ 11.65 vs Q-learning ~1000, GA 90-300) and fishery (RLS 13.67, LQ 14.69 vs Q-learning 274.7, GA 706.13) results. **[DEFERRED]** to update abstract if economical. |
| Knowledge ladder Θ(T) → O(log T) | :7 | Implicit in A8. |

### Section-level coverage gap (in conclusion)

The conclusion now references: applications (:5), causal_rl (:5), bandits (:7), rl_econ_models (:7), offline_rl (:9, :15), rl_games (:17, :27), deeprl_practice (:25), **world_models (:21, NEW)**, structural_equivalences (:33). Still **not referenced** in conclusion:
- `section:rl_macro` (ch06_macro, ~470 lines). **[DEFERRED]** — chapter has its own internal closing synthesis; same as prior.
- `section:rl_for_ci` (ch10b). **[DEFERRED]** — implicitly treated as part of "causal RL" block.
- `section:dist_robust_constrained` (ch11). **[DEFERRED]** — short chapter; still unmentioned in closing.
- `section:rlhf` (ch09). **[DEFERRED]** — conclusion:9 references preference learning under `section:offline_rl` (which is where the RLHF/DPO sim lives) but never `section:rlhf` directly.

### Verdict

**Improved from 2026-05-19.** The prior CRITICAL (intro roadmap stale) is fixed. The prior CRITICAL (conclusion silent on World Models) is fixed. Remaining mismatches are scope/announcement-level, not factual. The new ch12 paragraph (:21) accurately summarizes what the chapter actually reports.

**Verdict: ALIGNED** on substantive claims, **PARTIAL** on completeness (abstract still omits macro / causal / world models as named simulation domains, conclusion still silent on macro / robust / RLHF section).

---

## 2. Figure ↔ Claim support

| Figure / table | Cited at | Claim supported | Notes |
|----------------|----------|------------------|-------|
| `fig:coase_paths` (ch06_games) | rl_in_games.tex:202 | yes | Caption (rl_in_games.tex:181-183) describes price paths at $T \in \{10,50,200\}$ × $\delta \in \{0.5, ..., 0.99\}$. Prose claim "long horizons and patient buyers force the seller to start lower and end near marginal cost" is supported by stdout table: at T=200, δ=0.99: $p_T = 0.0000$, $p_1 = 0.1151$ |
| `fig:coase_collapse` (ch06_games) | rl_in_games.tex:202 | yes | Caption ("$p_T \to 0$ at all $\delta < 1$ once $T$ is large enough"). Stdout at T=200 confirms $p_T = 0.0000$ for δ ∈ {0.5, 0.75, 0.9, 0.95} and $p_T = 0.00004$ for δ=0.99 |
| `tab:coase_dp` (ch06_games) | rl_in_games.tex:200 (`\input{durable_goods_coase_results}`) | yes | Full numerical match: ratio 0.230 at T=200, δ=0.99 ✓; ratio 0.959 at T=2, δ=0.95 → 0.365 at T=200 ✓; ratio 0.828 at T=200, δ=0.50 → 0.230 at T=200, δ=0.99 ✓. Stationary MPE cross-check at δ≤0.95 matches to 5 decimals ✓ |
| `tab:coase` (ch06_games) | rl_in_games.tex:235 (`\input{durable_goods_results}`) | yes | Two-period CFR π-sweep at δ=0.5. Prose "P(Screen) ≈ 0.60 at π = 0.60 and reaches 0.90 only at π = 0.70" matches table rows: π=0.60 → 0.598, π=0.70 → 0.900. The new section structure (Coase asymptotic + Screening 2-period CFR) is internally consistent |
| `tab:cournot_bertrand` (ch06_games) | rl_in_games.tex:84-92 (`cournot_bertrand_results.tex`) | yes | Cournot Nash q*=3, all three algorithms within 0.17 of it; Bertrand p*=4, IQL/Nash-Q exact, WoLF-PHC 3.95±0.05. Stdout matches table exactly. Bertrand FOC formula in tex:82 `(a+bc)/(2b-e) = 4` correctly derived (a=10, b=2, c=1, e=1) |
| Cournot integer-grid 3 NE claim | rl_in_games.tex:82 | yes (mathematically) | Best-response correspondence on $q_i \in \{0..9\}$ at $q_j=2$: payoffs at $q_i \in \{2,3,4\}$ are $\{8, 9, 8\}$; correction: BR(2) = 3, BR(3) = 3, BR(4) = 2 or 3 (tied). With ties at the BR correspondence, $(2,4), (3,3), (4,2)$ are mutual best responses on the integer grid. Footnote at rl_in_games.tex:82 explains that Nash-Q picks (3,3) under joint-payoff-max tie-breaking. **[DEFERRED]** would be cleaner to write "$(3,3)$ is the unique symmetric NE; $(2,4)$ and $(4,2)$ arise from integer-grid BR ties," but the existing claim stands |
| `tab:offline_main` (ch08_offline_rl) | offline_rl.tex:149-154 (`offline_rl_pricing_results.tex`) | **no** | **[CRITICAL — C2 below]** Severe mismatch between the rendered table and the current cache contents. See below for full numbers. |
| `fig:offline_coverage` (ch08_offline_rl) | offline_rl.tex:165-169 | partial | The PNG file mtime is `May 19 03:13:17` — same timestamp as `offline_rl_pricing_results.tex` (the stale Phase-1 artifact). The coverage figure currently displayed in the paper corresponds to the same stale run as the main table. The prose claim ("BCQ-D collapses at $\epsilon_b = 0.9$ because a nearly uniform behavioral policy renders the action constraint vacuous") is Phase-1-flavored; under Phase 2 (regime-dependent BEHAVIORAL_MARKUPS = [5,7,8,9]) the behavioral is no longer concentrated on $p=10$, so the original mechanism ("action constraint vacuous") may not apply in the same form |
| `tab:offline_main` four-way identity collapse paragraph | offline_rl.tex:158 | **no** | **[CRITICAL — C2 below]** Paragraph claims BC/BCQ-D/DT/RvS all "report $169.27 \pm 0.60$, identical to four decimal places" and ascribes the collapse to "85\% of the dataset mass on the single action $p = 10$." Current code uses regime-dependent preferred prices [5,7,8,9], not uniform $p=10$. Current cache shows BC=186.28, BCQ-D=177.05, DT=185.27, RvS=186.58 — NOT identical |
| `fig:td_lambda_corridor` and `tab:td_lambda_corridor` (ch03_theory) | planning_learning_v3.tex:149-152 | yes | Stdout shows RMSVE = 0.0000±0.0000 at λ=1.0 across 20 seeds; table tex matches (`0.0000 ± 0.0000`). Prose claim "TD($\lambda = 1$) reaches RMSVE < 0.05 in fewer episodes than TD(0)" supported by "Episodes to RMSVE < 0.05" col (λ=1.0: 52±0 episodes; λ=0: >200 episodes). True value formula `V*(s) = γ^(18-s)` matches stdout:8 |
| `tab:ddc_estimation` (ch05_econ_models) | rl_in_se.tex:204 (`nfxp_ccp_td_results.tex`) | yes | "Seeds per cell: 10" in stdout matches "10 seeds" in tex (rl_in_se.tex:197). All numbers (RC bias, RC RMSE, θ1, θ2 with adjacent SE columns) match stdout exactly. The footnote at rl_in_se.tex:197 explicitly discloses omitting "Theorem 5 of that paper, the construction that yields $\sqrt{n}$-consistency" |
| `fig:ddc_scaling_time` (ch05_econ_models) | rl_in_se.tex:209-210 | yes | Caption ("means over 10 seeds") matches stdout. Wall-clock claim "NFXP wall-clock grows by roughly three orders of magnitude across the four scales" (rl_in_se.tex:214) is supported: NFXP 0.23s (K=1) → 163.51s (K=4), a factor of 710× ≈ 2.85 orders of magnitude ✓ |
| `fig:figure:fc_dyna_maze` (ch12) | s03_dyna_q.tex:60 | yes | Same as 2026-05-19 — 30 seeds match, prose claim "tabular Dyna-Q at K=50 delivering an order-of-magnitude improvement over K=0" supported (52.0/3.5 ≈ 15×) |
| `figure:fc_cobweb_curves` and `table:fc_cobweb_results` (ch12) | s09_dual_sim.tex:26-37 | yes | All numerical claims match stdout: MBPO 656.6/112.1/48.9; RLS ≈ 5 units; LQ 11.65 to 42.90; GA 92 to 309. 20-seed claim matches |
| `figure:fc_fishery_curves` and `table:fc_fishery_results` (ch12) | s09_dual_sim.tex:73-87 | yes | RLS 13.67, LQ 14.69, Q-learning 274.71, Naive 447.35, GA 706.13 — all match prose and stdout to integer precision |
| Conclusion world-models paragraph (ch12 sim claims) | conclusion.tex:21 | yes | Cobweb claim "small learned dynamics model paired with planning can outperform model-free baselines" supported by cobweb regret table (RLS 5.89, LQ 11.65 vs Q-learning ~1000, GA 90-300) and fishery (RLS 13.67, LQ 14.69 vs Q-learning 274.71, GA 706.13). "Self-referential or exogenous-stochastic" is faithful to the chapter's framing (s09:4 explicitly contrasts cobweb as "self-referential" and fishery as "exogenous"). "Value-aware losses give task-relevant accuracy where it matters for decisions but offer weaker guarantees against policy drift" is faithful to s06_value_aware.tex:20 (Voelcker2025 calibration footnote) |
| `fig:simB2` / `tab:simB2` (ch10b causal_bandit) | rl_for_ci.tex:337, :345 | yes | The CCTS rename is consistent throughout the prose at rl_for_ci.tex:226, :244, :329, :333. Numerical claim "CCTS accumulates only $0.66$" and "full $\mathrm{TS}_C$ accumulates $4.49$" present in prose; needs to be re-verified against `causal_bandit_results.tex` and the figure subpanel (c) — not done here due to scope, but the rename consistency is clean |
| `tab:dtr_qlearning_vs_murphy` (ch10b) | rl_for_ci.tex:68 | yes | **(Prior CRITICAL fixed)** Caption reads "50 Monte Carlo seeds" (tabular), "20 seeds" (high-dim) — matches `dtr_qlearning_vs_murphy.py:76` (`N_SEEDS = 50`) and `:310` (`N_SEEDS_HD = 20`) |

### Critical issue C2 — ch08 `tab:offline_main` and surrounding prose are stale

**Setup:** The simulation script `offline_rl_pricing.py` was rewritten on 2026-05-19 at 04:06:31 to change the behavioral policy from "always $p=10$" (Phase 1) to "regime-dependent preferred prices [5, 7, 8, 9]" (Phase 2). Comment block at `offline_rl_pricing.py:71-84` documents this change explicitly. Caches were re-saved at 04:07–04:25.

**Mismatch:**

(a) `offline_rl_pricing_results.tex` mtime is 03:13:17 (53 minutes **before** the script was edited). The table rendered in the paper is the pre-Phase-2 artifact.

(b) Cache contents under current `CONFIG_VERSION = 14` (read via pickle):

| Method | Current cache mean | Current cache SE | Current % of optimal | Table.tex shows |
|--------|--------------------|-------------------|------------------------|-----------------|
| DP Oracle | 192.41 | 0.33 | 100.0% | 192.41 ± 0.33 (100.0%) ✓ |
| BC | 186.28 | 0.31 | 96.8% | **169.27 ± 0.60 (88.0%)** ✗ |
| BCQ-D | 177.05 | 0.73 | 92.0% | **169.27 ± 0.60 (88.0%)** ✗ |
| DT | 185.27 | 0.33 | 96.3% | **169.27 ± 0.60 (88.0%)** ✗ |
| RvS | 186.58 | 0.34 | 97.0% | **169.27 ± 0.60 (88.0%)** ✗ |
| CQL | 178.08 | 1.48 | 92.6% | 176.73 ± 1.13 (91.9%) ≈ |
| IQL-argmax | 176.67 | 0.81 | 91.8% | 176.98 ± 0.56 (92.0%) ≈ |
| FQI | 47.48 | 8.42 | 24.7% | **156.18 ± 1.68 (81.2%)** ✗ |

The four-way collapse (BC = BCQ-D = DT = RvS = 169.27) is **gone** in Phase 2. The current data shows BC and RvS leading at ~97% of optimal, DT close behind at 96.3%, BCQ-D at 92%, CQL at 92.6%, IQL-argmax at 91.8%, and FQI catastrophically failing at 24.7% (versus 81.2% reported).

(c) **The four-way collapse paragraph** at `offline_rl.tex:158` (the central prose addition in this cycle) is built on numbers that the current code does not produce. Quote: "Four of the trained methods, BC, BCQ-D, DT, and RvS, all report $169.27 \pm 0.60$, identical to four decimal places (Table~\ref{tab:offline_main}). The coincidence is not numerical accident but a property of the behavioral distribution. With 85\% of the dataset mass on the single action $p = 10$..." The 85%-on-$p=10$ behavioral was the Phase 1 design and was explicitly replaced in Phase 2.

(d) Several other prose claims at `offline_rl.tex:147-169` are also Phase-1-anchored:
- offline_rl.tex:147: "the behavioral policy represents a conservative pricing team that always sets the maximum price ($p = 10$) regardless of demand regime, inventory, or time remaining, with probability 0.85" — Phase 1 only.
- offline_rl.tex:156: "FQI achieves 81.2\% of the DP optimal, substantially below the behavioral cloning baseline of 88.0\%" — both numbers Phase 1; under Phase 2, FQI is 24.7% and BC is 96.8%.
- offline_rl.tex:160-161: "CQL and IQL-argmax both exceed the behavioral baseline, achieving 91.9\% and 92.0\% respectively" — under Phase 2, CQL=92.6% and IQL=91.8%, but they no longer "exceed" BC (which is now 96.8%); the pessimism narrative is partially inverted because BC dominates on the regime-dependent behavioral.
- offline_rl.tex:160: "BCQ-D matching BC exactly" — no longer the case; BCQ-D=92.0%, BC=96.8% in Phase 2.

(e) The coverage figure (`offline_rl_pricing_coverage.png`, mtime 03:13:17) is also stale relative to current caches. Its supporting prose at offline_rl.tex:169 was reasoned about Phase 1 mechanisms ("uniform behavioral renders the action constraint vacuous") that may not hold under the regime-dependent Phase 2 behavioral.

**Why this matters:** The four-way identity collapse paragraph is the central methodological-self-reflection contribution of this cycle's offline_rl edits — it owns the prior critical artifact and turns it into a teaching moment about supervised-conditioning offline methods. The narrative is correct as a textbook claim and is well-written. But the data the prose claims to be describing does not exist in the current cache. Either:

1. Re-run `python3 ch08_offline_rl/sims/offline_rl_pricing.py --plots-only` (or with `--force` to invalidate Phase-1 caches), which under Phase 2 will produce a different table where the four-way collapse does **not** appear. The prose must then be substantially rewritten because its central claim is now false.

2. Roll back the Phase 2 edit to the script (set `BEHAVIORAL_MARKUPS = [10, 10, 10, 10]`) and re-run. This will reproduce the 169.27 four-way collapse and keep the prose accurate.

3. Decide whether the Phase 2 behavioral design (regime-dependent) is what the chapter wants pedagogically. If yes, re-run and rewrite. If no (the four-way collapse is the pedagogical point), revert the .py.

Either way, the published table must match the script that produced it, and the prose must match the table.

### Verdict

**Verdict: MISALIGNED** — one critical mismatch in `ch08_offline_rl` between the prose, the table, the figure, and the script. Everything else in the focus list (ch06_games durable_goods, ch03_theory td_lambda, ch05_econ_models nfxp_ccp_td, ch12 world models conclusion paragraph, ch10b causal_bandit rename) is internally consistent and numerically faithful.

---

## 3. Method reproducibility

| Method / experiment | S/A/r? | Hyperparams? | Seeds? | Eval protocol? | Missing / stale |
|---------------------|--------|---------------|--------|------------------|-----------------|
| `durable_goods_coase.py` (ch06_games, NEW this cycle) | yes (rl_in_games.tex:159-176) | yes (T grid {2,5,10,20,50,100,200}; δ grid {0.5,0.75,0.9,0.95,0.99}; $c=0$; uniform F on [0,1]; rl_in_games.tex:173 footnote spells the closed-form recursion) | n/a (deterministic DP — no seeds needed; backward induction) | yes (commitment vs no-commitment ratio; stationary-MPE sanity check at δ≤0.95) | none. The new asymptotic Coase simulation is fully reproducible from the paper alone. The footnote at rl_in_games.tex:173 gives the closed-form scalar recursion and terminal values; backward induction over T=200 runs in under a millisecond per the stdout. **Exemplary** |
| `durable_goods_monopoly.py` (ch06_games, two-period CFR) | yes (rl_in_games.tex:211-225) | yes ($v_L=100, v_H=200, T=2$; 5000 CFR iterations; Gaussian noise σ=0.05 on initial regrets; 10 seeds; π-sweep at δ=0.5, δ-sweep at π=0.7) | yes (10 seeds, per rl_in_games.tex:229 footnote) | yes (P(Screen), NashConv) | none |
| `cournot_bertrand_marl.py` (ch06_games) | yes (rl_in_games.tex:82) | yes (Cournot a=10, c=1, Q_max=9; Bertrand a=10, b=2, e=1, c=1, P_max=9; 50,000 iterations; final 5,000 for averaging) | yes (20 seeds, rl_in_games.tex:82 footnote) | yes (mean action, profit, |a-a*|) | **[DEFERRED]** rl_in_games.tex:82 footnote mentions "Nash-Q implementation here selects the joint-payoff-maximizing equilibrium when multiple pure Nash exist, a deviation from the canonical Hu-Wellman 2003 backup" — the deviation is disclosed, good. The Cournot 3-NE characterization could be tightened ("$(2,4)$ and $(4,2)$ arise from integer-grid best-response ties at $q_j = 2$") but this is editorial **[DEFERRED]** |
| `offline_rl_pricing.py` (ch08) | partial | **stale prose** | yes (20 seeds in code) | partial | **[CRITICAL — C2 above]** The prose at offline_rl.tex:147 still describes the Phase-1 behavioral ("always $p=10$ with probability 0.85"). The code at offline_rl_pricing.py:82 has `BEHAVIORAL_MARKUPS = [5, 7, 8, 9]` (Phase 2). Until prose and code are reconciled, the simulation is not reproducible in the sense that running the script produces numbers that do not match what the paper claims to be reporting |
| `td_lambda_corridor.py` (ch03_theory) | yes (planning_learning_v3.tex:139-141) | yes (20-state corridor, γ=0.99, α=0.05, 200 episodes, λ ∈ {0, 0.4, 0.8, 1.0}) | yes (20 seeds) | yes (RMSVE vs true V*(s) = γ^(18-s)) | none. RMSVE = 0.0000±0.0000 at λ=1.0 is exact: with γ=0.99 and complete eligibility trace, MC return at terminal is sampled exactly once per episode and converges to V* at all states. Match |
| `nfxp_ccp_td.py` (ch05_econ_models) | yes (rl_in_se.tex:197) | yes (4 methods × 4 component counts K∈{1,2,3,4} × state spaces 20 → 160,000; N=500 agents × T=100 periods panel; PyTorch optimizer seeded) | yes (10 seeds, rl_in_se.tex:197) | yes (RC bias, RC RMSE, θ1 RMSE, θ2 RMSE with seed-level SE) | none. The footnote disclosure of omitting Theorem 5 PMLE correction is now explicit (rl_in_se.tex:197 footnote) so a reader knows the $\sqrt{n}$-consistency guarantee does not transfer to the reported point estimates |
| `dyna_maze.py` (ch12) | yes (s03_dyna_q.tex:53-57) | yes (α=0.1, ε=0.1, γ=0.95, episode cap 200, 30 seeds, MLP 32 hidden, lr 3e-3, plan-interval 10, 16 imagined trajectories of length 10) | yes (30 seeds) | yes (cumulative reward at t=3000) | none |
| `cobweb_paradigms.py` (ch12) | yes (s09_dual_sim.tex:15) | yes (b/c sweep {0.5, 1.0, 2.0}, a=4, c=1, φ=0.2, γ=0.95, T=500; Q-learning 20×20×25 grid, α=0.1, ε decay 0.3→0.01; MBPO ensemble size 5, rollout horizon 5, 10 rollouts per real step) | yes (20 seeds, captions match) | yes (cumulative regret vs oracle on shared noise; parameter recovery; policy distance) | none |
| `fishery_paradigms.py` (ch12) | yes (s09_dual_sim.tex:67-70) | yes (r=0.4, K=10, p=2.0, c=0.2, σ=0.3, γ=0.95, T=500, oracle grid 50×25; Q-learn 30×21, α=0.1, ε decay 0.3→0.01) | yes (20 seeds) | yes (cumulative regret) | none |
| `dtr_qlearning_vs_murphy.py` (ch10b) | yes (rl_for_ci.tex:55) | yes (rl_for_ci.tex:55 footnote: tabular 50 seeds, HD 20 seeds, 64-hidden MLP, 200-epoch FQI, 8000-step DQN) | yes (50 tabular / 20 HD) | yes (V(π̂)/V*) | none. **(Prior CRITICAL caption fixed)** |
| `causal_bandit_parallel.py` (ch10b) | yes (rl_for_ci.tex:317-333) | yes (greedy casino with 2 contexts × 2 arms; T=1000; CCTS Beta prior; full TS_C with consistency-axiom seeding c=0.5 and RDC clip(0.01, 1)) | partial (rl_for_ci.tex:345 says "500 MC replications for (c)") | yes (cumulative regret) | The rename from `causal_thompson_sampling` to `context_conditional_thompson_sampling` is consistent throughout the prose. The footnote at rl_for_ci.tex:329 documents the fractional pseudo-count c=0.5 design choice and that it is disclosed-not-optimized |
| `dynamic_dml_snmm.py` (ch10b) | yes (rl_for_ci.tex:295-302) | yes (p=20, sparsity s=5, σ_η=0.6, ‖B‖_op=0.5, ψ_1*=1.0, ψ_2*=0.5, K=5 folds, Lasso/L1-logistic learners) | yes (200 MC reps) | yes (bias / RMSE / coverage) | **(Prior CRITICAL stale-path fixed)** rl_for_ci.tex:304 now reads `ch10b_rl_for_ci/sims/dynamic_dml_snmm.py`. Stdout file paths inside `dynamic_dml_snmm_stdout.txt:39-41` may still reference the old folder name; cosmetic. **[DEFERRED]** |
| `confounded_ope.py` and `counterfactual_ope.py` (ch10_causal) | yes | yes | yes (20 seeds each) | yes | Same as 2026-05-19 — `confounded_ope_stdout.txt` "loaded from cache" (recommend `--force` once before submission); `counterfactual_ope.py:2` header comment still attributes to "Chapter 12 (forecasting)" rather than ch10. **[DEFERRED]** |
| `rbc_dp_vs_drl.py` and `lq_mfg.py` (ch06_macro) | yes | yes | yes | yes | Same as 2026-05-19 — no changes in this cycle. `lq_mfg.py` external MFAX dependency note still **[DEFERRED]** |

### Verdict

**Verdict: PARTIAL.** All four cycle changes that touched reproducibility (ch06_games Coase asymptotic sim, ch03_theory exponent fix, ch05_econ_models PMLE footnote + 10-seed SE columns, ch10b rename) are now fully reproducible from the paper alone. The new ch06_games asymptotic Coase sim (durable_goods_coase.py) is exemplary: deterministic backward induction, closed-form scalar recursion in the footnote, two sanity checks in the stdout. One CRITICAL regression in ch08 offline_rl — the script was edited but the table, figure, prose, and prior-cycle four-way collapse paragraph were all written against the pre-edit state.

---

## Summary of critical issues (in priority order)

1. **[CRITICAL — C2] ch08 `tab:offline_main` is stale relative to current caches and prose.** Phase 2 cache: BC=186.28 (96.8%), BCQ-D=177.05 (92.0%), DT=185.27 (96.3%), RvS=186.58 (97.0%), CQL=178.08 (92.6%), IQL-argmax=176.67 (91.8%), FQI=47.48 (24.7%). Table.tex shows the Phase-1 numbers (BC=BCQ-D=DT=RvS=169.27, FQI=156.18). The four-way identity collapse paragraph at offline_rl.tex:158 — the central narrative addition this cycle — references numbers that no longer exist in the cache. The behavioral description at offline_rl.tex:147 ("always $p=10$ with probability 0.85") contradicts `offline_rl_pricing.py:82` (`BEHAVIORAL_MARKUPS = [5, 7, 8, 9]`). Fix path: decide which behavioral design the chapter wants, run `--plots-only` (Phase 2) or revert `BEHAVIORAL_MARKUPS` to all-10 (Phase 1), then rewrite or restore prose to match. The arxiv reader cannot reproduce the published table from the published script.

2. **[CRITICAL — C2 follow-on] `fig:offline_coverage` (offline_rl_pricing_coverage.png)** mtime 03:13:17 is the same Phase-1 artifact. Caption and surrounding prose at offline_rl.tex:165-169 reason about Phase-1 mechanisms ("nearly uniform behavioral renders the action constraint vacuous") that may not transfer to Phase 2's regime-dependent behavioral. Must be regenerated in lockstep with the main table.

3. (No third CRITICAL.) The other prior CRITICAL items (intro chapter map, stale `ch11_rl_for_ci` path in footnote, caption seed-count mismatch, conclusion silent on World Models) are all resolved.

## Summary of deferred issues

- Cournot 3-NE statement at rl_in_games.tex:82 is technically correct but would benefit from a half-sentence noting that $(2,4)$ and $(4,2)$ arise from best-response ties on the integer grid (BR to $q_j=2$ is $\{3,4\}$ on the integer grid; BR to $q_j=4$ is $\{2,3\}$).
- Abstract still does not announce causal inference / macro / robust-constrained / world models even though four full chapters are devoted to them.
- Conclusion still silent on `section:rl_macro`, `section:rl_for_ci`, `section:dist_robust_constrained`, and `section:rlhf` as named sections (though their content is partially covered under other rubrics).
- `confounded_ope_stdout.txt` is "loaded from cache" output rather than a fresh run; recommend `--force` once before arxiv submission.
- `counterfactual_ope.py:2` header comment still attributes to "Chapter 12 (forecasting and reinforcement learning)"; cosmetic, does not appear in PDF.
- `dynamic_dml_snmm_stdout.txt:39-41` may still reference `ch11_rl_for_ci/` paths internally; cosmetic.
- refs.bib trimmed from 811 to 469 entries this cycle (close to the cited 433 target). Recommend confirming all `\cite{...}` keys still resolve via a `bibtex main` pass on the master document before submission.
