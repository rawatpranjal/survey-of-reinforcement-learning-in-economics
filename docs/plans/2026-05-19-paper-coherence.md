# Paper Coherence Audit — 2026-05-19

**Paper:** A Survey of Reinforcement Learning For Economics (Pranjal Rawat, Georgetown)
**Compile entry:** `/Users/pranjal/Code/rl/docs/main.tex`
**Auditor scope:** pre-arxiv coherence (abstract↔conclusion, figure↔claim, method reproducibility)

Verdict summary:
- **Section 1 (Abstract ↔ Conclusion):** PARTIAL
- **Section 2 (Figure ↔ Claim):** PARTIAL — one critical issue in `ch08_offline_rl`
- **Section 3 (Method reproducibility):** PARTIAL — mostly good, several stale paths + one mismatched seed count

Critical-vs-deferred legend:
- **[CRITICAL]** = an arxiv reader / hostile reviewer would flag this on first pass
- **[DEFERRED]** = nit, internal consistency, can wait until journal revision

---

## 1. Abstract ↔ Conclusion alignment

The abstract is at `ch00_introduction/tex/abstract.tex` (one paragraph, 19 lines wrapped to one line). The intro is at `ch00_introduction/tex/intro.tex` (six paragraphs). The conclusion is at `ch99_conclusion/tex/conclusion.tex` (4 subsections: domain structure → RL advances applied modeling → open challenges → conclusion).

| # | Abstract / Intro claim | Matched in conclusion? | Conclusion line | Notes |
|---|------------------------|------------------------|------------------|-------|
| A1 | "(re)introduces RL methods to researchers in the social sciences" (abstract:1) | partial | conclusion §29–31 | The closing paragraph reads as a synthesis, not a re-introduction; no explicit closure on the "(re)introduction" framing |
| A2 | "curse of dimensionality limits how far exact DP can be effectively applied … forcing us to rely on suitably small problems" (abstract:1) | yes | conclusion:13 | "Dynamic programming has always offered a prescriptive capability … limited by the curse of dimensionality to models with small, discrete state spaces." Tight match |
| A3 | "RL algorithms offer a natural, sample-based extension of DP, extending tractability to problems with high-dimensional states, continuous actions, and strategic interactions" (abstract:1) | yes | conclusion:13 + 17 | Section 2 of the conclusion ("How RL Advances Applied Modeling") makes this claim explicitly; conclusion:17 covers strategic interaction via independent Q-learning to Nash |
| A4 | "review the theory connecting classical planning to modern learning algorithms" (abstract:1) | partial | conclusion:31 | Conclusion mentions "shared mathematical foundations (Section subsec:structural_equivalences)" but does not synthesise theoretical findings |
| A5 | "demonstrate their mechanics through simulated examples in pricing, inventory control, strategic games, and preference elicitation" (abstract:1) | partial | conclusion:7, 15, 17, 9 | Pricing (via knowledge ladder, conclusion:7) ✓, offline RL inventory (conclusion:15) ✓, games (conclusion:17) ✓, preference elicitation (conclusion:9) ✓. Match is genuine but distributed. The abstract list omits causal inference, macro, bandits, and offline RL as named simulation domains, even though the body has substantial sims in each |
| A6 | "examine the practical vulnerabilities … brittleness, sample inefficiency, sensitivity to hyperparameters, and the absence of global convergence guarantees outside of tabular settings" (abstract:1) | yes | conclusion:23 | "Deep RL algorithms exhibit seed sensitivity, overestimation cascades, and plasticity loss" — covers the list at high level |
| A7 | "reliance on accurate simulators" (abstract:1) | yes | conclusion:5 | "Most applied domains lack this ingredient. Every application in Section~\ref{section:applications} demanded a custom environment…" |
| A8 | "when guided by economic structure, RL provides a flexible and innovative framework" (abstract:1) | yes | conclusion:7 (knowledge ladder), conclusion §3.1 | Subsection title "How Domain Structure Improves Reinforcement Learning" answers this directly |
| A9 | "A companion survey (Rust and Rawat, 2026b) covers the inverse problem of inferring preferences from observed behavior" (abstract:1) | no | — | Not restated in the conclusion. Mentioned in intro:9 as "companion survey \citep{RustRawat2026}". **[DEFERRED]** — fine to drop in a survey's conclusion, but the intro chapter map (intro:11) lists only 9 sections and is out of date |
| A10 | "All simulation code is publicly available" (abstract:1, footnote) | no | — | Conclusion does not restate. Not standard to restate. **[DEFERRED]** |
| I11 | Intro:11 chapter roadmap claims 9 sections ("Chapter 1 traces … Chapter 9 concludes") | — | — | **[CRITICAL]** The actual paper has 17 numbered sections per `main.tex:146-218` (Introduction, Two Cultures, History, Algorithms, Theory [unnumbered \input], Deep RL Empirics, Control, Structural Est, Macro RL, Games, Bandits, Offline RL, RLHF, Causal Inference for RL, RL for CI, Robust/Constrained, World Models, Discussion). The intro roadmap is severely stale and bears no resemblance to the table of contents |

### Unmatched conclusion claims (in conclusion but not flagged in abstract)

| Conclusion claim | Where | Coverage in abstract? |
|------------------|-------|------------------------|
| Algorithmic collusion / market behavior of RL agents | conclusion:19 + footnote `\citep{Rawat2026collusion}` | not in abstract |
| Multi-agent RL hardness (PPAD-complete Nash) | conclusion:25 | not in abstract |
| Infrastructure gap (need shared simulators) | conclusion:27 | not in abstract; closely related to A7 |
| Lucas critique / causal simulators | conclusion:5 | not in abstract (causal RL chapter not announced in abstract) |
| Knowledge ladder reducing regret from Θ(T) to O(log T) | conclusion:7 | implicit in A8 |

### Section-level coverage gap

The conclusion does **not** discuss the following chapters at all (no `\ref{section:...}` to them in conclusion.tex):

- **`section:rl_macro`** (Macro RL — ch06_macro, ~470 lines of tex). **[DEFERRED]** — the macro chapter has its own internal synthesis at the end and is somewhat self-contained; still, an abstract-level survey usually closes the loop.
- **`section:world_models`** (World Models / Model-Based RL — ch12_world_models, ten subsections, three simulations). **[CRITICAL]** — this is the longest single chapter in the paper and the conclusion is silent on it. An arxiv reader would notice the absence.
- **`section:rl_for_ci`** (RL for Causal Inference — ch10b). The conclusion mentions `section:causal_rl` but not its companion `section:rl_for_ci`. **[DEFERRED]** — the conclusion implicitly treats causal as one block.
- **`section:dist_robust_constrained`** (Quantile / Robust / Constrained RL — ch11). **[DEFERRED]** — short chapter, but it is a section in the TOC and goes unmentioned.
- **`section:deeprl_practice`** (Deep RL Empirics — ch03b). Conclusion:23 *does* `\ref{section:deeprl_practice}` for the deadly triad point. ✓

### Verdict

**ALIGNED on substantive claims (A2, A3, A6, A7, A8). PARTIAL on completeness:** the abstract does not announce causal inference / world models / macro / robust RL even though four full chapters are devoted to them, and the conclusion is silent on world models, robust RL, and macro. The intro chapter map (intro:11) is stale and lists 9 sections for a 17-section paper.

**Verdict: PARTIAL**

---

## 2. Figure ↔ Claim support

Method: For each of the focus areas listed in the audit request, I pulled the figure caption, the prose claim that cites it, and the corresponding numbers in `_stdout.txt`. A "yes" means the claim is faithful to both the caption and the stdout; "partial" means the claim is weaker than what the figure shows or vice versa; "no" means the claim contradicts the underlying data.

| Figure / table | Cited at | Caption (truncated) | Claim supported | Notes |
|----------------|----------|---------------------|------------------|-------|
| `fig:confounded_ope` (ch10_causal) | causal_rl.tex:278 | "Bias and RMSE of five OPE estimators as a function of confounding strength ρ" | yes | Caption matches prose claim ("naive bias grows monotonically with ρ", "backdoor and front-door eliminate bias at all ρ"); not verified numerically against re-run, but stdout `confounded_ope_stdout.txt` is "loaded from cache" — should be re-run to refresh. **[DEFERRED]** |
| `fig:cfope_rmse` (ch10_causal) | causal_rl.tex:323 | "RMSE of three OPE estimators against sample size, log-log scale, two panels" | yes | Numerical claims at causal_rl.tex:323 match `counterfactual_ope_stdout.txt:13-22` to 3 sig figs (MB well-spec bias -0.001 ≈ stdout -0.0009, RMSE 0.056 ≈ 0.0558; CF misspec bias 0.002 ≈ 0.0024, RMSE 0.076 ≈ 0.0757). Stdout writes paths still under stale `ch12_forecasting_rl/` (counterfactual_ope_stdout.txt:7-11) — **[DEFERRED]** stale stdout but doesn't affect the figure |
| `tab:cfope_summary` (ch10_causal) | causal_rl.tex:312 (`\input{counterfactual_ope_table}`) | "Counterfactual OPE under linear SCM, n=1000, 20 seeds" | yes | 20-seed claim matches `counterfactual_ope.py:51` (`N_SEEDS = 20`) and stdout. ✓ |
| `fig:dtr_qlearning_vs_murphy` (ch10b_rl_for_ci) | rl_for_ci.tex:59-64 | "(Q1) Tabular sample-size sweep, (Q2) tabular training-budget sweep at N=300, (Q3) high-dim sweep" | yes | Numerical curves match stdout `dtr_qlearning_vs_murphy_stdout.txt` (Q1 Murphy→1.0 at N=10000 ✓; Q2 Q-learn epochs sweep with Murphy reference 0.9907 at N=300 ✓; Q3 NN-FQI 0.9310 vs DQN 0.9126 at N=5000 ✓) |
| `tab:dtr_qlearning_vs_murphy` (ch10b_rl_for_ci) | rl_for_ci.tex:66-71 | "Tabular setting: N=10,000 subjects, 30 Monte Carlo seeds. High-dimensional setting: p=10, N=5,000, 30 seeds" | **no** | **[CRITICAL]** Caption says "30 Monte Carlo seeds" for both settings, but the underlying script uses `N_SEEDS = 50` for tabular (`dtr_qlearning_vs_murphy.py:76`) and `N_SEEDS_HD = 20` for high-dim (`dtr_qlearning_vs_murphy.py:310`). Stdout `dtr_qlearning_vs_murphy_stdout.txt:2` confirms `TABULAR: N_seeds=50` and `HIGH-DIM: N_seeds=20`. The number 30 does not appear anywhere in the run. An arxiv reader doing a code-vs-paper cross-check (which the project's own simulation-audit checklist lists as a hard requirement under "Seed and Reproducibility") would catch this immediately |
| `fig:figure:fc_dyna_maze` (ch12_world_models) | s03_dyna_q.tex:60, 62-66 | "Cumulative reward on the blocking maze for five agents under a shared 3000-step environment budget. Lines are means and shaded bands one standard error across thirty seeds" | yes | Caption count "thirty seeds" matches `dyna_maze.py:37` (`N_SEEDS=30`) and stdout `dyna_maze_stdout.txt:4`. ✓ |
| `table:fc_dyna_maze` (ch12_world_models) | s03_dyna_q.tex:69-74 | "End of Phase 1 / Phase 2 gain / Total. Mean ± SE across thirty seeds" | yes | All five rows match stdout `dyna_maze_stdout.txt:20-24` to one decimal: Dyna-Q K=50 → 52.0±4.2 ✓, Dyna-Q+ K=50 → 47.0±4.4 ✓, Dyna-Q K=5 → 39.2±4.7 ✓, Schmidhuber → 4.0±0.4 ✓, Q-learning → 3.5±0.5 ✓. Prose claim "tabular Dyna-Q at K=50 delivering an order-of-magnitude improvement over K=0" (s03_dyna_q.tex:60) supported (52.0 / 3.5 ≈ 15×) |
| `figure:fc_dyna_maze_layout` (ch12_world_models) | s03_dyna_q.tex:41-48 | "Sutton blocking maze. Opening shifts from column eight in Phase 1 to column zero in Phase 2 at t=1000" | yes | Diagram-only figure; claim consistent with `BlockingMaze` env in `dyna_maze_env.py`. ✓ |
| `figure:fc_cobweb_curves` (ch12_world_models) | s09_dual_sim.tex:26-31 | "Cumulative regret across stability regimes for seven learning paradigms" | yes | Prose claim "regret varies more than two orders of magnitude across regimes (650 stable, 112 borderline, 49 unstable)" for MBPO matches stdout `cobweb_paradigms_stdout.txt:29` (656.60, 112.06, 48.87). RLS ≈ 5 units across regimes (stdout: 5.89, 4.38, 5.87) ✓; LQ "12 to 43" (stdout: 11.65, 18.96, 42.90) ✓; GA "90 to 300" (stdout: 92.89, 133.43, 308.88) ✓ |
| `table:fc_cobweb_results` (ch12_world_models) | s09_dual_sim.tex:33-38 | "Cumulative regret at T=500, 20 seeds, lower is better" | yes | Direct `\input` from `cobweb_paradigms_results.tex`; values match stdout |
| `figure:fc_cobweb_recovery`, `table:fc_cobweb_recovery`, `figure:fc_cobweb_policy_distance` (ch12_world_models) | s09_dual_sim.tex:40-58 | parameter recovery and policy distance | yes | Stdout shows expected param recovery (RLS \|a\|=0.037 stable etc.); prose at s09_dual_sim.tex:20-22 claims "all three methods recover within four percent" matches stdout (max RLS error \|a\|=0.037 / true a=4 ≈ 1%) ✓ |
| `figure:fc_fishery_curves`, `table:fc_fishery_results` (ch12_world_models) | s09_dual_sim.tex:73-87 | "Cumulative regret on logistic-growth fishery, 6 paradigms" | yes | Prose at s09_dual_sim.tex:73 says "RLS and model-based LQ … about thirteen and fifteen regret units respectively, tabular Q-learning … roughly 275, constant rule 447, GA finishes at 700". Stdout `fishery_paradigms_stdout.txt:19-24`: RLS 13.67, LQ 14.69, Q-learning 274.71, Naive 447.35, GA 706.13. ✓ Match to integer precision |
| `fig:macro:rbc-curves`, `tab:macro:rbc-results` (ch06_macro) | macro_rl.tex:364-392 | "Mean episode return, 30 evaluation episodes, 10 training seeds for PPO/DDPG" | yes | Prose at macro_rl.tex:366-369: "KPR matches VFI to within 0.001 in policy MSE" (stdout: KPR MSE 0.0004 ✓), "PPO converges … MSE of 0.009" (stdout: 0.0087 ✓), "DDPG mean return within 2% of VFI" (stdout: 45.173 vs 45.861 = 1.5% ✓). All numerical claims supported |
| `fig:macro:lqmfg-curves`, `tab:macro:lqmfg-results` (ch06_macro) | macro_rl.tex:641-664 | "Approximate exploitability over policy-gradient updates, official MFAX grid, 10 seeds" | yes | Prose at macro_rl.tex:636: "RSPG has lower exploitability than SPG" matches stdout `lq_mfg_stdout.txt:18-20` (SPG 86.64, RSPG 60.37). The selected LR claim is consistent with the post-sweep stdout (best SPG at 1e-2, best RSPG at 1e-3). ✓ |
| `tab:offline_main` (ch08_offline_rl) | offline_rl.tex:149-154 | "Policy value for each offline RL method … standard errors over 20 seeds" | **partial** | **[CRITICAL]** Two problems with this table:<br>(a) Prose at offline_rl.tex:147 says "evaluates the four offline RL algorithms" but the table at `offline_rl_pricing_results.tex` contains 8 rows: DP Oracle, IQL, CQL, BC, BCQ, **DT**, **RvS**, FQI. DT and RvS are present in the table but the prose body discusses only FQI/CQL/IQL/BCQ. The chapter does have a Decision Transformer subsection at offline_rl.tex:124+ that introduces DT/RvS textually, but the simulation prose at offline_rl.tex:147-167 narrates only the four "pessimism" methods.<br>(b) **More serious:** BC, BCQ, **DT**, and **RvS** all show identical values `$169.27 \pm 0.60$ / 88.0\%` to two decimals **and the same SE**. Four distinct algorithms collapsing to byte-identical mean and SE is not consistent with seed-dependent stochastic training of four different model classes. The prose only comments on BCQ collapsing to BC (offline_rl.tex:156, "BCQ matches the behavioral at 88.0%; its action constraint restricts the policy to prices near 10"); it is silent on DT and RvS also matching to exact byte equality. The project's own simulation-audit checklist in `CLAUDE.md` (point 1, "Algorithm Identity Check") flags exactly this failure mode. An arxiv reader doing a numerical cross-check would notice immediately. Possible explanations: shared deterministic-rollout evaluation path that bypasses the trained policy, or all three (BCQ/DT/RvS) collapsing to "always price 10" before evaluation. Either way the prose owes an explanation |
| `fig:offline_coverage` (ch08_offline_rl) | offline_rl.tex:160-165 | "Policy value vs ε_b. FQI peaks at moderate coverage (ε_b=0.3) and collapses at both extremes. BCQ degrades at high ε_b" | yes | Stdout `offline_rl_pricing_stdout.txt:50-58`: FQI 54.0 → 85.1 → 49.1 (peaks at 0.3 ✓); BCQ 87.8 → 87.8 → 53.0 (collapses at 0.9 ✓). Caption claim is correct |

### Verdict

**Verdict: PARTIAL** — most figures and tables faithfully support the prose. Two issues:
- **[CRITICAL]** `tab:offline_main`: four methods collapse to identical values, prose discusses only four of seven offline methods.
- **[CRITICAL]** `tab:dtr_qlearning_vs_murphy`: caption says "30 Monte Carlo seeds" but script and stdout use 50 (tabular) / 20 (high-dim).

Everything else is numerically faithful to the stdout files.

---

## 3. Method reproducibility

Method: For each recently-touched simulation, I cross-checked (state space, action space, reward, hyperparameters, seeds, episode count, evaluation protocol) between the `_stdout.txt`, the `.py` source, and the tex prose.

| Method / experiment | Has S, A, r? | Has hyperparams? | Has seeds? | Has eval protocol? | Missing / stale |
|---------------------|--------------|------------------|------------|--------------------|-----------------|
| `dtr_qlearning_vs_murphy.py` (ch10b) | yes (rl_for_ci.tex:55) | yes (footnote rl_for_ci.tex:55: 50-seed tabular, 20-seed HD, 64-hidden MLP, 200-epoch FQI, 8000-step DQN) | yes in script (50 tabular / 20 HD) | yes (V(π̂)/V*) | **[CRITICAL]** Table caption (rl_for_ci.tex:68) says 30 seeds for both settings, but script and stdout disagree. Either rerun with 30 seeds or correct the caption to "50 tabular / 20 high-dim". |
| `dyna_maze.py` (ch12) | yes (s03_dyna_q.tex:53-57) | yes (α=0.1, ε=0.1, γ=0.95, episode cap 200, MLP 32 hidden, lr 3e-3, plan-interval 10, 16 imagined trajectories of length 10) | yes (30 seeds) | yes (cumulative reward at t=3000) | none. Reproducibility complete. |
| `cobweb_paradigms.py` (ch12) | yes (s09_dual_sim.tex:11-15) | yes (b/c sweep {0.5, 1, 2}, a=4, c=1, φ=0.2, σ=0.1, γ=0.95, T=500; Q-learning grid 20×20×25, α=0.1, ε decay 0.3→0.01; MBPO ensemble size 5, rollout horizon 5, 10 rollouts per real step) | yes (20 seeds) | yes (cumulative regret vs oracle on shared noise) | none |
| `fishery_paradigms.py` (ch12) | yes (s09_dual_sim.tex:67-70) | yes (r=0.4, K=10, p=2.0, c=0.2, σ=0.3, γ=0.95, T=500, grid 50×25; Q-learn 30×21, α=0.1, ε decay 0.3→0.01) | yes (20 seeds) | yes (cumulative regret) | none |
| `rbc_dp_vs_drl.py` (ch06_macro) | yes (macro_rl.tex:336-348) | yes (β=0.96, α=0.36, δ=0.10, ρ=0.95, σ=0.007, T=200; PPO/DDPG 64-64 MLP; VFI 400×41 grid; footnote macro_rl.tex:354-362) | yes (10 seeds) | yes (30 eval episodes, shared shock paths, MSE on 1000 random states) | none. Exemplary reproducibility. |
| `lq_mfg.py` (ch06_macro) | yes (in JSON inputs and macro_rl.tex:619-633) | yes (γ=0.99, num_envs=8, iterations=200, eval_every=20, 10 seeds, LR grid {1e-4, 1e-3, 1e-2}) | yes (10 seeds) | yes (best-response DP exploitability) | **[DEFERRED]** This script is presentation-only — the heavy compute lives in the external MFAX repo (commit `9acc1eb`, github.com/CWibault/mfax.git). Reproducibility requires that repo to remain accessible. The JSON `mfax_lq_grid_results.json` is committed locally (220 kB), so the figure/table can be rebuilt without re-running JAX. Mention the dependency in the caption or methods footnote. |
| `offline_rl_pricing.py` (ch08) | yes (offline_rl.tex:147) | yes (I_max=30, H=20, 4 demand regimes, Poisson λ_0=(1.5,3,5,8), reward p·min(Q,i), spoilage \$2/unit, behavioral 85% at p=10 + 15% uniform; CQL α=0.1; 500 train episodes, 1000 eval episodes, 200 FQI iterations) | yes (20 seeds per ε_b cell) | yes (% of DP optimal) | **[CRITICAL]** See Section 2 above — the table contains DT and RvS rows with byte-identical results to BC/BCQ. The methods prose does not describe DT/RvS hyperparameters (transformer architecture, context length, target-return schedule, RvS MLP size) for the simulation. The earlier theoretical subsection at offline_rl.tex:124-142 introduces DT/RvS conceptually but no simulation hyperparameters are stated. Either remove DT/RvS from the results table or add reproducibility detail. |
| `confounded_ope.py` (ch10_causal) | yes (causal_rl.tex:265) | yes (5 states, 2 actions, γ=0.9, transition matrix in script, behavioral μ formula with ρ sweep) | yes (20 seeds, BASE_SEED=42) | yes (per-estimator bias / RMSE) | **[DEFERRED]** Stdout `confounded_ope_stdout.txt` shows "Loaded from cache" — no fresh run output. Recommend re-running with `--force` once before arxiv submission to confirm cache is not stale. |
| `counterfactual_ope.py` (ch10_causal) | yes (causal_rl.tex:292-307) | yes (DGP coefs in script, n_grid {200,500,1000,2000}, σ_u=0.5) | yes (20 seeds) | yes (oracle V via 10^6 MC) | **[DEFERRED]** Stale paths in `counterfactual_ope_stdout.txt:7-11` still reference `ch12_forecasting_rl/sims/cache/...`. The actual files now live in `ch10_causal/sims/`. Stdout should be regenerated; if the cache directory in the script is hardcoded to the old path, the script will break under a re-run. (Code uses `os.path.dirname(__file__)` so it self-locates correctly; only the stdout artifact is stale.) **[DEFERRED]** Header comment at `counterfactual_ope.py:2` still says "Chapter 12 (forecasting and reinforcement learning), §5.3 simulation" — needs updating to "Chapter 10 causal RL §subsec:cfope_sim". |
| `dynamic_dml_snmm.py` (ch10b) | yes (rl_for_ci.tex:295-302) | yes (p=20, sparsity s=5, σ_η=0.6, ‖B‖_op=0.5, ψ_1*=1.0, ψ_2*=0.5, K=5 folds, Lasso/L1-logistic learners) | yes (200 MC reps) | yes (bias / RMSE / coverage) | **[CRITICAL]** Stdout `dynamic_dml_snmm_stdout.txt:39-41` writes paths under `ch11_rl_for_ci/sims/` — those paths do not exist (folder is `ch10b_rl_for_ci`). Confirm via `find . -path './ch11*' -name '*.py'` → empty. Regenerate the stdout. **Separately:** rl_for_ci.tex:304 footnote says `Sim 1 source: \texttt{ch11\_rl\_for\_ci/sims/dynamic\_dml\_snmm.py}` — same stale path inside the PDF text. This will appear in the compiled paper. |

### Additional stale paths inside the compiled PDF

| File:line | Bad path | Should be |
|-----------|----------|-----------|
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:304` (footnote) | `ch11_rl_for_ci/sims/dynamic_dml_snmm.py` | `ch10b_rl_for_ci/sims/dynamic_dml_snmm.py` |
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:333` (footnote) | `ch10b_rl_for_ci/sims/causal_bandit_parallel.py` | already correct ✓ |
| `ch10b_rl_for_ci/tex/rl_for_ci.tex:55` (footnote) | `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py` | already correct ✓ |
| `ch12_world_models/tex/world_models.tex:1-15` (comments) | none in compiled output | n/a |
| `ch10_causal/sims/counterfactual_ope.py:2` (Python comment, not in compiled PDF) | `Chapter 12 (forecasting and reinforcement learning)` | should be `Chapter 10 causal RL` for accuracy, but does not appear in the paper. **[DEFERRED]** |

### Verdict

**Verdict: PARTIAL** — six of nine focus sims are fully reproducible from the paper alone. Three issues block reproducibility:
- **[CRITICAL]** `dtr_qlearning_vs_murphy`: caption says 30 seeds, script uses 50/20.
- **[CRITICAL]** `offline_rl_pricing`: DT and RvS rows in the published table without methods-level description; identical values to BC suggests an algorithm-identity check (per the project's own audit checklist).
- **[CRITICAL]** Stale path `ch11_rl_for_ci/sims/dynamic_dml_snmm.py` in compiled prose footnote (rl_for_ci.tex:304).
- **[DEFERRED]** Stale stdout paths in `counterfactual_ope_stdout.txt` (cosmetic), `dynamic_dml_snmm_stdout.txt:39-41` (cosmetic), and the Python file-header comment in `counterfactual_ope.py:2`.

---

## Summary of critical issues (in priority order)

These would surprise an arxiv reader within the first hour of reading:

1. **Intro chapter map at `ch00_introduction/tex/intro.tex:11` lists 9 sections; paper has 17.** A reader scrolling from the intro to the TOC will notice immediately. Update the roadmap to match `main.tex:146-218`.

2. **`tab:offline_main` in `ch08_offline_rl/sims/offline_rl_pricing_results.tex` lists DT and RvS with byte-identical values to BC** (169.27 ± 0.60 each, 88.0% of optimal). The simulation prose at `offline_rl.tex:147` says it "evaluates the four offline RL algorithms" but the table shows seven non-oracle methods. Either remove DT/RvS from the table, or add a results paragraph explaining why three distinct sequence-modelling / behavioral methods produce indistinguishable returns to the SE, or rerun with whatever caused the collapse fixed. The project's own simulation-audit checklist (`/Users/pranjal/CLAUDE.md`, "Bullshit Score" section, Algorithm Identity Check at point 1) flags exactly this pattern as a 50%+ score.

3. **`tab:dtr_qlearning_vs_murphy` caption (`ch10b_rl_for_ci/tex/rl_for_ci.tex:68`) says "30 Monte Carlo seeds" for both tabular and high-dim**, but the script uses 50 (tabular) / 20 (high-dim). Mechanical fix: change caption to "30 seeds for the tabular setting" → "50 seeds for the tabular setting, 20 seeds for the high-dimensional setting", or rerun at 30 to match the caption.

4. **Stale path inside the compiled PDF text** at `ch10b_rl_for_ci/tex/rl_for_ci.tex:304` footnote: `ch11_rl_for_ci/sims/dynamic_dml_snmm.py` does not exist; should be `ch10b_rl_for_ci/`.

5. **Conclusion is silent on the World Models chapter** (`section:world_models`, ~ten subsections, three simulations), the longest single chapter in the paper. Add at least one sentence to the open-challenges or applied-modeling subsection.

## Summary of deferred issues

These can wait until journal revision or are internal-only:

- Abstract (`ch00_introduction/tex/abstract.tex:1`) does not announce causal inference, macro, robust/constrained, or world models even though four full chapters are devoted to them. The abstract framing as "pricing, inventory, games, preferences" is a subset of what the paper actually covers.
- Conclusion does not synthesize macro RL (`ch06_macro/tex/macro_rl.tex`), robust/constrained RL (`ch11_dist_robust_constrained/`), or RL for CI (`ch10b_rl_for_ci/`) separately.
- Several `_stdout.txt` files reference stale folder paths (`ch12_forecasting_rl/`, `ch11_rl_for_ci/`) — cosmetic, doesn't appear in the compiled PDF.
- `counterfactual_ope.py:2` Python file-header comment is mis-attributed to old chapter.
- `lq_mfg.py` is a presentation-only script; the heavy compute is in external `mfax` repo (commit `9acc1eb`). The methods footnote should mention this dependency explicitly so a reader running the code knows they need the external repo.
