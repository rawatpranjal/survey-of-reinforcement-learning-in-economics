# Audit: ch09_rlhf/sims/job_search_preference_learning.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch09_rlhf/tex/rlhf.tex` (Section "Simulation Study: Preference Learning in Job Search", lines 76-116; figure `fig:preference_sample` on line 89-94 includes `job_search_sample_complexity.png`; table `tab:preference_diagnostics` on line 98-103 inputs `job_search_diagnostics.tex`; figure `fig:preference_horizon` on line 107-112 includes `job_search_horizon.png`).
**Cited paper PDFs read:** `papers/christiano:2017.pdf` (RLHF foundations), `papers/ouyang2022training.pdf` (InstructGPT, three-stage pipeline), `papers/rafailov2023direct.pdf` (DPO). All present in `papers/`. Bradley-Terry 1952 not in `papers/` (cited only as foundational reference; standard logit identity, no controversy). McCall 1970 (job search) not in `papers/` either.

---

## 1. Algorithm Identity

**Bradley-Terry reward MLE (NN RLHF and structural).** The code matches the textbook BT logit loss. In `train_reward_net` (line 405) and `fit_structural` (line 447), the negative log-likelihood is built from `logsigmoid(r_w - r_l)` over segment sums of discounted per-state rewards. This is exactly Equation (1) of Christiano 2017 generalized to discounted segment scoring; the tex's Eq.~(\ref{eq:rlhf_loss}) at line 9 matches. The discount weighting `gamma_powers` (line 312, line 407, line 450) is applied consistently in both the data generator and both fitters. *Match.*

**NN architecture.** Two hidden ReLU layers of width 32 over a 4-D feature (normalised log-wage, normalised amenity, employed flag, action). Tex footnote (line 87) says "4 inputs, 32 hidden units per layer, ~1,200 parameters" — code matches. Reward extracted as `mean(axis=1)` over the two actions per state (line 440); this is a presentation choice and harmless because policy is then re-solved by VI over the full $(s,a)$-conditioned Q. *Match.*

**DPO.** Tabular softmax with per-state logit $\phi_s$, reference policy uniform ($\pi^{SFT}(a|s)=0.5$), trained with Adam over $\lambda_{KL} \in \{0.01,\dots,5\}$, then a greedy policy. The loss in `train_dpo` (line 524) computes per-step `_log_sigmoid(sign*phi[state])` summed over the segment, then BT-style `log_sigmoid(lam*(logr_w - logr_l))`. This is Rafailov 2023 Eq.~(7) with the partition $\log Z$ cancelled by the same-state pairing, which the tex correctly notes (footnote line 87, "same initial state"). The reference cancels exactly because $\pi^{SFT}$ is uniform and constant. *Match.*

**Hostile-reviewer flag (minor).** The NN reward learner sees `STATE_FEATURES[w_states, w_actions]` (line 408) — a 4-D vector that *includes the action* as an input to the reward network. That is a state-action reward $r_\theta(s,a)$, which is fine. But then `extract_reward_table` averages over actions (line 440) before handing to VI. This means the reward used by VI is state-only $\bar{r}(s) = \tfrac{1}{2}(r_\theta(s,0)+r_\theta(s,1))$, while the learner has the freedom to fit an action-dependent reward. The action information is then thrown away. Since the underlying true reward `TRUE_REWARD_VEC` is action-independent (line 143), this is harmless in expectation, but a strict reading of "trained reward model serves as scalar signal" would prefer $r_\theta(s)$ or VI over a $(s,a)$ table. The result is still a valid two-stage RLHF estimator. *Pass with a footnote.*

**Misspecified model.** Replaces $z$ with $\bar{z}=3$ (constant). Code matches text claim "ignores amenity variation". *Match.*

## 2. Environment / MDP Fidelity

The tex says: 8 wages × 7 amenities = 56 (s,e) pairs; 112 states (searching + employed); $u(w,z) = \alpha\log(w)+(1-\alpha)z$ with $\alpha=0.6$; unemployment benefit $b=28$ giving $u_b=\alpha\log b$; layoff probability $0.05$; $\gamma=0.95$; offer correlation $\rho \approx -0.74$.

Code: `WAGE_LEVELS=8`, `AMENITY_LEVELS=7`, `NUM_STATES=112`, `ALPHA_TRUE=0.6`, `BENEFIT_WAGE=WAGES[1]=28`, `P_LAYOFF=0.05`, `GAMMA=0.95`, and offer correlation reported in stdout as `-0.740`. The wage grid in code is `[20,28,38,50,65,82,100,125]` — matches tex verbatim. The amenity grid `np.linspace(0,6,7)` = `[0,1,2,3,4,5,6]` — matches tex `z \in \{0,1,\ldots,6\}`. *Match.*

**McCall fidelity.** The tex calls this "McCall (1970)-style". Classical McCall is: search at constant benefit $b$, accept iff $w \geq w^*$, $w^* = b + \beta E[(w-w^*)_+]/(1-\beta)$. This model extends McCall in three nontrivial ways: (a) two-dimensional offers $(w,z)$ with compensating differentials, (b) on-the-job re-search (employed worker can quit), (c) involuntary layoffs. None of these violate the classic; they are documented McCall extensions (Burdett-Mortensen, McCall-Mortensen). The tex says "McCall (1970)-style", which is fair. *Match.*

**Quirk.** Searching-state reward is set to $\alpha\log b$ regardless of which offer is showing (line 142-149). That is, the search-state "reward" is the flow utility from unemployment benefit, not the value of the pending offer. The pending offer affects only the transition (line 198-205 in `_build_transition_matrices`): if the worker accepts, they transition to employed at that (w,z). This is the correct McCall structure (you receive benefit while waiting/deciding), but a careless reader of the reward vector might confuse the offer-bearing searching state with realised payoff. The code is right. *Match.*

## 3. Data Integrity

`compute_data` actually trains: `run_experiment()` runs VI (line 756), Q-learning (line 778-787), the K-sweep (line 794-862), the diagnostics block (line 864-933), and the segment-length ablation (line 935-975). No hardcoded results.

`evaluate_policy_mc` (line 271) returns 74.305±0.312, and analytic VI gives 74.129; the two agree within MC error (3 standard errors). The verification block (line 1008-1031) hard-codes the check `abs(dp_v0 - 74.13) < 0.5` — a sanity check, not the reported number. *Pass.*

**One real concern.** The script does *not* use the project's caching pattern (`sim_cache.py`); it does not write a `.pkl`. The stdout `job_search_preference_learning_stdout.txt` is 130 lines and visibly truncated mid-Experiment-3 ("L = 15."). The diagnostics.tex and sample_complexity.png are timestamped Mar 16 12:26; the env.png is Mar 17 04:06; the stdout file header says "2026-03-17 04:06:18". So at least one of the published artifacts (sample_complexity, diagnostics.tex) was produced by an earlier run than the saved stdout. The numbers in stdout and diagnostics.tex agree (NN 96.4, DPO 57.1, Correct 100.0, Misspec 50.0; mean amenity 4.67/3.38/4.84/3.00; mean wage 73/63/71/70), so the figure/table are consistent with stdout numbers, but the stdout file itself is half-written.

This is a *housekeeping* failure, not a data-integrity failure: every reported number can be regenerated by re-running the script. But a hostile reviewer demanding a complete replication log would write a snarky comment about the truncated stdout. *Pass-with-snark.*

## 4. Comparison Fairness

Per-seed seeding is `MASTER_SEED + seed*10000 + K` (line 809), and NN-RLHF, structural, and misspecified all consume *the same* cross-state comparison batch generated at that seed (line 814). DPO gets its own same-state batch generated at `rng_seed + 500000` (line 817-819). This is correct: DPO requires same-prompt comparisons (the tex explicitly justifies this on line 87: "to match the LLM setup where both completions condition on the same prompt"), while reward-model methods can use cross-state pairs. Both batches have the same $K$ and use a uniform random behavioural policy. Eval is identical: closed-form `policy_eval_vec` against the *true* reward vector at the inflow distribution.

Q-learning is a separate baseline (line 776-792) with 10,000 episodes × ~200 steps ≈ 2M environment queries (stdout line 46). RLHF gets $K \times 2 \times L \leq 5000\cdot 2\cdot 15 = 150{,}000$ trajectory transitions for data generation, then $\sim 100$ epochs $\times K/64$ batches of gradient steps. These are not directly comparable on "environment-interaction" axis (RLHF uses fewer real-environment transitions but only sees preferences, not rewards). The tex does not claim a head-to-head environment-budget comparison; the figure plots V vs $K$, not V vs queries. *Fair on its own terms.*

**Fair pairing for DPO?** A hostile reviewer would push back: the same-state DPO data has lower variance per pair than cross-state RLHF data (more informative comparisons), giving DPO an advantage in nominal $K$. But DPO still underperforms NN RLHF at every $K$, so this potential bias works *against* the simulation's main narrative ("RLHF beats DPO"), not for it. *Pass.*

## 5. Theoretical Sanity Checks

(a) **DP oracle vs MC verification.** $V^*(s_0) = 74.13$ analytically; MC roll-outs give $74.305 \pm 0.312$. The two are within 0.6 of each other (≈2 SEs). The discrepancy is because closed-form $V$ assumes infinite horizon while MC truncates at MAX_STEPS=200 with $\gamma^{200} \approx 4 \times 10^{-5}$; not a problem. *Pass.*

(b) **Q-learning convergence.** Q-learning reaches 99.3% of DP optimal (73.604/74.129) — sensible for tabular Q with 112 states, 2M queries, $\varepsilon=0.15$. *Pass.*

(c) **RLHF sample-complexity rate.** From stdout, NN RLHF gap-to-optimal in % goes 1.5, 1.1, 0.7, 0.4, 0.2, 0.2, 0.1, 0.1 over $K=$ 25→5000 (a 200× range). A $\sqrt{n}$ rate would predict the gap to shrink by $\sqrt{200} \approx 14\times$; we see 15×. Consistent. Correct structural gap shrinks from 0.1% to 0.0% immediately — sensible for a one-parameter model with $K=25$ comparisons (parametric efficiency at small $K$). *Pass.*

(d) **DPO plateau.** Plateaus at ~95% by $K=500$. The tex (line 96 footnote) attributes this to DPO inability to propagate value through undervisited states. This is reasonable but slightly hand-wavy: a tabular per-state-logit DPO trained on a random-policy dataset will of course look bad on rarely-visited states, and the policy diverges from $\pi^*$ on 43% of states (Agree% = 57.1 in diagnostics.tex). A more rigorous read: $L=15$ segments from a uniform random policy spend ~half their steps employed at low-wage offers (because random accept-while-searching has acceptance probability 0.5), so high-wage employed states are underrepresented and DPO's logit at those states is poorly identified. The tex's footnote is consistent with this. *Pass.*

(e) **Misspecified plateau.** 91% regardless of $K$. Theory predicts that under fixed bias the BT MLE converges to the pseudo-true $\alpha$ minimizing KL to the truth; the estimated $\alpha$ stabilizes at ~0.6 across $K$ (stdout `mi_alphas`), but plug-in DP under the wrong reward vector fails on the amenity dimension. *Pass.*

(f) **Segment length $L$.** At $L=1$, NN RLHF gets 76.4% — degenerate, because single-step comparisons cannot distinguish "accept at state $s$" from "reject at state $s$" when the per-period reward is identical for both (searching reward = constant benefit utility, regardless of the offer shown). The information lives in *future* states. Stdout shows the NN at $L=1$ has SE 2.278 — high variance, confirming under-identification. Recovers by $L=3$. *Pass.*

## 6. Information Leakage

The pairwise preference generator (`generate_comparisons` line 349, `generate_comparisons_same_state` line 368) calls `generate_segment` which computes `total_reward = np.dot(gamma_powers, TRUE_REWARD_VEC[states])` (line 326) using the *true* reward vector. This is the oracle teacher. The preference label is then sampled from $\sigma(r_1 - r_2)$ — Bradley-Terry. This is the correct simulated-human-feedback protocol (per Christiano 2017, Stiennon 2020 simulation studies); the learner never sees `TRUE_REWARD_VEC` directly, only binary labels.

The NN reward learner (`train_reward_net`) receives only `(w_states, l_states, w_actions, l_actions)` — no rewards. The structural estimators receive only states. DPO receives only states + actions. Policy evaluation (`policy_eval_vec`) for the *reported* policy value uses `TRUE_REWARD_VEC`, which is correct (this is the evaluation metric, not training signal). *No leakage.* *Pass.*

## 7. Seed & Reproducibility

`MASTER_SEED = 42`. Sample-complexity loop uses 30 seeds (above the 10 minimum). Ablation uses 20 seeds. Each seed sets both `np.random.seed` and `torch.manual_seed` (line 812-813). SEs are reported throughout (`std / sqrt(N_SEEDS)`).

**Quirk.** The DPO data is generated at `rng_seed + 500000` but `rng_seed = MASTER_SEED + seed*10000 + K`. For $K \in \{25, ..., 5000\}$ and $seed \in \{0,...,29\}$, $rng\_seed$ ranges up to $42 + 29\cdot 10000 + 5000 = 295{,}042$. Adding 500000 lands at 795042, well clear of any other seed in the script. No collisions. *Pass.*

**Final-loss check on DPO.** `run_dpo` (line 587-596) picks the $\lambda_{KL}$ with the *lowest training loss*, not the highest *held-out* policy value. This is the standard DPO recipe (no validation set; $\lambda_{KL}$ is a hyperparameter chosen by training fit). A hostile reviewer might argue this favors overfitting DPO; but since DPO consistently underperforms RLHF, it's not biasing the narrative. *Pass.*

---

## Hostile-Reviewer Summary

The script implements a clean two-stage RLHF (Bradley-Terry MLE on segment scores + value iteration on the learned reward) and a tabular DPO (BT on policy log-ratios with $\pi^{SFT}$ uniform). The environment is a McCall-extended job-search MDP with compensating differentials, 112 states, deterministic transitions modulo a 5% layoff hazard. Numerical results pass internal consistency: DP MC ≈ analytic VI; NN RLHF gap shrinks roughly as $\sqrt{K}$; misspecified model plateaus where theory predicts; DPO plateau at ~95% explained by the policy-coverage argument. Seeding is disciplined (30 seeds + SEs). No information leakage in the teacher–learner pipeline.

Two snark-worthy issues. (i) `extract_reward_table` averages over actions before solving VI, so the NN reward model's action-dependence is discarded — harmless for this environment (true reward is action-independent) but cosmetically inconsistent with the "$r_\theta(s,a)$" framing in the tex. (ii) The saved `_stdout.txt` is truncated mid-Experiment 3, and the figure/table mtime predates the stdout file; not a result-integrity issue (numbers in stdout and diagnostics.tex agree), but a hostile reviewer demanding a clean replication log would write a comment. Neither issue affects the main empirical claims.

Theory-vs-empirics alignment is the strongest part of the audit: the four-method ordering (Correct > NN RLHF > DPO > Misspecified) tracks the chapter's narrative on identification and welfare aggregation, and each method's failure mode (DPO underweights amenities at 3.38 vs optimal 4.84; misspecified plateaus at 91%) is internally diagnosable from the diagnostics table.

**Bullshit score: 15%** — Reviewer 2 catches the truncated stdout and the action-averaging in the reward extraction, but the substance is correct and the empirical pattern matches the theoretical claims.
