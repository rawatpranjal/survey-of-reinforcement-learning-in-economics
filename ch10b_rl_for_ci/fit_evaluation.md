# Fit Evaluation: 43 New RL-for-CI Papers vs the Current Chapter

This document evaluates 43 newly-acquired papers (downloaded + docling-extracted in the prior phase, sitting in `papers/`) against the current state of `tex/rl_for_ci.tex` (5 subsections + sim study + discussion, 359 lines). One paper of the original 44 (`wang2024adt2r` / Jeon et al, IEEE TNNLS) is paywalled and out of scope.

**Verdict legend.** `KEEP` = strong fit, recommend citing primary; `MARGINAL` = useful but borderline (footnote / one-line aside); `CUT` = doesn't add to the chapter's argument, or duplicates an existing citation.

The chapter's gap diagnosis (from earlier in the session): §2 (Dynamic DML) and §3 (Dynamic offline policy learning) lean on g-estimation / AIPW papers that re-use Bellman vocabulary without running RL. §4 (causal bandits) and §5 (adaptive experiments) are stronger on real RL. §1 (DTRs) is the historical-bridge section.

This file has three parts:
- **Part A** — per-paper verdicts (43 entries, organised by intended target wave).
- **Part B** — per-existing-section recommendations (which KEEP papers go where, what each enables).
- **Part C** — proposed new subsections (where a coherent cluster doesn't fit the existing 5).

---

## Part A — Per-paper verdicts

### Wave 1 — §2 Dynamic DML / SNMMs (10 papers)

**1. kallus2020doublerl** — Kallus & Uehara, "Double Reinforcement Learning for Efficient OPE in MDPs," JMLR 2020.
- *RL algorithm executed:* DRL — cross-fold neural q-function + marginalised density ratio plugged into the efficient influence function for the MDP model. Real RL, real DML.
- *Estimand:* Policy value E[∑ γ^t r_t] under target policy in finite-horizon MDP with semiparametric efficiency bound.
- *Identification:* Sequential ignorability + Markov assumption.
- **Verdict: KEEP — primary anchor for §2.** This is THE paper the chapter currently *implies* exists when it labels Lewis-Syrgkanis as "the orthogonal-score version of fitted-Q evaluation." Adding this lets §2 actually claim a deep-RL OPE estimator inheriting the √n rate, not just an SNMM-with-DML-relabeling. Goes mid-§2, paired with Lewis-Syrgkanis as the "two routes to the same orthogonality" framing.

**2. kallus2022curse** — Kallus & Uehara, "Efficiently Breaking the Curse of Horizon in OPE with Double RL," Operations Research 2022.
- *RL algorithm executed:* Infinite-horizon DRL — neural marginalised stationary-density ratio + neural q-function via DICE-family minimax saddle. Achieves the MDP efficiency bound under mixing.
- *Estimand:* γ-discounted infinite-horizon policy value.
- *Identification:* Sequential ignorability + stationarity + mixing.
- **Verdict: KEEP — primary anchor for §2 infinite-horizon claim.** Paired with paper 1: paper 1 is the finite-horizon result, paper 2 is the infinite-horizon one. Together they cover the regime the chapter currently leaves unaddressed (infinite-horizon causal estimands). Goes after the finite-horizon Lewis-Syrgkanis material in §2, opens a paragraph on "from finite to infinite horizon."

**3. nachum2019dualdice** — Nachum, Chow, Dai, Li, "DualDICE," NeurIPS 2019.
- *RL algorithm executed:* DualDICE — Fenchel-dual minimax program (two neural networks) for stationary-distribution corrections, behavior-agnostic.
- *Estimand:* Long-run policy value via stationary state-action density ratio.
- *Identification:* Sequential ignorability in MDP.
- **Verdict: MARGINAL — §2 footnote.** Foundational RL paper, not a causal-inference paper per se, but cited extensively by Kallus-Uehara 2020/2022 and downstream causal-OPE work. Worth a parenthetical "(the DICE-family of stationary-ratio estimators originated with Nachum et al 2019)" when introducing the marginalised density ratio in §2.

**4. kallus2020policygrad** — Kallus & Uehara, "Statistically Efficient Off-Policy Policy Gradients," ICML 2020.
- *RL algorithm executed:* EOPPG — efficient off-policy policy gradient with cross-fitted neural q, ∇q, and density-ratio nuisances; 3-way double robustness; achieves O(H⁴/n) MSE bound.
- *Estimand:* Policy gradient ∇_θ V(π_θ) of the policy value, viewed as a causal derivative.
- *Identification:* Sequential ignorability + Markov.
- **Verdict: KEEP — primary, §2.** Crucial for the chapter because it extends the DML-orthogonality story from policy *evaluation* to policy *learning* via gradients. The chapter currently leaves this gap (§2 is OPE-only; §3 is policy learning but uses AIPW-classification, not policy-gradient). One paragraph in §2 pivoting from OPE to learning.

**5. uehara2021minimax** — Uehara, Imaizumi, Jiang, Kallus, Sun, Xie, "Finite Sample Minimax Offline RL," 2021.
- *RL algorithm executed:* MWL/MQL minimax neural-network estimators for q and w functions; finite-sample fast rates under realisability + completeness; first-order efficient.
- *Estimand:* OPE policy value with general function approximation.
- *Identification:* Sequential ignorability + Bellman completeness.
- **Verdict: MARGINAL — §2 technical footnote.** Important refinement of paper 1 (finite-sample, not just asymptotic; identifies the realisability/completeness conditions that make √n inference possible). But it's a technical paper, not a story paper; one footnote alongside the main DRL citation suffices.

**6. shi2022confoundedpomdp** — Shi, Uehara, Huang, Jiang, "Minimax Learning for OPE in Confounded POMDPs," ICML 2022.
- *RL algorithm executed:* Minimax neural-network learning of value- and weight-bridge functions, plus three plug-in OPE estimators (value-based, IS, doubly-robust); semiparametric efficiency for the DR variant.
- *Estimand:* Policy value in a *confounded* POMDP under proxy/bridge identification.
- *Identification:* Proximal causal inference (negative-control / bridge functions). Not sequential ignorability.
- **Verdict: KEEP — primary anchor for a NEW SUBSECTION on POMDP OPE under proxy identification.** Doesn't fit cleanly into §2 (different identification regime). Pair with bennett2024proximalrl below as the two anchors of a new subsection between §2 and §3.

**7. bennett2024proximalrl** — Bennett & Kallus, "Proximal RL: Efficient OPE in POMDPs," Operations Research 2024.
- *RL algorithm executed:* Adversarial-ML estimation of bridge functions + cross-fitted DR estimator with explicit asymptotic normality; sepsis simulator demo.
- *Estimand:* γ-discounted policy value in POMDP with latent confounders.
- *Identification:* Proximal causal inference extended to dynamic / longitudinal setting.
- **Verdict: KEEP — co-anchor for the same proposed NEW SUBSECTION.** Bennett-Kallus is the more general identification result; Shi 2022 is the more developed estimator side. Together they cover "what is identified" and "how to estimate" for POMDP OPE under proxies. Sepsis demo gives the empirical hook.

**8. uehara2023futuredep** — Uehara, Bennett, Kiyohara, Chernozhukov, Jiang, Kallus, Shi, Sun, "Future-Dependent Value-Based OPE in POMDPs," NeurIPS 2023.
- *RL algorithm executed:* Future-dependent value functions via minimax Bellman saddle program; uses future + history observations as proxies for latent state.
- *Estimand:* Long-run policy value in POMDPs (focused on curse of *history* not curse of *confounding* — the POMDP here is non-confounded).
- *Identification:* Bridge functions for partial observability; not framed as confounding identification.
- **Verdict: MARGINAL — footnote in the new POMDP subsection.** Same bridge-function machinery as Shi/Bennett but the framing is partial observability, not unmeasured confounding. Worth a one-line "(see also Uehara et al 2023 for the related curse-of-history framing)."

**9. hao2021bootstrapfqe** — Hao, Ji, Duan, Lu, Szepesvári, Wang, "Bootstrapping FQE for Off-Policy Inference," ICML 2021.
- *RL algorithm executed:* FQE with linear/neural function approximation, plus bootstrap (and subsampled bootstrap) for confidence intervals; asymptotic normality + Cramér-Rao efficiency for linear FQE.
- *Estimand:* Policy value with confidence intervals.
- *Identification:* Sequential ignorability in episodic MDP.
- **Verdict: MARGINAL — §2 footnote on alternative routes to CIs.** Useful complement to the DRL/DML asymptotic-normality story (bootstrap is a different route to the same end), but not load-bearing. One-sentence aside.

**10. brunssmith2022uncertain** — Bruns-Smith, "Model-Free and Model-Based Policy Evaluation when Causality is Uncertain," ICML 2022.
- *RL algorithm executed:* FQE (model-free) and robust-MDP value iteration (model-based) under a per-period iid-confounding sensitivity model.
- *Estimand:* Worst-case lower bound on policy value under bounded confounding.
- *Identification:* Per-period sensitivity model (Tan-style bounded confounding); contrasts persistent vs iid confounder regimes.
- **Verdict: KEEP — primary, for a proposed NEW SUBSECTION on sensitivity / partial identification.** The chapter currently routes the "what when sequential ignorability fails" question to the sister chapter `causal_rl.tex`, but Bruns-Smith is squarely an offline-RL contribution under a sensitivity model — it belongs here. Anchor a brief 4-paragraph subsection on partial-identification OPE alongside Kallus-Zhou (2020) and Namkoong et al (2020) from Wave 2.

### Wave 2 — §3 Dynamic offline policy learning (8 papers)

**11. liao2021ivvi** — Liao, Fu, Yang, Wang, Ma, Kolar, Wang, "Instrumental Variable Value Iteration for Causal Offline RL," JMLR 2024.
- *RL algorithm executed:* IVVI — model-based value iteration where the transition kernel is learned via a neural conditional moment restriction (NPIV) using IVs, then planning runs on the de-confounded model.
- *Estimand:* Optimal-policy value in a confounded MDP with continuous actions and continuous IVs.
- *Identification:* Instrumental variables under additive nonlinear confounding.
- **Verdict: KEEP — primary anchor for proposed NEW SUBSECTION on sensitivity / IV / partial-id.** First provably efficient IV-aided RL algorithm. Pairs naturally with Fu (entry 12) on the infinite-horizon side. The chapter currently has zero discussion of IV-based identification within RL; this is the load-bearing addition.

**12. fu2022offlineiv** — Fu, Qi, Wang, Yang, Xu, Kosorok, "Offline RL with Instrumental Variables in Confounded MDPs," 2022.
- *RL algorithm executed:* Pessimistic value-function-based and MIS-based policy learning with IVs; doubly-robust combination; finite-sample suboptimality (welfare-regret) bound O(log(NT)/√(NT)).
- *Estimand:* Optimal in-class policy value in infinite-horizon confounded MDP.
- *Identification:* Discrete IV with memoryless unmeasured confounders.
- **Verdict: KEEP — primary co-anchor for the same NEW SUBSECTION.** Liao = finite-horizon model-based, Fu = infinite-horizon DR with welfare-regret. Together they cover both halves of the IV-based offline-RL story. The DR + pessimism + IV combination is exactly the kind of "real RL on a causal estimand with explicit identification strategy" that the chapter currently lacks.

**13. kallus2020confoundingrobust** — Kallus & Zhou, "Confounding-Robust Policy Evaluation in Infinite-Horizon RL," NeurIPS 2020.
- *RL algorithm executed:* Optimisation over stationary-occupancy ratios under a sensitivity-model constraint (MSM); DICE-style minimax program with non-convex projected gradient.
- *Estimand:* Sharp bounds on infinite-horizon policy value under stationary unobserved confounding.
- *Identification:* Marginal sensitivity model (bounded confounding odds-ratio).
- **Verdict: KEEP — primary, NEW SUBSECTION (sensitivity).** With Bruns-Smith (per-period iid) and Namkoong (one-decision), this is the third anchor covering the "stationary infinite-horizon" sensitivity regime. The trio together gives a complete spectrum of sensitivity assumptions for OPE in RL.

**14. namkoong2020unobserved** — Namkoong, Keramati, Yadlowsky, Brunskill, "OPE for Sequential Decisions Under Unobserved Confounding," NeurIPS 2020.
- *RL algorithm executed:* Loss-minimisation procedure over importance-weighted off-policy estimators within an MSM-style sensitivity ball, with neural function classes; sepsis + autism case studies.
- *Estimand:* Worst-case episodic policy value under one-decision unobserved confounding.
- *Identification:* One-decision Tan-style sensitivity model.
- **Verdict: KEEP — primary, NEW SUBSECTION (sensitivity).** Strong empirical anchor (sepsis/autism) for the partial-id story. Pair with Kallus-Zhou (stationary), Bruns-Smith (per-period), and the IV papers (Liao, Fu) for a 4-5 paragraph subsection.

**15. shi2024confoundedci** — Shi, Zhu, Shen, Luo, Zhu, Song, "OPE CI Estimation with Confounded MDP," JASA 2024.
- *RL algorithm executed:* Mediator-aided FQE with neural function approximation; cross-fitted DR score with explicit asymptotic normality; ridesharing case study.
- *Estimand:* √n-consistent CIs for infinite-horizon optimal policy value in a confounded MDP.
- *Identification:* Front-door / mediator identification (mediators block A→R, A→S′ paths).
- **Verdict: KEEP — primary, fits the proposed POMDP/proxy NEW SUBSECTION.** Different identification strategy than the bridge functions of Shi 2022 / Bennett 2024 (front-door instead of negative-control), but same problem: OPE in a confounded MDP. Belongs in the same subsection as a third identification flavour (proxy → front-door → IV are three natural identification strategies the new subsection should cover).

**16. hess2025sharp** — Hess, Frauen, Melnychuk, Feuerriegel, "Efficient and Sharp Off-Policy Learning under Unobserved Confounding," ICLR 2026.
- *RL algorithm executed:* Closed-form sharp-bound estimator + one-step bias correction for policy *learning* under MSM; not multi-step RL but single-stage off-policy learning under sensitivity.
- *Estimand:* Sharp lower-bound policy value (worst-case welfare).
- *Identification:* Marginal sensitivity model.
- **Verdict: MARGINAL — footnote in NEW SUBSECTION (sensitivity).** Single-stage, not multi-step. The chapter is about *dynamic* causal inference, so single-stage policy learning is a sister-chapter topic. But this paper does close a real gap (sharp + efficient bound for MSM-policy-learning), so worth a one-sentence footnote noting the single-stage analogue exists alongside Kallus-Zhou's multi-step version.

**17. liu2020batchoffpolicy** — Liu, Agarwal, Swaminathan, Brunskill, "Provably Good Batch RL Without Great Exploration," NeurIPS 2020.
- *RL algorithm executed:* Pessimistic API/AVI with neural function approximation; modified Bellman backups under bounded coverage.
- *Estimand:* Welfare regret of returned policy vs. best-in-class behaviour-supported policy.
- *Identification:* Sequential ignorability + partial coverage assumption.
- **Verdict: CUT.** This is pure offline-RL theory (relaxing concentrability) with no causal-inference angle — sequential ignorability is assumed and not the focus. The chapter's thesis is RL for CI; pure offline RL belongs in `ch08_offline_rl/`, not here. Skip.

**18. yin2022deconfoundingac** — Yin, Liu, Caterino, Zhang, "Deconfounding Actor-Critic with Policy Adaptation for DTRs," KDD 2022.
- *RL algorithm executed:* Actor-critic with LSTM state encoder, patient-resampling for balance, IPW reweighting on rewards, dynamic inverse-probability-of-treatment weighting; MIMIC-III + AmsterdamUMCdb mechanical ventilation.
- *Estimand:* Optimal DTR policy under hidden confounders in EHR data.
- *Identification:* Latent-variable model + IPW + balance.
- **Verdict: KEEP — primary, fits §1 (healthcare DTRs).** Strong real-world deep-RL paper that bakes a deconfounding module *into* the actor-critic training loop, not just on top of FQE. Distinct from CQL-on-sepsis papers in §1 because the causal machinery is internal to the algorithm. Goes in §1 as the contemporary "deconfounded RL on MIMIC" reference.

### Wave 3 — §4 Causal bandits (10 papers)

**28. lu2020causalbackground** — Lu, Meisami, Tewari, Yan, "Regret Analysis of Bandit Problems with Causal Background Knowledge," UAI 2020.
- *RL algorithm executed:* C-UCB and C-TS (causal versions of UCB and Thompson sampling); plus linear extensions CL-UCB and CL-TS. Cumulative-regret bounds.
- *Estimand:* Best-intervention identification + cumulative reward in a known causal DAG.
- *Identification:* Known causal graph + parental conditional distributions.
- **Verdict: KEEP — primary, §4.** Resolves Lattimore et al's 2016 open problem (cumulative regret, not just simple regret). The chapter currently has Lattimore-2016 as the §4 anchor for *simple* regret only; this is the natural cumulative-regret companion. Add a paragraph after the Lattimore discussion in §4.

**29. lu2022causalmdp** — Lu, Meisami, Tewari, "Causal Markov Decision Processes," CLeaR 2022.
- *RL algorithm executed:* C-UCBVI for tabular causal MDPs and CF-UCBVI for factored causal MDPs; plus linear-MDP extension. Regret O(HS√(ZT)) where Z is graph-derived (potentially exponentially smaller than action count A).
- *Estimand:* Optimal-policy value in a causal MDP (sequential causal bandit).
- *Identification:* Known factored causal graph over state and action variables.
- **Verdict: KEEP — primary, §4 extension.** Lifts the causal-bandit story from single-stage to sequential MDP. Currently §4 is single-stage only; this is the natural sequential-causal-bandit extension. Could anchor a brief "from causal bandits to causal MDPs" paragraph at the end of §4.

**30. liu2018idsgraph** — Liu, Buccapatnam, Shroff, "IDS for Stochastic Bandits with Graph Feedback," AAAI 2018.
- *RL algorithm executed:* IDS variants for graph-feedback bandits, plus refined Thompson sampling analysis.
- *Estimand:* Cumulative reward / regret under side-observation graph feedback.
- *Identification:* Side-observation graph (NOT a causal DAG — graph indicates which arms reveal info about which other arms, not causal cause-effect).
- **Verdict: CUT.** On close read, this is a graph-*feedback* bandit paper, not a causal-graph bandit paper. Side observation ≠ do-calculus. The bibliography placed it here optimistically as "interpretable as soft causal side-information" but the paper itself frames it as graph feedback à la Mannor-Shamir. Doesn't fit the chapter's RL-for-CI thesis.

**31. yabe2018propagating** — Yabe, Hatano, Sumita, Ito, Kakimura, Fukunaga, Kawarabayashi, "Causal Bandits with Propagating Inference," ICML 2018.
- *RL algorithm executed:* Two-step preprocessing + importance-sampling causal bandit algorithm for arbitrary (propagating) interventions, simple regret O(√(γ* log(|A|T)/T)).
- *Estimand:* Best intervention in a binary DAG causal bandit, where interventions can affect any subset of nodes.
- *Identification:* do-calculus on a known DAG over binary nodes.
- **Verdict: KEEP — primary, §4 extension.** Currently §4 cites Lattimore-2016, which only handles localized interventions (parallel graphs and a special general-DAG with localized interventions). Yabe extends to arbitrary propagating interventions. Add as a one-paragraph extension after the Lattimore Algorithm 2 reference.

**32. dekroon2022separating** — de Kroon, Belgrave, Mooij, "Causal Bandits without Prior Knowledge using Separating Sets," CLeaR 2022.
- *RL algorithm executed:* Thompson Sampling + UCB-normal variants augmented with on-the-fly conditional independence testing; uses separating-set estimator instead of full causal discovery.
- *Estimand:* Best intervention in an unknown causal DAG.
- *Identification:* Faithfulness + interleaved CI testing.
- **Verdict: KEEP — primary, §4.** Currently §4 assumes the graph is known in all citations. de Kroon relaxes this — significant scope expansion. Add a paragraph on "what if the graph isn't known" before the simulation study.

**33. feng2023combinatorial** — Feng & Chen, "Combinatorial Causal Bandits," AAAI 2023.
- *RL algorithm executed:* BGLM-OFU — UCB algorithm for binary generalized linear causal bandits with combinatorial interventions. O(√(T log T)) regret without scaling in 2^N intervention space.
- *Estimand:* Cumulative reward under combinatorial interventions on K out of N nodes.
- *Identification:* Known causal graph (Markovian + extension to hidden vars).
- **Verdict: KEEP — primary, §4.** §4 currently has zero combinatorial coverage. Real-world causal bandits frequently involve simultaneous interventions (drug combinations, multi-feature ad targeting). Add as the combinatorial-extension paragraph alongside Yabe.

**34. varici2022linearsem** — Varici, Shanmugam, Sattigeri, Tajer, "Causal Bandits for Linear SEMs," NeurIPS 2022.
- *RL algorithm executed:* LinSEM-UCB and LinSEM-TS — UCB and Thompson sampling exploiting linear SEM parameterisation. Cumulative regret O(d^((L+1)/2)√(NT)) without depending on intervention-space size 2^N.
- *Estimand:* Cumulative regret under soft interventions on a linear SEM with known DAG.
- *Identification:* Known DAG with linear SEM and stochastic soft interventions.
- **Verdict: KEEP — primary, §4.** Soft interventions are a different intervention model than do-calculus hard interventions; the chapter currently doesn't engage with soft interventions at all. Worth a paragraph on "soft vs hard interventions" with this as the load-bearing citation.

**35. sussex2023mcbo** — Sussex, Makarova, Krause, "Model-Based Causal Bayesian Optimization," ICLR 2023.
- *RL algorithm executed:* MCBO — model-based BO that explicitly models the SCM mechanisms; UCB-style acquisition with reparameterisation trick; first non-asymptotic cumulative-regret bound for CBO.
- *Estimand:* Optimal soft-intervention value in a causal graph with continuous variables and Gaussian-process-modelled mechanisms.
- *Identification:* RKHS-modelled structural functions on known DAG; do-operator semantics.
- **Verdict: KEEP — primary, §4 expansion.** Connects causal bandits to Bayesian optimisation and continuous-variable causal mechanisms. Different methodological toolkit than discrete causal bandits. Could be a paragraph at the end of §4 on the continuous/Bayesian-optimization angle, or a footnote pointing to a sister literature.

**36. yan2024linearcb** — Yan, Lu, Tewari, Tajer, "Linear Causal Bandits: Unknown Graph and Soft Interventions," 2024.
- *RL algorithm executed:* GA-LCB — graph-aware LCB-UCB algorithm with iterative causal-depth refinement. Near-minimax regret matching Õ((cd)^((L-1)/2)√T + d + RN).
- *Estimand:* Cumulative regret in linear causal bandits with unknown DAG and soft stochastic interventions.
- *Identification:* Linear SEM with known in-degree but unknown graph.
- **Verdict: KEEP — primary, §4.** Strict generalisation of Varici et al (entry 34) — same authors' lab, dispenses with the "graph known" assumption. Cite both: Varici as the known-graph baseline, Yan as the unknown-graph extension.

**37. forney2017datafusion** — Forney, Pearl, Bareinboim, "Counterfactual Data-Fusion for Online RL," ICML 2017.
- *RL algorithm executed:* Augmented Thompson Sampling that updates Beta posteriors using counterfactual quantities E[Y_{a'}|a, x] computed from both experimental data and the agent's own logs.
- *Estimand:* Counterfactual expected reward for actions the agent did not take.
- *Identification:* Pearl's counterfactual axioms applied across distinct data sources (counterfactual data fusion under unobserved confounding).
- **Verdict: KEEP — primary, §4.** The chapter cites Bareinboim 2015 MABUC but not the Forney-Pearl-Bareinboim 2017 follow-up that extends MABUC's counterfactual reasoning to data-fusion across observational + experimental + counterfactual sources. Strengthens §4's MABUC discussion with the data-fusion angle.

### Wave 4 — §1 Healthcare DTRs (8 acquired; ADT²R / Jeon paywalled)

**19. kaushik2022sepsiscql** — Kaushik, Kummetha, Moodley, Bapi, "CQL for Sepsis Distribution Shift," 2022.
- *RL algorithm executed:* Conservative Q-Learning (CQL) on MIMIC-III sepsis (47-dim state, 5×5 vasopressor/IV-fluid action grid, SOFA-/lactate-shaped intermediate reward).
- *Estimand:* Optimal sepsis-treatment policy value.
- *Identification:* Sequential ignorability assumed given EHR covariates.
- **Verdict: KEEP — primary, §1.** Currently §1 has zero deep-offline-RL-on-sepsis citations. CQL is the offline-RL algorithm of choice for distribution shift; this paper applies it to sepsis with a proper FQE evaluation. Worth a paragraph as the canonical "Komorowski-successor" citation.

**20. killian2022cftransfer** — Killian, Ghassemi, Joshi, "Counterfactually Guided Off-policy Transfer in Clinical Settings," CHIL 2022.
- *RL algorithm executed:* CFPT — counterfactually augmented offline RL using a Gumbel-max SCM on a sepsis simulator; KL-regularised target-policy refinement from a source policy.
- *Estimand:* Cross-site counterfactual policy value under domain shift + unobserved confounding.
- *Identification:* SCM with Gumbel-max counterfactuals (built on Oberst & Sontag 2019).
- **Verdict: KEEP — primary, §1.** Real-world cross-site transfer is a load-bearing problem the chapter doesn't currently address. The SCM-counterfactual-augmentation idea is also the bridge to Wave 5's cross-cutting model-based-counterfactual-RL cluster (Buesing, Lu, Oberst). Place in §1, but reference cross-cutting subsection.

**21. zhang2023ctdt** — Zhang, Mei, Xu, "Continuous-Time Decision Transformer for Healthcare," AISTATS 2023.
- *RL algorithm executed:* CTDT — causally-masked transformer trained autoregressively on return-to-go and observed states; predicts both treatment action and next-visit timing in continuous time.
- *Estimand:* Optimal sequential treatment + visit timing for chronic conditions (HIV, kidney transplant).
- *Identification:* Sequential ignorability given recorded history; non-Markovian via transformer context.
- **Verdict: MARGINAL — §1 footnote.** Real RL but the contribution is on continuous-time sequence modelling, not on causal-RL machinery. The chapter's thesis is RL-for-CI, not deep-RL benchmarks. Worth a one-line footnote acknowledging "decision transformers also extended to continuous-time DTRs (Zhang et al 2023)" in §1's discussion of methodological variants.

**22. kondrup2023deepvent** — Kondrup et al, "DeepVent: Safe Mechanical Ventilation via Deep Offline RL," AAAI 2023.
- *RL algorithm executed:* CQL with deep Q-network + clinically-shaped Apache II intermediate reward; off-policy evaluation via FQE; MIMIC-III mechanical ventilation.
- *Estimand:* Optimal ventilator-setting policy for 90-day survival.
- *Identification:* Sequential ignorability given recorded ICU covariates.
- **Verdict: KEEP — primary, §1.** Pair with Kaushik (sepsis) to cover the two main MIMIC-III DTR application domains (sepsis + ventilation). The intermediate-Apache-II-reward design is the kind of practitioner-detail the chapter's §1 currently lacks.

**24. roggeveen2024icmrl** — Roggeveen et al, "RL for Intensive Care Medicine: Cross-OPE + Policy Restriction," Intensive Care Medicine Experimental 2024.
- *RL algorithm executed:* Dueling Double-Deep Q-Network with hyperparameter grid search (69,120 models trained); novel cross-OPE evaluation across reward weightings; policy restriction; delta-Q metric.
- *Estimand:* Optimal PEEP / FiO₂ policy for COVID-19 ventilated ICU patients (Dutch ICU Data Warehouse).
- *Identification:* Sequential ignorability + cross-OPE robustness check.
- **Verdict: KEEP — primary, §1.** Strong methodological-rigor paper: 69k trained models, cross-OPE robustness, policy restriction for safety. The cross-OPE idea (varying reward weightings to identify robust policies) is a methodological innovation worth citing. Pairs with Luo (entry 25) as the "RL DTRs need very careful evaluation" pair.

**25. luo2024dtrbench** — Luo, Pan, Watkinson, Zhu et al, "DTR-Bench," ICML 2024.
- *RL algorithm executed:* Systematic benchmark of D3QN, BCQ, CQL, IQL, Decision Transformer across 4 simulation environments (sepsis, glucose, anaesthesia, oncology).
- *Estimand:* DTR welfare across multiple medical simulators with PK/PD variability, noise, missing data.
- *Identification:* Sequential ignorability assumed; explicit critique of when this fails.
- **Verdict: KEEP — primary, §1.** Position paper + benchmark. The chapter's §1 currently doesn't acknowledge the empirical fragility of RL DTRs under realistic noise/missing-data conditions. Cite as the "but watch out — algorithms degrade outside clean assumptions" caveat in §1's closing paragraph.

**26. liu2024oicrl** — Fang, Liu, Gong, "Offline Inverse Constrained RL: Constraint Transformer," 2024 (bib had wrong first-author attribution).
- *RL algorithm executed:* Constraint Transformer — non-Markovian transformer with causal attention for constraint inference; generative world model for exploratory data augmentation; offline ICRL on sepsis.
- *Estimand:* Constrained DTR / safe-action regions inferred from clinician demonstrations.
- *Identification:* IRL recovers the clinician's latent reward + constraint specification from observational EHR data.
- **Verdict: KEEP — primary, §1.** Adds the IRL angle that the chapter currently lacks. Real RL (transformer) + real causal machinery (constraint inference). The "physician demonstrations imply constraints, not just rewards" framing is a useful contribution to §1's broader discussion of what RL on clinician data can recover.

**27. hcis2025safesepsis** — Tu, Luo, Pan, Wang, Su, Zhang, Wang, "Offline Safe RL for Sepsis: Variable-Length Episodes with Sparse Rewards," Human-Centric Intelligent Systems 2025.
- *RL algorithm executed:* CQL with Apache-II-shaped intermediate rewards on variable-length MIMIC-III episodes.
- *Estimand:* Safe optimal sepsis treatment policy with bounded action-distribution shift.
- *Identification:* Sequential ignorability given recorded ICU covariates.
- **Verdict: CUT.** Same recipe as Kaushik (entry 19) and DeepVent (entry 22): CQL on MIMIC sepsis with intermediate clinical reward. Adds nothing new the prior two don't cover. Could be MARGINAL footnote ("see also Tu et al 2025 for a variable-length-episode variant") if §1 wants to gesture at the volume of work in this space; otherwise skip.

**(23. wang2024adt2r — STILL MISSING.** IEEE TNNLS paywall. If recovered later, expected verdict KEEP for §1 as the "decision transformer for sepsis with adaptive return conditioning" reference, but not load-bearing — Kaushik + DeepVent + Killian already cover the §1 deep-RL-on-MIMIC story.)

### Wave 5 — Cross-cutting: deconfounded RL, counterfactual SCMs, OPE under hidden confounding (7 papers)

**38. buesing2019woulda** — Buesing, Weber, Vinyals, Heess, Racanière et al, "Woulda Coulda Shoulda: Counterfactually-Guided Policy Search," ICLR 2019.
- *RL algorithm executed:* CF-GPS — model-based policy search where the world model is an explicit POMDP-SCM; abduction-action-prediction counterfactual rollouts feed a policy-improvement step; generalises Guided Policy Search and Stochastic Value Gradients.
- *Estimand:* Counterfactual reward of an alternative policy on the same individual episodes.
- *Identification:* Pearl's three-step counterfactual semantics on a postulated SCM.
- **Verdict: KEEP — primary, NEW SUBSECTION (Counterfactual model-based RL via SCMs).** Foundational paper for the SCM-counterfactual-RL line. Anchors a new subsection alongside Lu 2020 CF data aug and Oberst 2019. The chapter currently has no SCM-counterfactual material.

**39. lu2018deconfoundingrl** — Lu, Schölkopf, Hernández-Lobato, "Deconfounding RL in Observational Settings," 2018.
- *RL algorithm executed:* DRL — actor-critic extended with a latent-variable confounder model (Wang-Blei-style deconfounder); confounder-aware policy gradient.
- *Estimand:* Optimal-policy value in confounded MDP from observational trajectories.
- *Identification:* Latent-variable model of unmeasured confounders + Pearl-style adjustment.
- **Verdict: KEEP — primary, NEW SUBSECTION (Deconfounded RL via latent-variable models, possibly merged with the POMDP/proxy subsection).** First paper to extend an actor-critic RL algorithm to the confounded-MDP setting via a latent-variable model. Pair with Yin 2022 (Wave 2) on the methodology side and Wang 2021 DOVI / Zhou 2024 two-way on the deconfounded-RL theme.

**40. lu2020cfdataaug** — Lu, Huang, Wang, Hernández-Lobato, Zhang, Schölkopf, "Sample-Efficient RL via Counterfactual-Based Data Augmentation," 2020.
- *RL algorithm executed:* Q-learning trained on real plus counterfactually-augmented trajectories; SCM with neural-net mechanisms is learned from data; identifiability of counterfactual outcomes proved under mild conditions.
- *Estimand:* Optimal-policy value via counterfactual rollouts on a learned SCM.
- *Identification:* Shared SCM across subjects with abduction-action-prediction.
- **Verdict: KEEP — primary, NEW SUBSECTION (Counterfactual model-based RL via SCMs).** Sister paper to Buesing CF-GPS — Buesing assumed SCM is given; Lu 2020 learns it. Together they cover the "SCM-given vs SCM-learned" axis. Pair with Oberst (Gumbel-Max SCM, the discrete-state foundational paper) for a complete subsection.

**41. wang2021dovi** — Wang, Yang, Wang, "Provably Efficient Causal RL with Confounded Observational Data," NeurIPS 2021.
- *RL algorithm executed:* DOVI — UCBVI-style optimistic value iteration with backdoor (or frontdoor) adjustment for confounded offline data; warm-starts online exploration. Regret O(ΔH·d^(3/2)·H^(3/2)·√T) with ΔH < 1 when offline data informative.
- *Estimand:* Optimal-policy value in confounded MDP, online learning warm-started by deconfounded observational data.
- *Identification:* Backdoor or frontdoor criterion on a known causal graph between confounder and action.
- **Verdict: KEEP — primary, NEW SUBSECTION (Deconfounded RL via latent-variable / graph adjustment).** The "warm-start online exploration with deconfounded offline data" framing is novel and not covered anywhere in the chapter currently. Pair with Lu 2018 deconfounding RL (offline-only) as the offline-vs-warm-start contrast.

**42. bennett2021latentconf** — Bennett, Kallus, Li, Mousavi, "OPE in Infinite-Horizon RL with Latent Confounders," AISTATS 2021.
- *RL algorithm executed:* Optimal-balance estimation of stationary state-occupancy ratio under MDPUC (Markov decision process with unmeasured confounding) + iid-confounder assumption + DICE-style learning.
- *Estimand:* Long-run policy value in MDPUC with proxies for latent confounders.
- *Identification:* Latent-variable model of confounders + iid assumption + ergodicity/mixing.
- **Verdict: KEEP — primary, fits the proposed POMDP/proxy NEW SUBSECTION.** Bennett 2021 is the iid-confounder version; Bennett 2024 proximal RL (Wave 1, entry 7) is the general POMDP version. Group all three Bennett-Kallus + Shi-Uehara papers in the POMDP-OPE-with-proxies subsection as the "stack" of proxy-based identification strategies for OPE under unobserved confounding.

**43. zhou2024twoway** — Yu, Fang, Peng, Zhou, Shi, Qi, "Two-way Deconfounder for OPE in Causal RL," 2024.
- *RL algorithm executed:* Neural tensor world model jointly learning two-way fixed-effect latent confounders (time-specific + trajectory-specific) and system dynamics; model-based OPE estimator on the deconfounded model.
- *Estimand:* Policy value in confounded MDP with two-way unmeasured confounding.
- *Identification:* Two-way fixed-effects assumption (panel-data-style) on latent confounders.
- **Verdict: KEEP — primary, NEW SUBSECTION (Deconfounded RL via latent-variable models).** Different identification strategy than proxy/bridge or backdoor/frontdoor — uses panel-data 2FE assumption instead. Worth citing in the deconfounded-RL subsection as the "what if you have within-trajectory and within-time variation" alternative.

**44. oberst2019gumbelmax** — Oberst & Sontag, "Counterfactual OPE with Gumbel-Max Structural Causal Models," ICML 2019.
- *RL algorithm executed:* Counterfactual OPE pairing FQE with a Gumbel-Max SCM for discrete transitions; sepsis simulator + MIMIC-derived state space; counterfactual stability theorem.
- *Estimand:* Individual-level counterfactual outcomes under alternative policies in discrete-action MDPs.
- *Identification:* Gumbel-Max SCM gives an explicit (assumption-named) counterfactual posterior; not point-identified without monotonicity but assumption is named and defensible.
- **Verdict: KEEP — primary, NEW SUBSECTION (Counterfactual model-based RL via SCMs).** Foundational discrete-state SCM-counterfactual-RL paper. Anchors the subsection alongside Buesing (continuous, given SCM) and Lu 2020 (continuous, learned SCM). Killian 2022 (Wave 4 §1) is the Gumbel-max-application-to-cross-site-transfer follow-up; cite Oberst as the methodological origin and Killian as the deployment.

---

## Tally

- **KEEP:** 36 papers (clear primary citations)
- **MARGINAL:** 4 papers (footnotes / one-line asides)
- **CUT:** 3 papers (no fit; reasons documented above)
- **MISSING:** 1 paper (paywalled — wang2024adt2r / Jeon)

By section assignment of KEEP papers:

| Existing § | New KEEP additions |
|---|---|
| §1 Healthcare DTRs | 6 (kaushik, killian, kondrup, roggeveen, luo, liu/Fang) |
| §2 Dynamic DML | 4 (kallus2020doublerl, kallus2022curse, kallus2020policygrad, brunssmith2022uncertain — last one routed to NEW sensitivity subsection) |
| §3 Dynamic offline policy | 1 (yin2022deconfoundingac → §1 actually) |
| §4 Causal bandits | 9 (lu2020causalbg, lu2022causalmdp, yabe, dekroon, feng, varici, sussex, yan, forney) |
| §5 Adaptive experiments | 0 new — section already well-covered |

| New subsection | Anchor papers |
|---|---|
| **POMDP/proxy OPE** | shi2022confoundedpomdp, bennett2024proximalrl, uehara2023futuredep, shi2024confoundedci, bennett2021latentconf |
| **Sensitivity / IV / partial-id** | liao2021ivvi, fu2022offlineiv, kallus2020confoundingrobust, namkoong2020unobserved, brunssmith2022uncertain, hess2025sharp (footnote) |
| **Counterfactual model-based RL via SCMs** | buesing2019woulda, oberst2019gumbelmax, lu2020cfdataaug; cross-ref killian2022cftransfer (in §1) |
| **Deconfounded RL via latent-variable models** | lu2018deconfoundingrl, wang2021dovi, zhou2024twoway |

---

## Part B — Per-existing-section recommendations

### §1 Dynamic Treatment Regimes — Add 6 papers, no structural change

The current §1 (lines 22–94 of `rl_for_ci.tex`) tells the Murphy-2003 → Watkins-1992 bridge story and runs a Q-learning vs Murphy-batch sim. It does not engage with contemporary deep-offline-RL deployments. Add a new closing paragraph (after the dictionary table around line 91, before the §2 break) titled "Real-world deep-RL deployments" that integrates:

1. **kaushik2022sepsiscql** + **kondrup2023deepvent** — the canonical CQL-on-MIMIC pair (sepsis + ventilation), establishing what "doing the chapter's bridge" looks like at scale on real data.
2. **killian2022cftransfer** — cross-site counterfactual transfer using a Gumbel-max SCM. This is the natural place to cross-reference the new "Counterfactual model-based RL via SCMs" subsection.
3. **roggeveen2024icmrl** + **luo2024dtrbench** — the methodological-rigor pair: cross-OPE robustness check (Roggeveen) and the position paper showing RL DTRs are fragile under realistic noise/missing data (Luo). Together: "the chapter's bridge runs, but practitioners must be more careful than naive deployments suggest."
4. **liu2024oicrl** (Fang) — the IRL/safety angle, recovering a clinician's latent reward + constraint specification rather than just imitating actions. One paragraph; flags the IRL-of-RL story without going deep.
5. **yin2022deconfoundingac** — actor-critic with internal deconfounding module on MIMIC. Demonstrates that the "deconfounded RL" idea (Wave 5 cluster) is also being deployed on real EHR data, not just synthetic benchmarks.

Footnote-only: **zhang2023ctdt** (continuous-time decision transformer — methodologically interesting, light on causal-RL machinery). Cut: **hcis2025safesepsis** (duplicates Kaushik + DeepVent).

Nothing in current §1 needs cutting. The existing Murphy-Watkins bridge story remains the chapter's anchor; the new paragraph just adds the contemporary deployment context after the dictionary table.

### §2 Dynamic DML / SNMMs — Add 4 KEEP papers, restructure mid-section

Current §2 (lines 96–151) is the chapter's weakest section: the entire Lewis-Syrgkanis discussion runs without ever pointing to an actual deep-RL OPE estimator that inherits the √n rate. The fix has three parts:

1. **Insert kallus2020doublerl + kallus2022curse mid-section** as the load-bearing additions. Currently §2's last paragraph (around line 150) calls Lewis-Syrgkanis "the orthogonal-score version of fitted-Q evaluation" — a relabeling. Replace this with a real bridge: introduce DRL (Kallus-Uehara 2020 finite-horizon, 2022 infinite-horizon), explain that DRL is the actual deep-RL OPE estimator inheriting √n via cross-fold q + density-ratio estimation, and note that Lewis-Syrgkanis is the SNMM-parameter version of the same orthogonality story.
2. **Add kallus2020policygrad as a one-paragraph extension at end of §2** showing the bridge from OPE to off-policy policy *gradient* learning (extends DML orthogonality from evaluation to optimisation; not just policy value but its derivative). This pre-figures §3.
3. **Footnote cluster** for technical detail: **uehara2021minimax** (finite-sample minimax rates with general function approximation, completeness conditions); **nachum2019dualdice** (DICE-family origin); **hao2021bootstrapfqe** (bootstrap as alternative route to CIs). All can fit in two footnotes total.

Routed elsewhere: **shi2022confoundedpomdp**, **bennett2024proximalrl**, **uehara2023futuredep** all go to the new POMDP/proxy subsection (different identification regime, doesn't fit §2's sequential-ignorability framing). **brunssmith2022uncertain** goes to the new sensitivity subsection.

What to cut from current §2: the "labeled relabeling" sentences in line 150 (the chapter's own admission that it's calling DML "fitted-Q evaluation"). Replace with the genuine DRL bridge.

### §3 Dynamic Offline Policy Learning — Minimal addition; redirect most papers to new subsections

Current §3 (lines 153–195) is the second weakest section. Most KEEP candidates from Wave 2 actually belong in new subsections (sensitivity/IV) rather than expanding §3 itself.

Add to §3 directly: **yin2022deconfoundingac** as the contemporary deep-deconfounded-actor-critic deployment paper (one paragraph, possibly with cross-ref to the §1 deployment paragraph and to the new Deconfounded RL subsection).

Routed elsewhere: liao2021ivvi, fu2022offlineiv, kallus2020confoundingrobust, namkoong2020unobserved, brunssmith2022uncertain, hess2025sharp → new Sensitivity / IV subsection. shi2024confoundedci → new POMDP/proxy subsection. liu2020batchoffpolicy → CUT.

What to cut from current §3: nothing structural. The Sakaguchi backward-induction-AIPW story remains the §3 anchor; new material lives in adjacent subsections.

### §4 Causal Bandits — Add 9 papers, expand the section meaningfully

§4 (lines 197–235) is the strongest existing section but covers only Bareinboim-2015 MABUC and Lattimore-2016 parallel-bandit. Nine new KEEP papers expand this to a coherent story:

1. **lu2020causalbackground** — first paragraph after the existing Lattimore discussion, resolving Lattimore's open problem on cumulative regret with C-UCB / C-TS.
2. **forney2017datafusion** — extending the MABUC discussion to data fusion across observational/experimental/counterfactual sources (one paragraph after Bareinboim).
3. **yabe2018propagating** + **feng2023combinatorial** — extension paragraph: arbitrary propagating interventions + combinatorial causal bandits (intervene on K vars at once).
4. **dekroon2022separating** — relax the "graph known" assumption; one paragraph on causal discovery interleaved with bandit learning.
5. **varici2022linearsem** + **yan2024linearcb** — soft-intervention paragraph: linear SEMs with both known-graph (Varici) and unknown-graph (Yan) variants.
6. **sussex2023mcbo** — closing paragraph or footnote linking causal bandits to causal Bayesian optimisation in the continuous-variable / GP-modelled mechanism setting.
7. **lu2022causalmdp** — final extension paragraph or §4 closing transition: from single-stage causal bandits to causal MDPs (sequential causal bandits).

This roughly doubles §4's length but with material that's all genuinely §4-shaped (no taxonomic cheating). The current Sim-2 sub-experiment (lines 311–338) needs no change — it already runs Lattimore Algorithm 1 + MABUC Causal TS.

### §5 Adaptive Experimentation — No new additions

Wave 1–5 produced no Wave-5-coverage papers. The §5 gap (deep-RL design + post-experiment inference) genuinely remains, as flagged in the original bibliography. Existing citations (Kasy-Sautmann + Hadad + Bibaut) remain the load-bearing trio. Nothing to add or cut here.

---

## Part C — New subsection proposals

The chapter has 5 subsections + sim study + discussion (~360 lines, ~25 PDF pages double-spaced). Adding 4 new subsections would unbalance it; merging is required. Recommendation: add **2 new subsections** plus **1 paragraph** within an existing section.

### NEW SUBSECTION 1: "Off-Policy Evaluation Under Unobserved Confounding"

Place between current §3 (Dynamic Offline Policy Learning) and current §4 (Causal Bandits). This is the natural location: §1–§3 all assume sequential ignorability; this new subsection is "what happens when sequential ignorability fails"; §4 then pivots to the bandit setting where the failure has different solutions.

**Scope:** 4 paragraphs, ~2 PDF pages, covering three identification regimes for OPE in confounded MDPs: (a) sensitivity models (Tan-style bounds), (b) instrumental variables, (c) proxy / bridge functions / front-door. Each regime gets one paragraph; final paragraph is a comparative discussion.

**Anchor papers (organised by regime):**
- *Sensitivity models:* **kallus2020confoundingrobust** (stationary infinite-horizon MSM), **namkoong2020unobserved** (one-decision MSM, sepsis demo), **brunssmith2022uncertain** (per-period iid model + robust MDP), with **hess2025sharp** as footnote (single-stage sharp bound).
- *Instrumental variables:* **liao2021ivvi** (finite-horizon IV-aided VI), **fu2022offlineiv** (infinite-horizon IV with welfare-regret + DR).
- *Proxy / bridge functions:* **shi2022confoundedpomdp** (minimax bridge learning), **bennett2024proximalrl** (proximal RL identification + sepsis), **bennett2021latentconf** (MDPUC iid case + DICE), **uehara2023futuredep** (future-dependent value, footnote — partial observability not strictly confounding), **shi2024confoundedci** (front-door / mediator identification, JASA — alternative).

**Why this is a new subsection rather than expansion of §3:** §3 is about *policy learning* under sequential ignorability. The new subsection is about *evaluation* (and learning) when sequential ignorability fails. Different problem; different identification machinery; would cluttering §3 to merge.

### NEW SUBSECTION 2: "Counterfactual Reasoning with Structural Causal Models"

Place between the new "OPE Under Unobserved Confounding" subsection and §4. Or alternatively, fold into §4 as a methodological aside if space is tight.

**Scope:** 3 paragraphs, ~1.5 PDF pages, on SCM-based counterfactual reasoning for both OPE and policy learning. Distinct from the proxy/bridge approach (which uses observed proxies for unobserved confounders) — SCMs use named structural assumptions (Gumbel-Max for discrete, latent-variable models for continuous) to generate counterfactual rollouts.

**Anchor papers:**
- **buesing2019woulda** (CF-GPS — given SCM, abduction-action-prediction, GPS extension)
- **oberst2019gumbelmax** (Gumbel-Max SCM for discrete-state OPE, sepsis simulator, counterfactual stability theorem)
- **lu2020cfdataaug** (learned SCM with neural mechanisms, counterfactual data augmentation, Q-learning identifiability)
- Cross-ref **killian2022cftransfer** (in §1) and **lu2018deconfoundingrl** + **wang2021dovi** + **zhou2024twoway** (which form a sub-cluster on "deconfounded RL via latent-variable models" — could be a paragraph within this subsection or merged with the proxy subsection above).

**Why this is a new subsection rather than expansion of §1 or §3:** SCM-counterfactual machinery is methodologically distinct from both the Murphy DTR bridge and the AIPW backward-induction line. The chapter currently has zero SCM-counterfactual material. This is the cleanest cluster among the 7 cross-cutting Wave-5 papers.

### NEW PARAGRAPH (not a subsection): "Online RL warm-started by deconfounded observational data"

Place at the end of §3 (Dynamic Offline Policy Learning) or start of the new "OPE Under Unobserved Confounding" subsection. One paragraph.

**Anchor papers:** **wang2021dovi** (DOVI with backdoor/frontdoor adjustment on offline data, online UCBVI with deconfounded warm-start), with cross-ref to **fu2022offlineiv** (which similarly bridges offline to online via IVs).

**Why one paragraph and not a subsection:** This is a narrow methodological theme (offline → online warm-start under confounding) with only 1–2 load-bearing papers; doesn't warrant its own subsection but is too important to omit entirely from a chapter framed as "RL for CI."

### Sections NOT recommended as new

I considered three other clusters and rejected them:

- *"Deconfounded RL via latent-variable models" as a standalone subsection* — papers (lu2018deconfoundingrl, wang2021dovi, zhou2024twoway) overlap thematically with the proxy/bridge cluster (both are "what to do under hidden confounders"). Merging into NEW SUBSECTION 1 above as a final paragraph or fourth-regime-discussion is cleaner.
- *"Real-world deployments of RL in healthcare" as a new subsection* — papers (kaushik, killian, kondrup, roggeveen, luo, liu/Fang, yin) all fit naturally in §1's "real-world deployment" closing paragraph. Promoting to a separate subsection would over-emphasise §1 relative to the chapter's RL-for-CI thesis.
- *"OPE inference (CIs / bootstrap)" as a new subsection* — only hao2021bootstrapfqe and shi2024confoundedci fit; the latter belongs in NEW SUBSECTION 1 (proxy/front-door identification) and the former is a §2 footnote. Not enough mass for a subsection.

---

## Verification

Spot-check three KEEP verdicts at random:

1. **kallus2020doublerl** — Re-checked `papers/kallus2020doublerl.md` lines 11, 37: confirms DRL = cross-fold q-function + marginalised density ratio estimator achieving semiparametric efficiency in MDPs. RL algorithm present (DRL). Verdict KEEP justified. ✓
2. **lu2020causalbackground** — Re-checked `papers/lu2020causalbackground.md` lines 11, 37: confirms C-UCB and C-TS algorithms with cumulative regret bounds Õ(√(k+1)^n T) and explicit resolution of Lattimore's open problem. Real RL (UCB/TS) + real causal structure. Verdict KEEP justified. ✓
3. **buesing2019woulda** — Re-checked `papers/buesing2019woulda.md` lines 9, 17–19: confirms CF-GPS = model-based policy search with SCM-based counterfactual rollouts (abduction-action-prediction). Real RL (policy search) + real causal (SCM). Verdict KEEP justified. ✓

Duplication check on §2 KEEP recommendations: chapter currently cites Lewis & Syrgkanis 2021 (SNMM dynamic-DML), Chernozhukov et al 2023 (recursive Riesz). New §2 KEEP additions are kallus2020doublerl (deep-RL OPE), kallus2022curse (infinite-horizon DRL), kallus2020policygrad (off-policy policy gradient). None overlaps. ✓

Out of scope for this evaluation pass (per plan): chapter rewrites, BibTeX entries, prose drafts. Those follow in subsequent passes.
