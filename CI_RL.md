# Beyond the Current Chapter: A Topic-Cluster Map for Causal Inference × Reinforcement Learning

## TL;DR
- The survey chapter already covers the "Pearl–Bareinboim" pillar of causal RL (confounded MDPs, IV/proxy OPE, counterfactual SCM‑based policies, transportability). The biggest gaps are on the **econometrics/statistics side of the bridge** — DML for dynamic treatment effects, sensitivity-analysis OPE, policy learning from observational data, and the historically central marginal-structural-models / g-methods framework that economists already know — plus a handful of model-based and credit-assignment additions on the RL side.
- The single most important addition for an economist audience is **Lewis & Syrgkanis (NeurIPS 2021), "Double/Debiased Machine Learning for Dynamic Treatment Effects"** (the proceedings title; the arXiv version is subtitled "via g-Estimation"). It is *the* canonical DML×RL paper: a recursively-orthogonalised version of Robins' g-estimation that yields √n inference on structural nested mean model (SNMM) blip parameters and — in its extended arXiv v5 abstract — explicitly states these "structural parameters can be used for off-policy evaluation of any target dynamic policy at parametric rates."
- Below I identify **10 topic clusters** with one canonical paper each plus 2–4 representatives, and recommend the **top four** to add given an 80/20 budget: (i) DML for dynamic treatment effects (Lewis & Syrgkanis 2021), (ii) Robins g-methods / MSMs as the econometric precursor to OPE (Robins, Hernán & Brumback 2000; Murphy 2003), (iii) sensitivity analysis / partial identification in sequential OPE (Namkoong, Keramati, Yadlowsky, Brunskill 2020; Kallus & Zhou 2020), and (iv) policy learning from observational data (Athey & Wager, Econometrica 2021).

## Key Findings
The literature splits cleanly into the two directions the user identified, and there is one critical "third rail" — DML — that bridges them. The clusters are:

1. **DML for sequential causal inference / OPE** (THE critical addition)
2. **Sensitivity analysis & partial identification for OPE under unmeasured confounding**
3. **Policy learning from observational data** (a.k.a. batch policy learning, the "L" in OPE/L)
4. **Marginal structural models, g-methods, and DTRs as the econometric precursor to RL**
5. **Causal world models & local causal structure for model-based RL**
6. **Counterfactual / hindsight credit assignment beyond Buesing–Oberst**
7. **Causal influence detection for exploration**
8. **Causal bandits** (a small but rock-solid sub-literature with regret theorems)
9. **Causal imitation learning** (sequential causal IL with hidden confounding)
10. **Adaptive experimentation & post-bandit inference** (RL meets adaptive trial design)

Plus three "honourable mentions" the chapter might add as one-paragraph notes: distributionally-robust OPE (Si et al. 2020; Kallus et al. 2022), conformal OPE (Taufiq et al. 2022; Y. Zhang–Shi–Luo 2023), and dynamic mediation analysis under an RL framework (Luo, Shi, Wang, Wu, Li, Annals of Statistics 2025).

## Details

### Cluster 1 — Double/Debiased ML for Dynamic Treatment Effects [THE critical DML×RL paper]

**Canonical paper.** Greg Lewis & Vasilis Syrgkanis (2021), *"Double/Debiased Machine Learning for Dynamic Treatment Effects,"* NeurIPS 2021 (the arXiv version, arXiv:2002.07285, carries the additional subtitle "via g-Estimation"). https://arxiv.org/abs/2002.07285 ; https://proceedings.neurips.cc/paper_files/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html

**Core mathematical idea.** Take Robins' (1994, 2004) g-estimation for structural nested mean models — a panel SNMM in which at each period $t = 1, \dots, m$ a "blip" function $\theta_t$ encodes the marginal causal effect of a treatment "blip" $T_t$ holding future treatments at a reference level, given history $\bar X_t = (X_1, \dots, X_t)$ and $\bar T_{t-1}$. Lewis & Syrgkanis introduce **sequential (recursive) residualisation**: at each period, residualise the calibrated outcome on history and residualise the contemporaneous treatment on history, then identify $\theta_t$ from the Robinson-style Neyman-orthogonal moment $\mathbb E[(\tilde Y_t - \theta_t \tilde T_t)\tilde T_t] = 0$, then "peel off" $\hat\theta_t T_t$ from the outcome and recurse to period $t-1$. Verbatim from the arXiv PDF: *"we propose a sequential residualization approach, where the effects at every period are estimated in a Neyman orthogonal manner and then peeled-off from the outcome, so as to define a new 'calibrated' outcome, which will be used to estimate the effect of the treatment in the previous period."* Nuisance functions (the outcome regression $q_t$ and the conditional-treatment expectations $p_{j,t}$ for $j \le t$) can be any ML estimator whose MSE product rate is $o(n^{-1/2})$ — e.g. $n^{-1/4}$ each — and the resulting $\hat\theta$ is √n-asymptotically normal.

**Memorable example/application.** A concrete linear Markovian high-dimensional state-space model: $X_t \in \mathbb R^p$ with $p$ large, $T_t \in \mathbb R$, and a linear blip $\theta_t T_t$. The paper validates √n coverage of confidence intervals in simulations and exhibits a "recursive Lasso" variant for sparse high-dimensional state. The introduction frames the motivation in digital-advertising / customer-LTV and dynamic-pricing terms — exactly the economics audience. (For a worked semi-synthetic *corporate* dataset using the same machinery, cite the companion paper Battocchi, Dillon, Hei, Lewis, Oprescu & Syrgkanis (2021) on customer-investment long-term effects.)

**Related papers in this cluster.**
- Chernozhukov, Newey, Singh & Syrgkanis (2023, *arXiv:2203.13887*), *"Automatic Debiased Machine Learning for Dynamic Treatment Effects and General Nested Functionals."* Replaces the hand-derived Neyman moment with **recursive Riesz representers**: a sequence of loss-minimisation problems whose minimisers are the multipliers of the debiasing correction, removing the need to write down closed-form IPW products. Extends from SNMMs to general nested functionals (mediation, long-term effects). Verbatim: *"We then apply a recursive Riesz representer estimation learning algorithm that estimates de-biasing corrections without the need to characterize how the correction terms look like."*
- Battocchi, Dillon, Hei, Lewis, Oprescu & Syrgkanis (NeurIPS 2021, arXiv:2103.08390), *"Estimating the Long-Term Effects of Novel Treatments."* Applies dynamic DML to a real semi-synthetic corporate dataset (three-year customer investments) via a surrogate-index approach — concrete economics example.
- Dylan J. Foster & Vasilis Syrgkanis (Annals of Statistics 2023; *Ann. Statist.* 51(3):879–908, doi:10.1214/23-AOS2258, arXiv:1901.09036), *"Orthogonal Statistical Learning."* The general theory: Neyman orthogonality gives excess-risk guarantees for two-stage ML pipelines with second-order nuisance-error impact. The mathematical foundation under all DML×RL work.
- Bibaut, Chambaz, Dimakopoulou, Kallus & van der Laan (NeurIPS 2021, arXiv:2106.00418), *"Post-Contextual-Bandit Inference."* TMLE-style adaptive cross-fitted estimators for adaptively collected data — bridges DML and adaptive experimentation.

**Why economists care.** This is the deepest cluster for an economics audience: it expresses RL/OPE problems in Robins–Robinson language (residualise, project orthogonally, peel off), uses semiparametric efficiency theory the audience already knows, and the canonical paper's extended arXiv v5 abstract explicitly states: *"These structural parameters can be used for off-policy evaluation of any target dynamic policy at parametric rates, subject to semi-parametric restrictions on the data generating process."* (This OPE sentence is in the arXiv v5 abstract only; the shorter NeurIPS 2021 proceedings abstract ends at *"This allows us to show root-n asymptotic normality of the estimated causal effects."*) The arXiv version is the right one to cite for the OPE bridge.

**Relation to chapter.** Complements (rather than overlaps with) the Kallus–Uehara Double Reinforcement Learning approach already implicit in the chapter's backdoor-adjusted OPE section. Kallus–Uehara works directly with influence functions for the policy value under MDP structure; Lewis–Syrgkanis works with SNMM blip parameters and recovers the policy value as a smooth functional. Together they form the "two faces" of doubly-robust dynamic causal inference.

---

### Cluster 2 — Sensitivity Analysis & Partial Identification for OPE under Unmeasured Confounding

**Canonical paper.** Hongseok Namkoong, Ramtin Keramati, Steve Yadlowsky & Emma Brunskill (NeurIPS 2020), *"Off-Policy Policy Evaluation for Sequential Decisions Under Unobserved Confounding."* arXiv:2003.05623. https://proceedings.neurips.cc/paper/2020/hash/da21bae82c02d1e2b8168d57cd3fbab7-Abstract.html

**Core mathematical idea.** Drop the sequential-ignorability assumption. Under a Rosenbaum-style sensitivity model bounding the odds-ratio influence of unmeasured confounders by Γ, derive **worst-case bounds** on the value $V^\pi$ of an evaluation policy. The key technical move is functional convex duality: the worst-case bound over confounding distributions reduces to a tractable loss-minimisation problem solvable on observed transitions. They prove consistency and analyse the difference between "one-decision confounding" (only the first decision is confounded) and "per-decision confounding" (every decision is confounded, where they show even tiny Γ blows the bounds up).

**Memorable example.** Two clinical simulations — sepsis ICU management (from Oberst–Sontag, also in the chapter) and SMART trial for autistic children. They demonstrate that the worst-case bounds give meaningful "robustness certificates": with their one-decision-confounding model, OPE results survive realistic Γ; with per-decision confounding, even Γ ≈ 1.5 invalidates results.

**Related papers in this cluster.**
- Kallus & Zhou (NeurIPS 2020, arXiv:2002.04518), *"Confounding-Robust Policy Evaluation in Infinite-Horizon Reinforcement Learning."* Sharp bounds in the infinite-horizon setting using stationary density ratios.
- Bruns-Smith & Zhou (NeurIPS 2023, arXiv:2302.00662), *"Robust Fitted-Q-Evaluation and Iteration under Sequentially Exogenous Unobserved Confounders."* Closed-form robust Bellman operator; orthogonalised loss-minimisation problem for the robust Q-function. Bridges sensitivity analysis with the chapter's existing causal Bellman material.
- Yadlowsky, Namkoong, Basu, Duchi & Tian (Annals of Statistics 2022), *"Bounds on the Conditional and Average Treatment Effect with Unobserved Confounding Factors."* The static foundational paper.

**Why economists care.** Sensitivity analysis is *the* econometric response to "unconfoundedness is untestable." This cluster gives the dynamic generalisation. The notation (Γ as marginal sensitivity parameter, sharp bounds, IPW reweighting) is exactly Rosenbaum-style.

**Relation to chapter.** Directly complements the chapter's IV/proxy material: when no IV or proxy is available, sensitivity analysis is the fallback. The chapter says "what if unconfoundedness fails?" and answers "use an IV"; this cluster answers "use a sensitivity model and report bounds."

---

### Cluster 3 — Policy Learning from Observational Data (Off-Policy Optimisation under Unconfoundedness)

**Canonical paper.** Susan Athey & Stefan Wager (Econometrica 2021), *"Policy Learning with Observational Data,"* Econometrica 89(1):133–161. https://doi.org/10.3982/ECTA15732

**Core mathematical idea.** Construct doubly-robust (augmented IPW) scores $\hat\Gamma_i$ for the causal effect of each treatment using cross-fitted nuisance estimates (à la Chernozhukov et al. 2018), then pick the policy in a restricted class $\Pi$ (decision trees, linear policies, budget-constrained rules) that maximises $\frac1n \sum_i \hat\Gamma_i(\pi(X_i))$. The main theorem gives $O(\sqrt{VC(\Pi)/n})$ regret bounds — matching the parametric rate of the policy-learning literature with known propensities — under generic ML rates on the nuisances. Handles selection-on-observables and IV identification strategies, binary or continuous treatments.

**Memorable example.** Targeting decisions: which customers should receive an offer (constrained-class policy), which patients should be assigned which dose (budget-constrained DTR). The Stanford GRF / EconML pipeline is the practical embodiment.

**Related papers in this cluster.**
- Zhou, Athey & Wager (Operations Research 2023, arXiv:1810.04778), *"Offline Multi-Action Policy Learning: Generalization and Optimization."* Extends to multiple discrete treatments with budget/fairness constraints; gives mixed-integer-programming and tree-search algorithms.
- Kitagawa & Tetenov (Econometrica 2018), *"Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice."* The econometric precursor with known propensities.
- Kallus & Zhou (2018, arXiv:1805.08593), *"Confounding-Robust Policy Improvement."* Adds Rosenbaum-style sensitivity to the policy-optimisation step.
- Athey, Chernozhukov, Kallus, Spindler & Syrgkanis (2024, arXiv:2403.02467), *"Applied Causal Inference Powered by ML and AI"* — the textbook treatment.

**Why economists care.** This is *the* offline policy-learning paper in economics. It builds directly on semiparametric efficiency theory the audience uses daily, has clean √n regret theorems, and explicitly allows IV-based identification — distinguishing it from pure RL approaches.

**Relation to chapter.** The chapter's backdoor-adjusted OPE section ends at *evaluation*. Athey–Wager closes the loop to *optimisation*. Together with Kallus–Uehara's DRL (evaluation) they constitute the OPE/L pair the econometrics audience expects.

---

### Cluster 4 — Marginal Structural Models, g-Methods & DTRs (the econometric precursor to OPE the chapter is missing)

**Canonical paper.** James M. Robins, Miguel Á. Hernán & Babette Brumback (Epidemiology 2000), *"Marginal Structural Models and Causal Inference in Epidemiology,"* 11(5):550–560.

**Core mathematical idea.** Under a sequential-randomisation (no-unmeasured-confounding) assumption and longitudinal positivity, the average potential outcome under a treatment history $\bar a$ is identified by the **g-computation formula** $\mathbb E[Y(\bar a)] = \int \mathbb E[Y \mid \bar A = \bar a, \bar L = \bar l] \prod_t f(l_t \mid \bar l_{t-1}, \bar a_{t-1}) d\bar l$. **Marginal structural models** parameterise this counterfactual mean directly and estimate parameters via **inverse-probability-of-treatment weighting**, with weights given by products of propensity ratios. The trio of g-methods — g-formula, IPTW for MSMs, and g-estimation for structural nested models — is the foundational toolkit Robins introduced in 1986–2000.

**Memorable example.** HIV antiretroviral treatment in the original paper: time-varying CD4 count both confounds and is affected by past treatment, so standard regression is biased. MSM/IPTW correctly recovers the effect.

**Related papers in this cluster.**
- S. A. Murphy (JRSS-B 2003), *"Optimal Dynamic Treatment Regimes."* The first formal definition of an optimal DTR; Q-learning and A-learning recursions.
- Chakraborty & Moodie (2013 book, Springer), *Statistical Methods for Dynamic Treatment Regimes.* The reference text.
- Schulte, Tsiatis, Laber & Davidian (Statistical Science 2014), *"Q- and A-learning Methods for Estimating Optimal Dynamic Treatment Regimes."* Bridges DTR/biostatistics to RL terminology.
- Luckett, Laber, Kahkoska, Maahs, Mayer-Davis & Kosorok (JASA 2020), *"Estimating Dynamic Treatment Regimes in Mobile Health Using V-learning."* Modern RL-style DTR estimation in mHealth.

**Why economists care.** This is the **econometric/biostatistical precursor of off-policy evaluation that economists should already know exists**. Q-learning *is* dynamic programming on conditional means; A-learning *is* g-estimation; OPE *is* MSM-style IPTW; FQE *is* sequential outcome regression. Showing this equivalence in econometric notation is the highest-pedagogical-value move the chapter can make.

**Relation to chapter.** Provides the missing historical/notational scaffold. Every modern RL OPE estimator has a Robins-era twin; making that explicit lets economists "skip the new vocabulary and read the math."

---

### Cluster 5 — Causal World Models & Local Causal Structure for Model-Based RL

**Canonical paper.** Silviu Pitis, Elliot Creager & Animesh Garg (NeurIPS 2020), *"Counterfactual Data Augmentation Using Locally Factored Dynamics."* arXiv:2007.02863.

**Core mathematical idea.** Many control environments have a **global** causal model in which the per-step dynamics decompose into **locally independent causal mechanisms** when conditioned on a subset of state. Formally, the global SCM induces "local causal models" (LCMs) by conditioning on values that break the dependence between subprocesses. The authors learn these LCMs from object-centric representations and use them for **Counterfactual Data Augmentation (CoDA)**: replay-buffer transitions are recombined across trajectories whenever the local independence guarantees they would be causally valid under the global SCM, yielding sample-efficient off-policy RL.

**Memorable example.** A robotic-arm manipulation environment with multiple objects — the arm's interaction with object A is locally independent of object B's state, so a transition observed with object B in configuration $b_1$ can be transferred to a trajectory with $b_2$ and remain a valid global counterfactual.

**Related papers in this cluster.**
- Huang, Feng, Lu, Magliacane & Zhang (ICLR 2022), *"AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning."* Learns a latent SCM with domain-shared and domain-specific factors for adaptive transfer.
- Z. Wang, X. Xiao, Z. Xu, Zhu & Stone (ICML 2022, arXiv:2206.13452), *"Causal Dynamics Learning for Task-Independent State Abstraction"* (CDL).
- Zhu, Chen, Tian, Zhang & Yu (arXiv:2206.01474), *"Offline Reinforcement Learning with Causal Structured World Models."* Proves causal world models dominate plain world models for offline RL under distribution shift.

**Why economists care.** Local causal structure is the operational answer to: "I have a structural model but estimating the full joint distribution is hopeless — can I do counterfactuals only on the subgraph I care about?" Same logic as partial identification.

**Relation to chapter.** The chapter's brief causal-representation-learning section (Schölkopf et al. 2021, da Costa Cunha 2025) points in this direction but stops short of the **decision-relevant** counterfactual-data-augmentation idea. CoDA is the cleanest worked example.

---

### Cluster 6 — Counterfactual / Hindsight Credit Assignment Beyond Buesing–Oberst

**Canonical paper.** Mesnard, Weber, Viola, Thakoor, Saade, Harutyunyan, Dabney, Stepleton, Heess, Guez, Moulines, Hutter, Buesing & Munos (ICML 2021), *"Counterfactual Credit Assignment in Model-Free Reinforcement Learning,"* PMLR 139. arXiv:2011.09464. https://proceedings.mlr.press/v139/mesnard21a.html

**Core mathematical idea.** Standard policy-gradient methods estimate $\nabla V^\pi$ using averages over futures, paying no attention to which actions actually caused which outcomes. **Counterfactual Credit Assignment (CCA)** conditions value/critic functions on **future statistics** $\Phi_t$ — features of the trajectory after time $t$ — to produce trajectory-specific counterfactual return estimates. The key constraint is that $\Phi_t$ must be **independent of the agent's action $A_t$** under the policy, guaranteeing the policy-gradient estimator remains unbiased while typically having lower variance. The math generalises the earlier Hindsight Credit Assignment (HCA) of Harutyunyan et al. (NeurIPS 2019), which uses the "hindsight distribution" $h(a \mid x, y) / \pi(a \mid x)$ to reweight returns.

**Memorable example.** Sky-fall-style environments (and other illustrative tasks): the agent's action affects only the protagonist, while environmental noise (e.g. wind) is captured by $\Phi_t$; CCA explicitly separates "skill" from "luck," reducing variance dramatically.

**Related papers in this cluster.**
- Harutyunyan, Dabney, Mesnard, Heess, Azar, Piot, van Hasselt, Singh, Wayne, Precup & Munos (NeurIPS 2019), *"Hindsight Credit Assignment."* The originator.
- Meulemans, Schug, Kobayashi, Ferret, Sacramento (NeurIPS 2023), *"Would I Have Gotten That Reward? Long-Term Credit Assignment via Contribution Coefficients."* Refines CCA with intervention-style contribution scores.
- Pitis–Creager–Garg (CoDA, NeurIPS 2020) — also fits here as a credit-assignment-via-counterfactuals method.

**Why economists care.** This is the cleanest RL example of "treatment effect estimation inside an algorithm": each gradient step is asking *would the return have been different had I taken a different action?* — exactly an individual-level counterfactual. The architecture mirrors AIPW estimators (baseline + residual term).

**Relation to chapter.** Extends the chapter's Buesing/Oberst counterfactual-policy-optimisation material from twin-network/Gumbel-Max SCM (Atari) to **model-free** policy-gradient algorithms — a strictly broader applicability. Should sit directly after Buesing/Oberst.

---

### Cluster 7 — Causal Influence Detection for Exploration

**Canonical paper.** Maximilian Seitzer, Bernhard Schölkopf & Georg Martius (NeurIPS 2021), *"Causal Influence Detection for Improving Efficiency in Reinforcement Learning."* arXiv:2106.03443.

**Core mathematical idea.** Define a **situation-dependent causal influence** measure $\text{CAI}(s) = I(A_t; S'_t \mid S_t = s)$ — the conditional mutual information between the agent's action and a downstream entity's state, conditional on the current state. This identifies states in which the agent has *causal control* over an object. Two algorithmic uses: (i) **exploration bonus** rewarding visits to high-CAI states; (ii) **prioritised replay** weighting high-CAI transitions. Both produce strong sample-efficiency gains on robotic-manipulation benchmarks (e.g. FetchPickAndPlace).

**Memorable example.** A robot arm trying to manipulate an object: most timesteps the arm and object are not in contact and the agent has no causal influence; CAI correctly identifies the few "contact" states as the high-value learning opportunities.

**Related papers in this cluster.**
- Jaques, Lazaridou, Hughes, Gulcehre, Ortega, Strouse, Leibo & de Freitas (ICML 2019, arXiv:1810.08647), *"Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning."* CAI in MARL — rewards agents for causal influence over other agents' actions. Also the canonical causal MARL paper.
- Foerster, Farquhar, Afouras, Nardelli & Whiteson (AAAI 2018), *"Counterfactual Multi-Agent Policy Gradients"* (COMA). Counterfactual baselines for MARL credit assignment.

**Why economists care.** CAI is essentially **measuring the agent's marginal product** in causal terms. The MARL extension (social influence) corresponds to peer effects / spillovers.

**Relation to chapter.** A new entry; the chapter does not currently discuss exploration or curiosity, and CAI is the cleanest causal entry point. Doubles as the bridge to causal MARL.

---

### Cluster 8 — Causal Bandits

**Canonical paper.** Finnian Lattimore, Tor Lattimore & Mark D. Reid (NeurIPS 2016), *"Causal Bandits: Learning Good Interventions via Causal Inference."* arXiv:1606.03203. https://proceedings.neurips.cc/paper/2016/hash/b4288d9c0ec0a1841b3b3728321e7088-Abstract.html

**Core mathematical idea.** Standard MAB treats arms as opaque; in **causal bandits** the arms are interventions $do(X_i = x)$ on a known causal graph, and pulling an arm reveals not only its reward but also the **non-intervened** variables' values. The algorithm exploits this side information via importance reweighting: even arms not pulled yield information through the joint distribution. The simple-regret bound is strictly better in all problem-dependent quantities than any algorithm that ignores the graph.

**Memorable example.** A small Bayes-net with a binary outcome where $do(X_1)$, $do(X_2)$, ... are competing interventions and the parental conditional probabilities are known. With $N$ pulls, causal-bandit algorithms achieve simple regret scaling with a graph-derived quantity $m^*$ that is typically much smaller than the arm count $K$.

**Related papers in this cluster.**
- Bareinboim, Forney & Pearl (NeurIPS 2015), *"Bandits with Unobserved Confounders: A Causal Approach"* (MABUC). Originator of the unobserved-confounders bandit problem; the "drunk gambler" example.
- Lee & Bareinboim (NeurIPS 2018), *"Structural Causal Bandits: Where to Intervene?"* Uses do-calculus to reduce the intervention set before any standard bandit algorithm is applied.
- Zhang & Bareinboim (IJCAI 2017), *"Transfer Learning in Multi-Armed Bandits: A Causal Approach."* Cross-environment transfer with causal bounds.
- Lu, Meisami, Tewari & Yan (AISTATS 2020), *"Regret Analysis of Bandit Problems with Causal Background Knowledge."* C-UCB and C-TS with logarithmic cumulative regret.

**Why economists care.** Causal bandits are the cleanest formal setting for **adaptive experimentation with structural assumptions** — exactly the world of dynamic pricing, A/B testing, and email-campaign optimisation.

**Relation to chapter.** A small omission in the current chapter, easily fixed in 1–2 paragraphs. The Forney–Pearl–Bareinboim (ICML 2017) counterfactual-data-fusion paper is partially covered, so causal bandits is the natural completion.

---

### Cluster 9 — Causal Imitation Learning with Hidden Confounding

**Canonical paper.** Daniel Kumor, Junzhe Zhang & Elias Bareinboim (NeurIPS 2021; arXiv:2208.06276), *"Sequential Causal Imitation Learning with Unobserved Confounders."*

**Core mathematical idea.** The demonstrator's actions may be confounded with the imitator-unobservable context (e.g. an expert who sees vital signs the imitator's camera does not). Naïve behaviour cloning then fails. The authors give a **graphical criterion** — a sequential generalisation of the backdoor criterion — for when a sensor-mismatched imitator can recover demonstrator-equivalent performance using observational data only, and when interventional data is needed. The conditions are tight (sound and complete).

**Memorable example.** "Monkey see, monkey do" — an expert who watches both the road and the rear-view mirror demonstrates driving, but an imitator with only forward-camera input incorrectly latches onto post-action mirror checks. The paper's "imitability" theorem gives precise conditions under which observational demonstrations suffice.

**Related papers in this cluster.**
- Zhang, Kumor & Bareinboim (NeurIPS 2020), *"Causal Imitation Learning with Unobserved Confounders."* Single-stage originator.
- de Haan, Jayaraman & Levine (NeurIPS 2019), *"Causal Confusion in Imitation Learning."* Empirical demonstration of the failure mode.
- Swamy, Choudhury, Bagnell & Wu (CMU 2022), *"Causal Imitation Learning under Temporally Correlated Noise."* Frames IL as an IV problem.
- van der Laan, Kallus & Bibaut (2025, arXiv:2509.21172), *"Inverse Reinforcement Learning Using Just Classification and a Few Regressions"* — connects IRL/MaxEnt to dynamic discrete choice (Hotz–Miller, Rust) with debiased ML.

**Why economists care.** Inverse reinforcement learning is the cousin of **dynamic discrete choice estimation** (Hotz–Miller, Rust). The recent van der Laan–Kallus–Bibaut (2025) line makes this connection explicit and gives a semiparametrically efficient IRL/DDC estimator using debiased ML. Pedagogically powerful.

**Relation to chapter.** The chapter has nothing on imitation learning. Given how central IRL is in economic structural estimation, this is a real gap.

---

### Cluster 10 — Adaptive Experimentation & Post-Bandit Inference

**Canonical paper.** Vitor Hadad, David A. Hirshberg, Ruohan Zhan, Stefan Wager & Susan Athey (PNAS 2021), *"Confidence Intervals for Policy Evaluation in Adaptive Experiments,"* 118(15). https://doi.org/10.1073/pnas.2014602118 ; arXiv:1911.02768.

**Core mathematical idea.** Once data are collected by a bandit algorithm, IPW-based estimators behave badly: as propensities decay toward zero on sub-optimal arms, the variance explodes and distributions become heavy-tailed. The authors propose **adaptively-weighted AIPW** estimators that scale each per-period contribution by a chosen weight $h_t$ (e.g. $h_t = \sqrt{e_t}$ — "constant allocation rate") so that the resulting estimator is asymptotically normal even when propensities decay. The result gives valid confidence intervals after a Thompson-sampling or UCB trial.

**Memorable example.** Athey, Byambadalai, Hadad, Krishnamurthy, Leung & Williams (2022, arXiv:2211.12004), *"Contextual Bandits in a Survey Experiment on Charitable Giving: Within-Experiment Outcomes versus Policy Learning"* — the goal is "to use a participant's survey responses to determine which charity to expose them to in a donation solicitation," and adaptive-weighting AIPW delivers valid CIs on the value of sub-optimal arms a standard bandit would otherwise under-sample.

**Related papers in this cluster.**
- Maximilian Kasy & Anja Sautmann (Econometrica 2021, 89(1):113–132, doi:10.3982/ECTA17527), *"Adaptive Treatment Assignment in Experiments for Policy Choice."* The complementary design problem: choose assignment probabilities to maximise welfare of post-experiment policy choice. Their "exploration sampling" algorithm is asymptotically optimal for policy choice and beats both Thompson sampling and non-adaptive RCTs.
- Bibaut, Chambaz, Dimakopoulou, Kallus & van der Laan (NeurIPS 2021, arXiv:2106.00418), *"Post-Contextual-Bandit Inference."* TMLE-based debiased estimators after contextual-bandit data collection.
- K. W. Zhang, Janson & Murphy (NeurIPS 2020), *"Inference for Batched Bandits."* Batched-data inference.
- Bibaut (PhD thesis, Berkeley 2021), *"Statistical Methods for Causal Inference from Sequentially Collected Data."* The reference text.

**Why economists care.** Exactly the RCT-design-meets-bandit literature economists are now adopting. A. Stefano Caria, Gordon, Kasy, Quinn, Soha O. Shami & Alexander Teytelboym (*Journal of the European Economic Association* 22(2):781–836, April 2024, doi:10.1093/jeea/jvad067), *"An Adaptive Targeted Field Experiment: Job Search Assistance for Refugees in Jordan,"* tested "a small cash grant, information and psychological support" via a "Tempered Thompson Algorithm" — a textbook applied case of this cluster.

**Relation to chapter.** A new topic, not currently in the chapter. Sits cleanly between the OPE section (offline) and an "online RL" section the survey may well have elsewhere.

---

### Honourable mentions (one-paragraph candidates)

**Distributionally-robust OPE.** Si, Zhang, Zhou & Blanchet (Operations Research 2024, arXiv:2011.04102), *"Reliable Off-Policy Evaluation for Reinforcement Learning."* Confidence-bound OPE via KL/Wasserstein uncertainty sets. Refined by Kallus, Mao, K. Wang & Zhou (ICML 2022, arXiv:2202.09667), *"Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning."*

**Conformal OPE.** Taufiq, Ton, Cornish, Teh & Doucet (NeurIPS 2022, arXiv:2206.04405), *"Conformal Off-Policy Prediction in Contextual Bandits"*; Y. Zhang, Shi & Luo (2023, arXiv:2206.06711), *"Conformal Off-Policy Prediction."* Distribution-free finite-sample prediction intervals for policy value — important for safety-critical deployment.

**Dynamic mediation under RL.** Luo, Shi, J. Wang, Z. Wu & L. Li (Annals of Statistics 2025, 53(1):400–425, arXiv:2310.16203), *"Multivariate Dynamic Mediation Analysis under a Reinforcement Learning Framework."* Markov mediation process + system of time-varying SEMs; decomposes ATE into immediate-direct, immediate-mediation, delayed-direct, and delayed-mediation components.

**Statistical inference for RL value functions.** Shi, S. Zhang, Lu & Song (JRSS-B 2022, arXiv:2001.04515), *"Statistical Inference of the Value Function for Reinforcement Learning in Infinite-Horizon Settings."* Sieve-based confidence intervals for $V^\pi$ — the inferential complement to point-estimation OPE.

## Recommendations

### Top four to add (the 80/20 picks)

**Pick 1 — DML for dynamic treatment effects (Cluster 1).** Spotlight: Lewis & Syrgkanis (NeurIPS 2021). The single highest-value addition. Pedagogically clean (Robinson partial-linear moments, applied recursively), uses the math the audience already knows, has named follow-ups (Chernozhukov–Newey–Singh–Syrgkanis 2023 for automatic debiasing via recursive Riesz representers), and explicitly bridges to OPE in its arXiv v5 abstract. Mention the recursive-Riesz successor and Foster–Syrgkanis "Orthogonal Statistical Learning" (*Ann. Statist.* 2023) as the theoretical backbone.

**Pick 2 — Robins g-methods / MSMs as econometric precursor to OPE (Cluster 4).** Spotlight: Robins, Hernán & Brumback (Epidemiology 2000), with a sub-page on Murphy (JRSS-B 2003) and Schulte–Tsiatis–Laber–Davidian (*Statistical Science* 2014). This is **historical/pedagogical scaffolding**, not a new method — but it gives the chapter the missing translation between the biostatistics literature (where DTRs were invented) and the modern RL/OPE literature. Without this section, an economist reader will keep asking "haven't I seen this before?"

**Pick 3 — Sensitivity analysis & partial identification for sequential OPE (Cluster 2).** Spotlight: Namkoong, Keramati, Yadlowsky & Brunskill (NeurIPS 2020). Complements the chapter's existing IV/proxy material by giving a fallback when no IV or proxy exists — exactly the toolkit economists use to defend observational identification claims. Pair with Kallus & Zhou (NeurIPS 2020) for infinite-horizon and Bruns-Smith & Zhou (NeurIPS 2023) for the orthogonalised robust-FQE link.

**Pick 4 — Policy learning from observational data (Cluster 3).** Spotlight: Athey & Wager (*Econometrica* 2021). Closes the OPE→OPL loop and gives economists a paper they may already know — the easiest "anchor" entry into the RL world. Include Zhou–Athey–Wager (*Operations Research* 2023) for the multi-action / decision-tree extension and a one-paragraph note on Kitagawa–Tetenov.

### Optional fifth pick if space permits

**Counterfactual / hindsight credit assignment (Cluster 6)** — Mesnard et al. (ICML 2021). Strictly extends what the chapter already says about Buesing–Oberst, is model-free (so applies to any deep-RL setup), and the "skill vs. luck" framing lands well with economists who already think in terms of returns decomposition.

### What to deprioritise (and why)

- **Causal world models / state abstraction (Cluster 5)** — important but more relevant to robotics/control than economics applications; cite Pitis–Creager–Garg in a footnote.
- **Causal MARL beyond Jaques et al. (Cluster 7 second half)** — sparse mathematical theory, mostly applied; cite Jaques et al. once and move on.
- **Causal discovery for RL** — interesting but the survey isn't about discovery; one-line reference to Wang–Xiao–Xu–Zhu–Stone CDL suffices.
- **Causal imitation learning (Cluster 9)** — *unless* the chapter wants to bridge to dynamic discrete choice; if so, prioritise van der Laan–Kallus–Bibaut (2025) over Kumor–Zhang–Bareinboim because the economics audience cares about IRL/DDC.

### Benchmarks that would change these recommendations

- If the chapter's audience is more **biostatistics / public-health economics**: promote Cluster 4 (g-methods/DTRs) and Cluster 2 (sensitivity) above Cluster 1.
- If the audience is more **IO / digital economics / pricing**: promote Cluster 10 (adaptive experimentation) and Cluster 8 (causal bandits) above Cluster 2.
- If the chapter has space for ≤ 2 additions: drop Picks 3 and 4 and add only Picks 1 and 2 — DML and g-methods together carry the entire econometric pedagogy.
- If the chapter is updated within 12 months: watch the **automatic debiasing** literature (Chernozhukov–Newey–Singh–Syrgkanis 2023 and successors) and the **debiased IRL/DDC** line (van der Laan–Kallus–Bibaut 2025) — both are likely to mature into the new canonicals.

## Caveats

- **Forward-looking material flagged.** Several "canonical" papers in active clusters are very recent (Chernozhukov–Newey–Singh–Syrgkanis 2023; van der Laan–Kallus–Bibaut 2025; Bruns-Smith–Zhou 2023). They are likely to be revised; cite the arXiv version with a date stamp.
- **The Lewis–Syrgkanis OPE statement is in the arXiv v5 abstract only.** The shorter NeurIPS 2021 proceedings abstract ends at "root-n asymptotic normality of the estimated causal effects." When the chapter quotes "off-policy evaluation of any target dynamic policy at parametric rates," cite the arXiv version (arXiv:2002.07285).
- **Real-data applications are sparse for Lewis & Syrgkanis (2021) itself.** Per targeted verification, the paper's experiments are simulation-based (linear Markovian state-space simulations + recursive Lasso) and the dynamic-pricing / customer-LTV motivation is in the introduction rather than a worked application. The Battocchi et al. (NeurIPS 2021, arXiv:2103.08390) companion paper supplies a real semi-synthetic corporate dataset and is the natural citation if a worked economics example is needed.
- **Some clusters overlap.** CoDA (Cluster 5), CAI (Cluster 7), and CCA (Cluster 6) all use *counterfactual* logic in slightly different RL components. The chapter should make the division clean: CoDA augments data, CAI shapes exploration, CCA reduces gradient variance.
- **The Annals-of-Statistics / JRSS-B "Shi-Song-Lu" line** is highly mathematical but uses notation closer to biostatistics. For an economics audience, prefer the Athey–Wager–Kallus side as primary references and cite the AOS/JRSS-B line as "for further theoretical depth."
- **Causal-discovery-with-RL** (Zhu, Ng & Chen, ICLR 2020) flips Direction 1 and Direction 2 — RL is *used to discover* the causal graph, not the other way around. It belongs in the chapter only if the author wants to make this point explicitly; otherwise it confuses the directional framing.