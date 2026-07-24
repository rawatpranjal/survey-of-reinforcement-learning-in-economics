# ch10b claim-to-source ledger

Date: 2026-07-24

Scope: every citation retained in `tex/rl_for_ci.tex` after the revision. The
full-text files are in the primary checkout's `ch10b_rl_for_ci/papers/`
library, except for Precup, Sutton, and Singh (2000), which is in
`ch13_field_deployments/papers/ope_references/`. Every listed Markdown file
passed the `read` skill's full-text integrity check. The locations below were
then read against the corresponding manuscript claim. A citation was retained
only when it supports a specific sentence, equation, theorem statement, or
empirical description.

## Dynamic treatment regimes and the dictionary

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `murphy2003dtr` | A DTR is a sequence of history-dependent rules; observed-data backward induction identifies an optimal benefit-to-go under no unmeasured confounding. The Fast Track example motivates adaptive treatment intensity. | `murphy2003dtr.md`, pp. 3-8 and the Fast Track discussion; backward induction and finite-horizon Bellman equation in Section 2. | Retained. The chapter's theorem adds explicit standard-Borel, positivity, terminal-kernel, and tie-breaking conditions rather than attributing those details to Murphy. |
| `schulte2014qlearning` | Q-learning is backward recursive outcome regression for DTRs; A-learning estimates treatment contrasts; the notation connects DTR and RL methods. | `schulte2014qlearning.md`, abstract and Sections 2-5, especially the description of Q- and A-learning near the beginning. | Retained. The dictionary footnote states where the analogy is structural rather than literal. |
| `robins2000msm` | Marginal structural models use inverse-probability treatment weighting for longitudinal treatment-confounder feedback. | `robins2000msm.md`, abstract, Sections 2-3, and the stabilized-weight construction. | Retained only in the method dictionary and the MSM benchmark description. |
| `robins2004snmm` | Structural nested mean models parameterize blip contrasts and are estimated by g-estimation. | `robins2004snmm.md`, Sections 3-5 on blip functions, structural nested models, and g-estimation. | Retained. The chapter now defines the blip counterfactually under a fixed continuation regime, not as an observed conditional-mean contrast. |
| `precup2000` | Trajectory and per-decision importance sampling evaluate a policy from off-policy trajectories under support. | `precup2000_eligibility.md`, Sections 3-4 on likelihood ratios, eligibility traces, and per-decision weighting. | Retained for the dictionary and IS identification. The prose says per-decision weighting *usually* reduces variance, not that it always does. |

## Empirical illustrations

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `laber2014dtrchallenges` | The ADHD SMART analysis retains 138 of 155 children, 81 of whom receive a second randomization; it uses the 32-week teacher Impairment Rating Scale outcome. The fitted rules depend on adherence and prior medication exposure. Adaptive 90% intervals do not support a unique first-stage recommendation but do distinguish the stage-2 actions at low adherence, with intervals `[-2.21,-0.57]` and `[-2.51,-0.60]`; near-zero contrasts create nonregular inference. | `laber-2014-dtr-technical-challenges.md`, application at pp. 23-27, especially Tables 6-9, and open problems at pp. 27-29. | Added as the lead DTR application. The prose distinguishes an estimated argmax from the evidence for its prescribed action and separates contrast inference from evaluation of a selected rule. |
| `wangtom2025dtrtutorial` | Adaptive intervals and resampling methods address nonregularity in Q-learning, and the STAR*D tutorial illustrates how ordinary bootstrap SEs can be misleading near a zero contrast. | `wang-tom-2025-optimal-dtr-tutorial.md`, inference discussion and STAR*D application. | Added as a supporting tutorial. Bibliographic metadata was updated to the 2026 journal publication while preserving the citation key. |
| `kaushik2022sepsiscql` | A retrospective sepsis study fit conservative Q-learning to vasopressor and intravenous-fluid decisions. | `kaushik2022sepsiscql.md`, abstract, methods, and policy-action discussion. | Retained with “retrospective study” and no deployment-effect claim. |
| `kondrup2023deepvent` | DeepVent studies offline RL for mechanical-ventilation decisions. | `kondrup2023deepvent.md`, abstract and methods. | Retained with the same offline-only qualification. |
| `roggeveen2024icmrl` | Cross-policy evaluation across reward weightings exposes sensitivity in ICU RL policy evaluation. | `roggeveen2024icmrl.md`, abstract, reward-weight experiments, and discussion. | Retained. “Reward specifications” was narrowed to “reward weightings.” |
| `luo2024dtrbench` | Controlled DTR benchmarks vary noise, missingness, and pharmacokinetic structure and reveal instability. | `luo2024dtrbench.md`, benchmark design and robustness experiments. | Retained as benchmark evidence, not clinical evidence. |

## Off-policy evaluation

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `ueharaShiKallus2022ope` | OPE efficiency depends on the assumed NMDP, time-varying MDP, or stationary MDP model; the review gives the bandit efficient influence function and bound. | `uehara-2022-review-ope.md`, model taxonomy and semiparametric-efficiency sections. | Retained. The chapter separates identification assumptions from structural Markov restrictions. |
| `dudik2014doubly` | In contextual bandits, the DR score is unbiased when either the reward regression or logging-policy model is correct. | `dudik-2014-doubly-robust-policy-evaluation-optimization.md`, Sections 2.1 and 3.3-3.4. | Retained for the $H=1$ boundary case and the bandit DR equation. |
| `jiangli2016doubly` | Sequential DR follows a backward recursion; Theorem 1 decomposes its variance; Theorem 2 gives the tree-MDP lower bound; Observation 1 gives oracle-Q attainment. | `jiang-2016-doubly-robust-ope.md`, Sections 3-5, Theorems 1-2, Observation 1. | Retained with corrected theorem numbering. |
| `farajtabar2018mrdr` | MRDR fits the direct-model component to minimize the variance of the DR estimator. | `farajtabar-2018-more-robust-doubly-robust-ope.md`, abstract and Section 4. | Retained; no stronger finite-sample dominance claim is made. |
| `thomasBrunskill2016magic` | WDR self-normalizes the DR score and MAGIC blends partial-horizon estimates by estimated MSE. | `thomas-2016-data-efficient-ope.md`, WDR and MAGIC sections. | Retained with the finite-sample bias tradeoff explicit. |
| `kalluszhou2018continuous` | Continuous-treatment OPE replaces exact action matching by kernel smoothing, with a bandwidth bias-variance tradeoff. | `kallus-zhou-2018-continuous-treatments.md`, Sections 3-4. | Retained in one scope footnote. |
| `xie2019marginalized` | Cumulative weights can be approximately log-normal; MIS uses marginal state ratios; Theorem 4.1 gives polynomial-in-horizon MSE up to a factor of the lower bound. | `xie-2019-marginalized-is-ope.md`, introduction, log-weight example, estimator construction, Theorem 4.1. | Retained. The general NMDP impossibility is now attributed to Double RL rather than to this estimator-specific result. |
| `nachum2019dualdice` | DualDICE estimates a stationary density ratio through a convex saddle-point formulation without knowing the behavior policy. | `nachum2019dualdice.md`, abstract and Sections 3-4. | Retained for the stationary infinite-horizon case only. |
| `kallusUehara2022doubleRL` | MDP and NMDP efficiency bounds differ, generally polynomial versus exponential in horizon; cross-fitted Double RL is efficient under a product-rate nuisance condition. | `kallus2020doublerl.md`, Theorems 1-2, Remark 6, and Theorems 5 and 12. | Retained. The nuisance statement now uses the exact product rate rather than an unconditional fourth-root shorthand. |
| `thomas2015hcope` | HCOPE forms lower confidence bounds by collapsing the upper tail of importance-weighted returns and applying a modified empirical Bernstein inequality. | `thomas-2015-high-confidence-ope.md`, Theorem 1 and the discussion around the winsorization construction. | Retained with “winsorize” and “modified empirical Bernstein bound,” replacing the vague phrase “truncated weights.” |
| `sakhi2024logsmoothing` | Logarithmic smoothing supplies concentration bounds for OPE and supports policy selection and learning. | `sakhi-2024-logarithmic-smoothing-pessimistic-ope.md`, Sections 3.3 and 4. | Retained without claiming it is universally tighter. |
| `saito2021robustope` | OPE conclusions can change with estimator hyperparameters and evaluation policies. | `saito-2021-evaluating-robustness-ope.md`, introduction and benchmark protocol. | Retained as a reason to report stability rather than a global estimator ranking. |
| `udagawa2023pas` | Policy-adaptive estimator selection chooses different estimators for different target policies. | `udagawa-2023-policy-adaptive-estimator-selection-ope.md`, Sections 3-4. | Retained alongside, but not conflated with, hyperparameter robustness. |
| `hao2021bootstrapfqe` | Linear FQE admits a trajectory-level bootstrap under policy completeness; entire independent episodes, not individual transitions, are resampled. | `hao2021bootstrapfqe.md`, method and distributional-consistency result. | Added to support a direct-method interval with its model restrictions stated. |
| `liaoMurphy2021longterm` | The HeartSteps application analyzes 37 of 44 participants from a 42-day micro-randomized trial with five daily decisions and treatment probability 0.6 when available. It evaluates three long-run policies with stationary density ratios. The location policy has value 3.155 and interval `[2.893,3.417]`; neither reported policy contrast excludes zero. The paper back-transforms its location-versus-no-suggestion point difference to about 55 steps, or 22% of the mean post-decision count. | `liao-2021-long-term-ope-mobile-health.md`, Section 7, pp. 21-23. | Added as the lead genuinely sequential OPE application. The back-transformation is identified as a point-estimate interpretation, and the nonsignificant contrasts are reported rather than promoting the highest policy rank. |

## Dynamic DML and recursive orthogonalization

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `lewisSyrgkanis2021dynamicDML` | Cross-fitted orthogonal residual moments estimate low-dimensional dynamic effects; Theorem 4 gives product-rate asymptotic normality; Corollary 9 gives fixed-policy value inference. | `lewis2021dml.md`, partially linear model, Algorithm 3, Theorem 4, and Corollary 9. | Retained. The chapter now uses unconditional moments, a joint upper-triangular covariance, and distinguishes fixed-target from target-independent blips. |
| `chernozhukov2023automatic` | Recursive Riesz representers debias nested mean regressions and yield an exact mixed-bias product remainder. | `chernozhukov2023automatic.md`, recursive representer definition, Riesz loss, debiased score, and mixed-bias theorem. | Retained. The terminal convention and coefficient index in the displayed score were corrected. |
| `fosterSyrgkanis2023orthogonal` | Orthogonality makes nuisance error enter excess risk at second order in two-stage learning. | `foster2023orthogonal.md`, main oracle inequality and strong-convexity discussion. | Retained in a scope footnote. The chapter no longer attributes a universal MSE rate to the paper. |
| `jaman2025penalizedg` | The hemodiafiltration study contains 474 patients and 170,761 sessions; its repeated-outcome analysis uses the first six sessions. Penalized g-estimation selects effect modifiers and reports sandwich standard errors. Under the AR1 working correlation, the CHUM effect is -1.85 litres (SE 0.31) and its cancer interaction is 3.89 litres (SE 0.78), but ordinary selected-model intervals ignore selection uncertainty. | `jaman-2025-penalized-g-estimation-repeated-outcomes.md`, Section 4, pp. 14-16, especially Table 5 and the post-selection warning; Section 2.4 for sandwich inference. | Added as the closest empirical SNMM bridge, explicitly not as an application of the exact Lewis-Syrgkanis algorithm or as a randomized facility comparison. |
| `jaman2025postselectiong` | Naive selected-model intervals can undercover; uniformly valid and decorrelated-score methods provide post-selection confidence intervals for the repeated-session g-estimation setting. | `jaman-2025-valid-post-selection-g-estimation.md`, Sections 2.2-2.4 and 4. | Added to make the difference between nuisance orthogonality, joint sandwich inference, and model-selection uncertainty explicit. |

## Offline policy learning

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `kitagawa2018who` | Empirical welfare maximization with known propensities has matching finite-sample upper and minimax lower regret rates governed by policy-class complexity. | `kitagawa2018who.md`, main regret theorems and JTPA application. | Retained as the $T=1$ precursor. |
| `athey2021policy` | Cross-fitted doubly robust policy scores support $n^{-1/2}$ best-in-class regret under product-rate nuisances and can accommodate IV identification. | `athey2021policy.md`, score in Equation 14 and main regret theorem. | Retained. The statement is asymptotic and best-in-class. |
| `zhou2023offline` | Multi-action offline policy learning admits exact optimization for finite-depth decision-tree classes. | `zhou2023offline.md`, multi-action setup and exact tree-search algorithm. | Retained as a one-sentence extension. |
| `sakaguchi2024dynamicpolicy` | Algorithm 1 performs cross-fitted AIPW backward induction; Theorem 4.3 gives the dynamic regret rate under overlap, entropy, product-rate, and later-stage class-correctness conditions; Theorem 5.1 gives joint optimization. Section 7 analyzes 1,877 Project STAR students, reports fitted tree splits at 19 years of teacher experience and a test-score threshold of 914, and gives welfare contrasts of 8.16% and 1.27%, with about 23% assigned to an aide class in at least one grade. | `sakaguchi2024dynamicpolicy.md`, Algorithm 1, Equations 4-5, Theorems 4.3 and 5.1, Section 7, Figure 1, and Table 3. | Retained. The simulation now fits each stage-1 continuation using a stage-2 model trained outside the same scoring fold. Project STAR numbers are stated as cross-validated observational value contrasts without reported standard errors or deployment evidence. |

## Perspective and open issues

| Citation | Manuscript claim | Full-text location checked | Disposition |
|---|---|---|---|
| `bannon2020causality` | Causal modeling and batch RL are complementary: the former clarifies counterfactual identification and the latter supplies planning and approximation machinery. | `bannon-2020-causality-batch-rl.md`, abstract, introduction, and comparison sections. | Added only to frame the open-issues synthesis; no theorem is attributed to it. |

## Exclusions made during this audit

- `robins1986` remains excluded because no theorem or historical priority claim
  in the chapter depends on it. A full-text OCR of the scanned article is now
  present in the source library. Its prose is usable, but its mathematical
  notation must be checked against the rendered scan before quotation.
- `liu2018curse` remains excluded because its stationary-distribution estimator
  is not needed for the finite-horizon claim retained in the chapter. A
  matching full-text conversion is now present in the source library. The
  estimator-specific statement is supported by Xie et al.; the model-class
  impossibility is supported by Kallus and Uehara.
- General RL textbooks, the original Watkins paper, a benefit-to-go naming
  aside, and several neighboring method citations were removed because they
  did not carry a chapter-specific claim.
