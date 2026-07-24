# Chapter 10b paper coverage index

Date: 2026-07-24

Scope: papers cited in `tex/rl_for_ci.tex` after the inference, application,
and open-issues revision. “Lead” means the chapter develops the paper’s result
or application in enough detail to reproduce the estimand, assumptions, and
uncertainty calculation. “Foundational” supplies the identification or
estimation backbone. “Supporting” supplies a specific extension. “Narrative”
is empirical context and is not treated as evidence of deployment effects.
Every row has a full-text artifact; claim-level locations are recorded in
`claim_source_ledger_2026-07-24.md`.

| Citation key | Role | Chapter use and fairness note | Full-text artifact | Status |
|---|---|---|---|---|
| `murphy2003dtr` | Foundational | Defines DTR backward induction and motivates it with Fast Track. The chapter uses the trial's intervention-versus-control design, semester-level family-functioning assessment, adaptive home-visiting rule, and overtreatment concerns to explain why the policy target differs from a fixed-protocol contrast. It also maps Murphy's observed sequence and potential variables to the running example without selecting a terminal endpoint that the paper leaves generic. The chapter states its own formal regularity conditions. | `murphy2003dtr.md` | Full text checked |
| `schulte2014qlearning` | Foundational bridge | Connects causal DTR notation to Q-learning and A-learning. Used for the translation, not a priority claim. | `schulte2014qlearning.md` | Full text checked |
| `robins2000msm` | Foundational | Supports longitudinal IPTW/MSMs under treatment-confounder feedback. Used as a benchmark, not the chapter’s lead estimator. | `robins2000msm.md` | Full text checked |
| `robins2004snmm` | Foundational | Supplies counterfactual blip functions and g-estimation for SNMMs. | `robins2004snmm.md` | Full text checked |
| `laber2014dtrchallenges` | Lead application and inference | Three-paragraph ADHD SMART application covers the sample and treatment sequence, fitted rules and adaptive intervals, and the nonregular inference problem. | `laber-2014-dtr-technical-challenges.md` | Newly sourced; full text checked |
| `wangtom2025dtrtutorial` | Supporting synthesis | Supports nonregular DTR inference options and later-stage practical interpretation. Bibliographic key retained although publication year is now 2026. | `wang-tom-2025-optimal-dtr-tutorial.md` | Newly refreshed; full text checked |
| `kaushik2022sepsiscql` | Narrative | Retrospective sepsis CQL study. Mentioned as offline analysis, not clinical benefit evidence. | `kaushik2022sepsiscql.md` | Full text checked |
| `kondrup2023deepvent` | Narrative | Retrospective ventilation policy study. No deployment claim. | `kondrup2023deepvent.md` | Full text checked |
| `roggeveen2024icmrl` | Cautionary support | Shows sensitivity of ICU policy evaluation to reward weighting. | `roggeveen2024icmrl.md` | Full text checked |
| `luo2024dtrbench` | Cautionary support | Controlled benchmark for noise, missingness, and pharmacokinetic structure. | `luo2024dtrbench.md` | Full text checked |
| `precup2000` | Foundational | Supplies trajectory and per-decision importance sampling under support. | `precup2000_eligibility.md` | Full text checked |
| `ueharaShiKallus2022ope` | Lead synthesis | Organizes OPE into bandit, NMDP, time-varying MDP, and stationary MDP classes and explains their efficiency consequences. | `uehara-2022-review-ope.md` | Full text checked |
| `dudik2014doubly` | Foundational boundary case | Supplies the contextual-bandit doubly robust score. | `dudik-2014-doubly-robust-policy-evaluation-optimization.md` | Full text checked |
| `jiangli2016doubly` | Lead method | Sequential DR recursion, variance decomposition, and tree-MDP lower bound. | `jiang-2016-doubly-robust-ope.md` | Full text checked |
| `farajtabar2018mrdr` | Supporting | Trains the value model to reduce DR estimator variance. No universal dominance claim. | `farajtabar-2018-more-robust-doubly-robust-ope.md` | Full text checked |
| `thomasBrunskill2016magic` | Supporting | WDR self-normalization and MAGIC’s estimated-MSE mixture. | `thomas-2016-data-efficient-ope.md` | Full text checked |
| `kalluszhou2018continuous` | Special case | Kernel-smoothed OPE for continuous treatments. One scoped footnote is sufficient for routine coverage. | `kallus-zhou-2018-continuous-treatments.md` | Full text checked |
| `xie2019marginalized` | Lead method | Marginal state ratios and polynomial horizon dependence in the tabular MDP setting. | `xie-2019-marginalized-is-ope.md` | Full text checked |
| `nachum2019dualdice` | Supporting | Stationary density-ratio estimation without the behavior policy. | `nachum2019dualdice.md` | Full text checked |
| `kallusUehara2022doubleRL` | Lead theory | Separates NMDP and MDP efficiency bounds and gives product-rate efficient Double RL. | `kallus2020doublerl.md` | Full text checked |
| `hao2021bootstrapfqe` | Supporting inference | Trajectory bootstrap for linear FQE under policy completeness. The scope restrictions are explicit. | `hao2021bootstrapfqe.md` | Newly sourced; full text checked |
| `thomas2015hcope` | Supporting safety | One-sided lower bounds for importance-weighted returns. Not presented as an ordinary Wald standard error. | `thomas-2015-high-confidence-ope.md` | Full text checked |
| `sakhi2024logsmoothing` | Supporting safety | Log-smoothed pessimistic OPE and selection bounds. | `sakhi-2024-logarithmic-smoothing-pessimistic-ope.md` | Full text checked |
| `saito2021robustope` | Supporting practice | Estimator and hyperparameter robustness across target policies. | `saito-2021-evaluating-robustness-ope.md` | Full text checked |
| `udagawa2023pas` | Supporting practice | Policy-adaptive estimator selection. Kept distinct from hyperparameter robustness. | `udagawa-2023-policy-adaptive-estimator-selection-ope.md` | Full text checked |
| `liaoMurphy2021longterm` | Lead application | Three-paragraph HeartSteps application covers the micro-randomized design, stationary density-ratio estimator, policy values, contrasts, and cautious step-count interpretation. | `liao-2021-long-term-ope-mobile-health.md` | Newly sourced; full text checked |
| `lewisSyrgkanis2021dynamicDML` | Lead theory | Dynamic DML moments, joint sandwich covariance, product-rate normality, and fixed-policy value bridge. It has simulations but no real-data application. | `lewis2021dml.md` | Full text checked |
| `chernozhukov2023automatic` | Lead supporting theory | Recursive Riesz representation for nested means, including the operational influence-score standard error. | `chernozhukov2023automatic.md` | Full text checked |
| `fosterSyrgkanis2023orthogonal` | Supporting theory | General second-order nuisance-error principle. Kept in a scope footnote. | `foster2023orthogonal.md` | Full text checked |
| `jaman2025penalizedg` | Empirical bridge | Three-paragraph repeated-session SNMM application reports the cohort, selected blip estimates and sandwich SEs, and the post-selection limitation. It is not described as the Lewis-Syrgkanis estimator. | `jaman-2025-penalized-g-estimation-repeated-outcomes.md` | Newly sourced; full text checked |
| `jaman2025postselectiong` | Inference bridge | Corrects the selected-model uncertainty left unresolved by ordinary post-selection sandwich intervals. | `jaman-2025-valid-post-selection-g-estimation.md` | Newly sourced; full text checked |
| `kitagawa2018who` | Foundational precursor | Static empirical welfare maximization and minimax regret. | `kitagawa2018who.md` | Full text checked |
| `athey2021policy` | Foundational precursor | Cross-fitted doubly robust policy scores and best-in-class regret. | `athey2021policy.md` | Full text checked |
| `zhou2023offline` | Supporting | Multi-action exact search over finite-depth decision trees. | `zhou2023offline.md` | Full text checked |
| `sakaguchi2024dynamicpolicy` | Lead theory and application | Dynamic AIPW backward induction, regret theorem, and a three-paragraph Project STAR application covering design, fitted trees, value contrasts, and missing empirical SEs. | `sakaguchi2024dynamicpolicy.md` | Full text checked |
| `bannon2020causality` | Perspective | Explains why causal identification and batch RL planning are complementary. Used to frame open issues, not as a technical theorem source. | `bannon-2020-causality-batch-rl.md` | Newly sourced; full text checked |

## Coverage decisions

- The chapter goes deepest on the DTR identification theorem, the OPE estimator
  hierarchy, dynamic DML, and dynamic policy learning.
- The ADHD SMART, HeartSteps, repeated-session dialysis study, and Project STAR
  are the four substantive applications. Each is tied directly to the theory
  immediately preceding it.
- ICU studies remain short narrative examples because they are retrospective
  and do not establish deployment effects.
- Continuous actions, high-confidence bounds, stationary density-ratio methods,
  and post-selection inference are supporting or special-case material that a
  practitioner should know exists but need not derive for routine work.
- Papers on unmeasured confounding, instrumental variables, proxies, and
  partial identification are routed to Chapter 10 on causal RL. Adding them
  here would blur the maintained sequential-exchangeability model rather than
  fill a citation gap.
