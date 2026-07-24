# ch10b revision verification

Date: 2026-07-24

## Result

The revised chapter passes the source, proof, simulation, artifact, visual,
reference, and build gates. The independent proof review rejected the first
theorem draft because it exchanged maximization and expectation, omitted
regularity for continuous histories, and underspecified the induced decision
process. The shipped proof uses fixed-policy evaluation followed by a dominance
induction, assumes standard Borel state spaces and finite ordered actions, and
defines both the transition and terminal reward kernels.

The claim-source ledger and paper coverage index cover all 36 citation keys
retained in the chapter.
Every cited paper was checked from full text, and the ledger records the exact
claim, source location, and disposition.

The DTR dictionary now contains only identities implied by the formal
full-history MDP construction. It maps histories, actions, regimes, rewards,
assignment laws, evaluation and optimality recursions, optimal rules, and
regime values. A separate paragraph states why g-computation and fitted-Q
evaluation, IPTW and importance sampling, g-estimation and orthogonal OPE,
blips and RL advantages, sequential exchangeability and a known logger, and
positivity and coverage are related but not generally equivalent. The
distinctions were checked directly against Schulte et al. (2014), Robins et
al. (2000), Precup et al. (2000), Robins (2004), and Lewis and Syrgkanis
(2021).

The follow-up pass adds four potential-outcome assumptions, plain-language
data-generating processes before every simulation, estimator-level standard
errors, lead empirical applications, a literature-role index, and a dedicated
open-issues section. A final prose pass removes unnecessary connective
language, keeps exposition in the present, and makes the role of each cited
paper explicit. Four missing or stale bibliography records were repaired.
Seven targeted sources passed both the source-integrity and full-text gates,
including the sequential HeartSteps OPE application and the repeated-session
SNMM application. Full text is also present for the two excluded Robins (1986)
and Liu et al. (2018) citations, with the OCR limitation on Robins's
mathematical notation recorded in the source header and claim ledger.

The Dynamic Treatment Regimes section now proceeds in linear order from the
Fast Track population and trial arms to the semester-level home-visiting
sequence, the limits of a one-time A/B comparison, and the abstract DTR
notation. The formal potential-outcome section maps the histories, actions,
potential states, potential terminal outcomes, regimes, and identification
assumptions back to the home-visiting example. It keeps the terminal outcome
generic because Murphy's formal setup does not select one Fast Track endpoint.
It now states that earlier visits can change later family status, later
assignments, and the terminal outcome. It also explains why an unadjusted
regression of outcome on total visits inherits the same selection problem as a
many-versus-few comparison. These details were checked against Murphy's full
text and added to the claim-source ledger.

The theorem is followed by a four-paragraph STAR*D example that maps the
depression scores, treatment choices, potential states, identification
assumptions, and backward regressions to the formal objects. It distinguishes
randomization among specific treatments from the observational
switch-versus-augmentation choice and records the extra complete-case
condition. The ADHD SMART remains the application for nonregular inference.
The other substantive empirical applications retain three evidence-bearing
paragraphs each. Every added number has a full-text location in the
claim-source ledger. Text extracted from the standalone PDF contains no
internal script name, simulation path, local filesystem path, or source-note
reference.

All four numerical simulations use one stylized Fast Track-inspired
home-visiting context. Each writeup states that the design is a methodological
analogue rather than a reconstruction of the trial. Each simulation appears at
the end of its section under a numbered `Simulation Study:` heading, with two
prose paragraphs, one consolidated table, and one figure.

Each major technical section now closes with one or two plain sentences naming
the object recovered in that section and the uncertainty or evaluation that
accompanies it. Open Issues closes with the three chapter-level outputs, a
fixed-policy value, dynamic effects with standard errors, and a learned regime
whose value is evaluated separately.

## Simulation audit

### DTR recursion and Q-learning

1. The tabular plug-in estimator is sequential g-computation, tabular
   Q-learning uses replay updates, neural FQI uses separate stage regressions,
   and DQN uses a frozen stage-2 target for the stage-1 update.
2. The two neural methods use paired cohorts, a common two-layer 64-unit MLP,
   the same smooth sign-changing contrast, and separate evaluation draws.
3. FQI runs 1,000 full-batch epochs at every cohort size. DQN uses 50 data
   passes for every cohort size, with a hard
   target update every five passes. A 200-pass diagnostic failed the unchanged
   recovery gate because constant-learning-rate training deteriorated after the
   useful solution. The 50-pass diagnostic used five seeds; the remaining 15
   held-out seeds reach 0.9726 of oracle value at \(N=20{,}000\).
4. The continuous oracle uses nested Gauss-Hermite quadrature and agrees with
   an independent Monte Carlo calculation within 1.20 standard errors.
5. At the right edge, plug-in g-computation and tabular Q-learning reach 1.0000
   and 0.9968 of the tabular oracle. Neural FQI and DQN reach 0.9842 and 0.9740
   of the continuous-state oracle.
6. The unadjusted regression of outcome on total visit count has a coefficient
   of -0.2870 with Monte Carlo standard error 0.0023 at \(N=10{,}000\). It
   selects the never-visit schedule in all 50 replications and reaches 0.8124
   of oracle value. The always-visit schedule reaches 0.9422. This benchmark
   uses the same paired cohorts as the other tabular estimators.

Bullshit score: 12%. The algorithms, paired data, independent oracle, held-out
diagnostic seeds, and recovery gates are explicit. The residual risk is
dependence on the chosen shared architecture and optimizer schedules.

### Off-policy evaluation

1. DM, IS, PDIS, WIS, DR, WDR, and MIS score the same logged datasets. DR and
   WDR use out-of-fold value predictions, and WDR normalizes cumulative weights
   globally at each time step.
2. Exact dynamic programming supplies the policy value. An independent
   on-policy Monte Carlo calculation agrees within 1.17 standard errors.
3. IS and PDIS pass unbiasedness gates at every sample size. DR passes all
   three correct-or-one-nuisance-correct cells, while the both-wrong cell has a
   nonzero bias by construction.
4. At \(n=2000\) and \(H=16\), DM and IS have RMSE 0.111 and 1.555. At \(H=64\),
   IS and MIS have relative RMSE 2.847 and 0.134, a 21.3-fold contrast.
5. IS, PDIS, and cross-fitted DR now carry trajectory-score analytic standard
   errors. At \(H=8,n=1000\) over 1,000 repeated datasets, their mean analytic
   SE divided by empirical SD is 0.976, 0.992, and 1.014; 95 percent coverage
   is 0.943, 0.953, and 0.945.

Bullshit score: 10%. The exact oracle, theory-directed invariants, and
misspecification cells directly test the claims. The tabular model remains
deliberately favorable to DM.

### Dynamic DML

1. The estimator implements the upper-triangular Lewis-Syrgkanis moments with
   five-fold cross-fitted outcome and treatment nuisances.
2. Its standard errors use the full joint upper-triangular sandwich, so
   uncertainty in the second-stage blip propagates into the first-stage
   standard error.
3. At \(n=4000\), mean formula standard errors divided by Monte Carlo standard
   deviations are 1.065 and 1.005. Coverage is 0.97 and 0.93, with balanced
   left and right tail misses.
4. The dynamic-DML biases are -0.0047 and -0.0013. Naive OLS retains a 0.9326
   second-stage bias and zero coverage; the IPTW-fitted MSM has bias 0.1456 and
   coverage 0.73.
5. The joint covariance check has relative Frobenius error 0.134. For the
   contrast \(\psi_1-\psi_2\), mean analytic SE is 0.0537 against Monte Carlo
   SD 0.0506, with 0.950 coverage and 0.025 misses in each tail.
6. The IPTW-MSM interval is labeled as a naive fixed-weight sandwich because
   it does not propagate propensity estimation or trimming.

Bullshit score: 10%. Recovery, standard-error calibration, coverage, and tail
symmetry are hard gates. The sparse linear DGP is intentionally compatible with
the target structural model.

### Backward-induction policy learning

1. Backward AIPW, plug-in Q, and IPW learn the same two-stage threshold class by
   exact sorted-prefix optimization.
2. Stage-1 fitted-Q targets are constructed separately inside each outer fold.
   No stage-2 model trained on a scoring fold supplies that fold's stage-1
   training target.
3. The oracle thresholds are \(c_1^*=0.51558269\) and \(c_2^*=0.4\). Local
   perturbations reduce value, the exact threshold search matches exhaustive
   prefix enumeration, and quadrature agrees with one million Monte Carlo draws
   within 0.75 standard errors.
4. At \(n=4000\), plug-in Q, backward AIPW, and IPW reach 0.9996, 0.9966, and
   0.9911 of oracle value. AIPW regret remains between 0.0127 and 0.0203 in the
   two one-sided misspecification cells. No double-robustness claim is made for
   the both-wrong cell.

Bullshit score: 12%. The recursive cross-fitting, oracle, search, and
misspecification behavior are directly checked. The threshold class and correct
outcome basis are deliberately favorable to plug-in Q.

### Sequential-treatment diagrams

The figure is illustrative rather than empirical. Potential outcomes are
dashed, realized variables are solid, the consistency links have no causal
arrowheads, treatment-confounder feedback is highlighted, and the decision
process uses the full observed history as its state.

Bullshit score: 5%. The remaining risk is only the compression inherent in a
schematic.

## Artifact and build gates

- All five scripts compile with `python -m py_compile`.
- All 12 generated figures and result tables round-trip byte-identically
  through `--plots-only`.
- The standalone chapter is 39 pages. It has no undefined citations, no
  internal undefined references, and no overfull boxes. Its unresolved
  references point only to chapters omitted by the standalone driver.
- The open-issues table produces a float-size warning. Visual inspection
  confirms that the full table remains inside the physical page and is not
  clipped.
- The full book is 309 pages with no undefined citations or references or
  LaTeX errors. No overfull boxes originate in ch10b; the full-book log retains
  preexisting overfull warnings from other chapters.
- The chapter PDF was inspected page by page at the diagram, theorem, four
  empirical applications, four simulation blocks, both new inference tables,
  the policy-inference caveat, and the open-issues table. Labels are legible
  and no result is detached from its section.
- The arXiv package compiles independently in its regenerated submission
  directory to 309 pages. The final tarball is 18 MB with 217 archive entries.
  Its manifest
  includes every ch10b figure and table and the previously omitted full-book
  dependencies exposed by this build.
- Every numerical sentence in the four simulation writeups matches the frozen
  result tables or stdout artifacts.
- An independent read-only audit found and triggered fixes for inconsistent
  potential-outcome notation, incomplete NMDP histories, overbroad DR
  unbiasedness language, high-confidence-bound conditions, horizon arithmetic,
  and the target of learned-policy inference. The second audit found no
  remaining theorem, application, bibliography, or inference blocker.
- Deleted labels `subsec:simstudy`, `subsec:rl_for_ci_discussion`, and
  `subsec:murphy_watkins` have no remaining references.
- The final prose contains no em dashes, en dashes, `\textbf`, unlocated direct
  quotations, first-person phrasing, vague temporal modifiers, prose colons,
  `i.e.`, `e.g.`, or `and/or`.
