# ch10b revision verification

Date: 2026-07-24

## Result

The revised chapter passes the simulation, artifact, reference, and build gates. The independent proof review rejected the first theorem draft because it exchanged maximization and expectation, omitted regularity conditions for continuous histories, and underspecified the induced MDP. The shipped theorem instead uses fixed-policy evaluation followed by a dominance induction, assumes standard Borel state spaces and finite ordered actions, and includes the terminal reward kernel and a measurable tie-breaking rule.

## Simulation audit

### DTR recursion and Q-learning

1. Algorithm identity: the tabular plug-in estimator is sequential g-computation, tabular Q-learning uses replay updates, neural FQI uses separate stage regressions, and DQN uses a frozen target network updated every 500 steps.
2. Environment fidelity: both estimators use the same two-stage data-generating process and the same policy-value functional. The continuous treatment contrast is smooth with temperature 0.25 and changes sign at 0.3466.
3. Data integrity: all methods use paired cohorts and unchanged seed schemes. Cache hashes include the smoothness, training-budget, architecture, and target-update settings.
4. Fairness: FQI and DQN share the same two-layer 64-unit MLP class and cohorts. They are intentionally not equated by gradient count, which the chapter reports.
5. Theory sanity: the continuous oracle is computed by nested Gauss-Hermite quadrature and agrees with an independent Monte Carlo check within 1.20 standard errors. Tabular methods approach 1.0 of oracle value.
6. Leakage: training cohorts, evaluation Monte Carlo draws, and the analytical oracle use separate random streams.
7. Reproducibility: `--plots-only` hits all caches and leaves the generated result table byte-identical.

Headline check: at \(N=20{,}000\), neural FQI reaches 0.9842 of oracle value and DQN reaches 0.9687. The remaining gap is reported rather than hidden. A reserve run with a wider network and a smoother temperature of 0.5 performed worse, so it was not adopted.

Bullshit score: 15%. The result is load-bearing and reproducible. The residual risk is that the neural comparison depends on a chosen architecture and optimizer budget.

### Backward-induction policy learning

1. Algorithm identity: backward AIPW, plug-in Q, and IPW learn the same two-stage threshold class by exact sorted-prefix optimization.
2. Environment fidelity: the observational DGP has logistic propensities, treatment-confounder feedback, and sign-changing treatment gains.
3. Data integrity: all three estimators share cohorts, folds, threshold search, and evaluation.
4. Fairness: nuisance specifications differ only in the planned misspecification cells. The correct stage-1 outcome basis includes truncated-normal continuation features.
5. Theory sanity: the oracle thresholds are \(c_1^*=0.51558269\) and \(c_2^*=0.4\). The policy-value integral is split at the discontinuous stage-1 threshold. Local perturbations reduce value, and the oracle agrees with one million Monte Carlo draws within 0.75 standard errors.
6. Leakage: nuisances are two-fold cross-fitted, and the oracle is computed from the known DGP rather than fitted cohorts.
7. Reproducibility: the exact-threshold optimizer was checked against brute force, Python compilation succeeds, and `--plots-only` leaves the result table byte-identical.

Headline check: at \(n=4000\), plug-in Q, backward AIPW, and IPW reach 0.9996, 0.9963, and 0.9911 of oracle value. Under one-sided misspecification, AIPW regret remains between 0.0126 and 0.0201. Plug-in Q fails when its outcome model is wrong, and IPW fails when its propensity is wrong.

Bullshit score: 15%. The double-robust comparison survives direct perturbation and integration checks. The main limitation is that the threshold class and correct outcome basis are deliberately favorable to plug-in Q.

### Sequential-treatment diagrams

The figure is illustrative rather than empirical. Potential outcomes are dashed, observed variables are solid, treatment-confounder feedback is highlighted, and the MDP panel preserves the full-history state. The potential-outcome links are selection links without causal arrowheads.

Bullshit score: 5%. The only residual risk is interpretive compression in a schematic.

## Artifact and build gates

- All three scripts compile with `python -m py_compile`.
- Both generated result tables round-trip byte-identically through `--plots-only`.
- The policy oracle local optimum, exact threshold optimizer, regret direction, and smooth-contrast cutoff pass direct invariant checks.
- The visualization lint passes for both statistical figures. Its axis-label failure for the DAG is inapplicable because the artifact is a diagram without axes.
- The standalone chapter builds to 26 pages with no undefined citations, no internal undefined references, and no overfull boxes. Its three unresolved references are intentionally external chapter links.
- The full book builds to 300 pages with no undefined citations or references. Existing overfull boxes elsewhere in the book remain outside this change.
- Deleted labels `subsec:simstudy`, `subsec:rl_for_ci_discussion`, and `subsec:murphy_watkins` have no remaining references.
- The final prose contains no em dashes, en dashes, `\textbf`, unlocated direct quotations, colon splices, `i.e.`, `e.g.`, or `and/or`.
