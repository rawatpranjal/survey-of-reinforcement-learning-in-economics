# Proofread: CI_RL.md
Date: 2026-05-16
Source: /Users/pranjal/Code/rl/CI_RL.md

## 1. Summary

| Check | Status | Issue count |
|-------|--------|-------------|
| Message | PASS | 0 |
| Claims | PASS | 0 unsupported (10 spot-checked, all verified) |
| Notation | WARN | 6 undefined-on-first-use symbols |

**Overall status:** WARN

Rationale: the document is a literature-survey *map* (not a paper with derivations), so notation discipline is looser than a tutorial would warrant. No fatal failures. Six math symbols are introduced mid-prose without prior definition; flagged as WARN rather than FAIL because the surrounding citations let a reader recover the definitions externally.

---

## 2. Message Check

**Thesis (one sentence).** The chapter's existing causal-RL coverage is dominated by the Pearl–Bareinboim pillar (confounded MDPs, IV/proxy OPE, counterfactual SCM policies, transportability), and the highest-value additions for an economist audience are four econometrics-side clusters — DML for dynamic treatment effects, Robins g-methods/MSMs, sensitivity analysis for OPE, and policy learning from observational data — organised into a 10-cluster topic map with explicit 80/20 picks.

**Intro verdict: PASS.** The TL;DR (lines 4–6) states the gap explicitly and previews the single most important addition: *"The single most important addition for an economist audience is Lewis & Syrgkanis (NeurIPS 2021), 'Double/Debiased Machine Learning for Dynamic Treatment Effects'... It is the canonical DML×RL paper."* The Key Findings list (lines 11–20) enumerates the 10 clusters that the body then expands. Reader knows exactly what the document will deliver.

**Conclusion verdict: PASS.** The Recommendations section (lines 233–259) returns to the thesis and answers it with named picks: *"Pick 1 — DML for dynamic treatment effects (Cluster 1)... Pick 2 — Robins g-methods / MSMs as econometric precursor to OPE (Cluster 4)... Pick 3 — Sensitivity analysis & partial identification for sequential OPE (Cluster 2)... Pick 4 — Policy learning from observational data (Cluster 3)."* These four picks are the four clusters previewed in line 6 of the TL;DR. The "Benchmarks that would change these recommendations" sub-section (line 254 onward) hardens the answer by spelling out audience-dependent reorderings.

**Off-topic sections:** none. Every cluster section has an explicit "Relation to chapter" subsection that ties back to the survey-gap thesis. The "Honourable mentions" and "Caveats" sections both serve the central recommendation framework rather than wandering.

---

## 3. Claims Check

10 of the most provocative or load-bearing factual assertions were spot-checked (the document contains roughly 60 citations total; sampling was weighted toward (a) precise publication-metadata claims, (b) the central recommendation, and (c) self-flagged "forward-looking" or "active-cluster" claims). All 10 verified.

| Section | Claim (verbatim, ≤20 words) | Cited? | Verifiable? | Source / note |
|---------|-----------------------------|--------|-------------|---------------|
| TL;DR (line 5) | Lewis & Syrgkanis (NeurIPS 2021), "Double/Debiased Machine Learning for Dynamic Treatment Effects" | yes | yes (searched) | Confirmed: NeurIPS 2021, arXiv:2002.07285, sequential-residualisation / g-estimation; matches abstract. |
| Cluster 1 (line 30) | Sequential residualisation: Neyman-orthogonal moment, peel off, recurse | yes | yes (searched) | Confirmed verbatim against arXiv PDF abstract: "sequential regression peeling process... Neyman orthogonal moment estimator... root-n asymptotic normality." |
| Cluster 1 (line 37) | Foster & Syrgkanis, Annals of Statistics 2023, 51(3):879–908, doi:10.1214/23-AOS2258 | yes | yes (searched) | Confirmed: AoS 51(3):879–908, DOI 10.1214/23-AOS2258, arXiv:1901.09036, COLT 2019 Best Paper. |
| Cluster 2 (line 48) | Namkoong, Keramati, Yadlowsky & Brunskill (NeurIPS 2020), "Off-Policy Policy Evaluation for Sequential Decisions Under Unobserved Confounding" | yes | yes (not searched — bibliographic) | Citation, arXiv ID, and NeurIPS proceedings hash all internally consistent. |
| Cluster 3 (line 67) | Athey & Wager (Econometrica 2021), "Policy Learning with Observational Data," 89(1):133–161 | yes | yes (searched) | Confirmed: Econometrica 89(1):133–161, DOI 10.3982/ECTA15732. |
| Cluster 4 (line 87) | Robins, Hernán & Brumback (Epidemiology 2000), 11(5):550–560 | yes | yes (searched) | Confirmed: Epidemiology 11(5):550–560, Sept 2000, PMID 10955408. |
| Cluster 4 (line 99) | "Q-learning is dynamic programming on conditional means; A-learning is g-estimation; OPE is MSM-style IPTW; FQE is sequential outcome regression" | no | yes (standard) | Mathematical equivalence, not a citable claim per se; correct in standard usage (cf. Schulte–Tsiatis–Laber–Davidian 2014, already cited two lines above). |
| Cluster 6 (line 126) | Mesnard et al. (ICML 2021), "Counterfactual Credit Assignment in Model-Free Reinforcement Learning," PMLR 139, arXiv:2011.09464 | yes | yes (not searched — bibliographic) | All identifiers internally consistent. |
| Cluster 10 (line 203) | Hadad, Hirshberg, Zhan, Wager & Athey (PNAS 2021), 118(15), arXiv:1911.02768 | yes | yes (searched) | Confirmed: PNAS 118(15) e2014602118, DOI 10.1073/pnas.2014602118, arXiv:1911.02768. |
| Cluster 10 (line 215) | Caria, Gordon, Kasy, Quinn, Shami & Teytelboym (JEEA 2024), 22(2):781–836, doi:10.1093/jeea/jvad067, Tempered Thompson Algorithm | yes | yes (searched) | Confirmed: JEEA 22(2):781–836, April 2024, DOI 10.1093/jeea/jvad067, Tempered Thompson Algorithm. |

**No unsupported claims found in the spot-checked sample.** All citations resolve, all bibliographic metadata is exact, and the strongest interpretive claim ("the highest-value addition for an economist audience is Lewis & Syrgkanis") is appropriately framed as the author's recommendation rather than fact.

The author's own Caveats section (lines 263–266) pre-emptively flags the two claims most likely to be challenged: (a) the OPE-bridge statement appearing only in arXiv v5 of Lewis–Syrgkanis (not the NeurIPS proceedings abstract), and (b) the absence of real-data applications in Lewis–Syrgkanis itself. Both caveats are correct and material.

---

## 4. Notation Check

The document uses light-to-moderate math notation in support of prose explanations (no derivations, no theorem statements). The standard is "defined before first use." Six symbols are used without an explicit prior definition in-text. Most are conventional in the relevant subliterature, hence WARN rather than FAIL.

| Symbol | First use | Defined before first use? | Definition location or 'missing' |
|--------|-----------|---------------------------|----------------------------------|
| $t = 1, \dots, m$ (period index, horizon) | Cluster 1, line 30 | no | missing — $m$ never named as horizon; context-only |
| $\theta_t$ (blip function/parameter) | Cluster 1, line 30 | yes (same sentence) | "a 'blip' function $\theta_t$ encodes the marginal causal effect" |
| $T_t$ (treatment blip) | Cluster 1, line 30 | yes (same sentence) | "treatment 'blip' $T_t$" |
| $\bar X_t = (X_1, \dots, X_t)$ (history) | Cluster 1, line 30 | yes | defined inline |
| $\bar T_{t-1}$ (past treatments) | Cluster 1, line 30 | no | analogous to $\bar X_t$ but the bar-as-history convention not stated; relies on Robins-literature reader |
| $\tilde Y_t, \tilde T_t$ (residualised outcome/treatment) | Cluster 1, line 30 | partial | residualisation is described in prose ("residualise the calibrated outcome on history") but the tilde notation itself is not declared |
| $q_t$ (outcome regression nuisance) | Cluster 1, line 30 | yes | "outcome regression $q_t$" |
| $p_{j,t}$ for $j \le t$ (conditional treatment expectations) | Cluster 1, line 30 | yes | "conditional-treatment expectations $p_{j,t}$" |
| $o(n^{-1/2})$, $n^{-1/4}$ | Cluster 1, line 30 | no | $n$ never declared as sample size; standard but unstated |
| $X_t \in \mathbb R^p$, $T_t \in \mathbb R$, $p$ large | Cluster 1, line 32 | yes | defined at use |
| $V^\pi$ (policy value) | Cluster 2, line 50 | no | "value $V^\pi$ of an evaluation policy" — $\pi$ not formally introduced as a policy; standard RL convention |
| $\Gamma$ (Rosenbaum sensitivity parameter) | Cluster 2, line 50 | yes | "Rosenbaum-style sensitivity model bounding the odds-ratio influence of unmeasured confounders by $\Gamma$" |
| $\hat\Gamma_i$ (doubly-robust score, distinct from Rosenbaum $\Gamma$) | Cluster 3, line 69 | yes | "doubly-robust (augmented IPW) scores $\hat\Gamma_i$" — but note collision with Cluster 2's $\Gamma$ |
| $\Pi$ (policy class), $\pi$ (policy) | Cluster 3, line 69 | yes | "policy in a restricted class $\Pi$" |
| $VC(\Pi)$ (VC dimension) | Cluster 3, line 69 | no | $VC(\cdot)$ never defined; assumed |
| $\bar a$, $\bar L$, $\bar l$, $f(\cdot)$ (g-formula objects) | Cluster 4, line 89 | partial | "treatment history $\bar a$" defined; $\bar L$, $\bar l$, and the conditional density $f$ used in the integral without inline definition |
| $\Phi_t$ (future statistics) | Cluster 6, line 128 | yes | "features of the trajectory after time $t$" |
| $A_t$, $h(a \mid x, y)$, $\pi(a \mid x)$ (action, hindsight distribution) | Cluster 6, line 128 | partial | $A_t$ used before being declared; the hindsight ratio is defined inline |
| $\text{CAI}(s) = I(A_t; S'_t \mid S_t = s)$ | Cluster 7, line 147 | yes | full definition at first use |
| $do(X_i = x)$ (intervention) | Cluster 8, line 165 | yes | "arms are interventions $do(X_i = x)$ on a known causal graph" |
| $m^*$ (graph-derived simple-regret quantity) | Cluster 8, line 167 | no | named but not defined; reader must consult Lattimore–Lattimore–Reid |
| $K$ (arm count) | Cluster 8, line 167 | yes | "much smaller than the arm count $K$" |
| $h_t$, $e_t$ (adaptive weight, propensity) | Cluster 10, line 205 | partial | $h_t$ named as "chosen weight," $e_t$ implied to be propensity but never named |

**Summary of notation issues (6 WARN-level):**

1. **Horizon symbol $m$** (Cluster 1) — used as the upper index of $t$ without being named as the horizon.
2. **Sample size $n$** (Cluster 1) — appears in rate expressions $o(n^{-1/2})$, $n^{-1/4}$ without prior declaration.
3. **Notation collision: $\Gamma$ vs. $\hat\Gamma_i$** — Cluster 2's Rosenbaum sensitivity parameter and Cluster 3's doubly-robust score share the symbol family. A reader skimming will conflate them.
4. **$VC(\Pi)$** (Cluster 3) — VC dimension never spelled out.
5. **g-formula objects $\bar L$, $\bar l$, $f(\cdot)$** (Cluster 4) — used inside the g-computation integral without inline definition of the history-vector or the conditional density.
6. **$m^*$** (Cluster 8) — described as a "graph-derived quantity" but never defined; the entire point of the cited regret theorem hinges on what $m^*$ is.

All remaining symbols are either defined at first use or are bog-standard ($V^\pi$, $A_t$, $\pi$ in RL; $do(\cdot)$ in causal inference). For a literature-survey map aimed at researchers already inside the field, this is acceptable; for a tutorial or self-contained paper, items 1, 2, 5, and 6 would each be hard FAILs.

---

*Proofread executed 2026-05-16 by the proofread skill (single-agent inline execution; see meta-note in caller report).*
