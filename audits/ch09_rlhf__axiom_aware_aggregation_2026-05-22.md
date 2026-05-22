# Audit — `ch09_rlhf/sims/axiom_aware_aggregation.py`

**Date:** 2026-05-22
**Auditor:** main agent, end-of-session
**Sim:** Bradley-Terry MLE vs Leximax Copeland subject to PO (LCPO)
**Anchor:** Ge, Halpern, Micha, Procaccia, Shapira, Vorobeychik, Wu, "Axioms for AI Alignment from Human Feedback," NeurIPS 2024
**Bib key:** `ge2024axioms`
**Tex section:** §5.7 (subsubsections 5.7.2–5.7.4) of `ch09_rlhf/tex/rlhf.tex`

## 1. Algorithm identity
BT-MLE implementation is logistic maximum likelihood over the linear-in-features reward family $r_\theta(c) = \langle \theta, x_c \rangle$, optimised by L-BFGS-B. Matches paper §3.1 loss definition. LCPO implementation is sequential rank assignment: for each unfilled position, pick the unranked candidate with the highest Copeland score for which an LP-feasibility test (via `scipy.optimize.linprog`) confirms a $\theta$ exists consistent with the partial ranking + Pareto-dominance constraints. Matches paper §4 description (lines 235–237, 265–266 of the docling extraction). Tiebreak via lexicographic comparison of sorted margin vectors. PASS.

## 2. Environment fidelity
Construction is the paper's Theorem 3.1 6-candidate setup at lines 177–179 of the docling extraction: $x_a = (2,1)$, $x_b = (1,1)$, $x_c = (0,0)$, $x_{a'} = x_a + (-\varepsilon, 0)$, $x_{b'} = x_b + (-\varepsilon, 0)$, $x_{c'} = x_c + (-\varepsilon, \delta\varepsilon)$, with voter type 1 parameter $(1,1)$ and type 2 parameter $(-1, 0)$. Caveat: paper Lemma constraint says $\delta \in (0, 1)$, but with that range and the stated $x_{c'}$ formula, voter type 1's reward on $c'$ is $(\delta - 1) \varepsilon < 0 = r(c)$, contradicting the paper's claim (footnote 7 of extraction) that $r(c') = (1 - \delta) \varepsilon > 0$ for type 1. Either footnote 7 has a sign typo or $x_{c'}$ should have $+\varepsilon$ rather than $-\varepsilon$. Sim uses $\delta = 2$ so the Pareto-dominance $c' \succ c$ actually holds on both voter types; the qualitative Theorem-3.1 prediction (BT-MLE asymptotic PO + PMC failure) reproduces. FLAGGED (paper inconsistency, sim documents the override).

## 3. Data integrity
`compute_data()` runs `bt_mle_linear` and `leximax_copeland_po` on freshly sampled pairwise comparisons each seed. No hardcoded outcomes. Caches keyed on full `SHARED_CONFIG`, `BT_CONFIG`, `COPELAND_CONFIG` via MD5 of JSON-serialised dicts. Stdout printed at end of `generate_outputs` reads from `data` only. PASS.

## 4. Comparison fairness
Both methods receive identical comparison samples per seed (seeded RNG). Same sweep grid $N \in \{5, 10, 20, 50, 100, 500, 2000\}$ pairs per candidate pair. 30 seeds each. Evaluation criteria fixed in advance: PO violation indicator from oracle dominance set, PMC violation indicator from oracle PMC ranking (type 1's induced ranking under $p > 1/2$), worst-group utility under both voter types. Note: LCPO consumes the oracle Pareto-dominance set as a hard constraint, but this is the paper's algorithm specification (§4 "leximax Copeland subject to PO"), not external information. PASS.

## 5. Theoretical sanity
At $N = 2000$, BT-MLE PO violation rate = $1.00 \pm 0.00$ and PMC violation rate = $1.00 \pm 0.00$ over 30 seeds. Asymptotic prediction of Theorem 3.1 reproduces exactly. LCPO PO violation rate = $0.00$ at every sample size (axiom enforced by the dominance constraint). LCPO PMC violation rate falls to $0.00$ at $N \geq 100$. Theorem 4.3 prediction reproduces. PASS.

## 6. No information leakage
BT-MLE sees only the sampled $\{(i, j, \text{winner})\}$ tuples. LCPO sees comparisons plus the oracle Pareto-dominance set per paper specification. Neither method sees ground-truth voter parameters, voter types per comparison, or the population fraction $p$ except implicitly through sampled labels. PASS.

## 7. Seed and reproducibility
`np.random.default_rng(seed)` per outer seed; 30 seeds. Mean and standard error reported in summary and table. Cache hashes invalidate on any config change. Re-running with cached state yields identical numbers. PASS.

## Result

| N | BT_PO | BT_PMC | CP_PO | CP_PMC | BT_wu | CP_wu |
|---|---|---|---|---|---|---|
| 5 | 0.600 | 0.833 | 0.000 | 0.500 | $-1.531$ | $-1.328$ |
| 10 | 0.533 | 0.700 | 0.000 | 0.467 | $-1.832$ | $-1.462$ |
| 20 | 0.400 | 0.500 | 0.000 | 0.367 | $-1.899$ | $-1.596$ |
| 50 | 0.600 | 0.633 | 0.000 | 0.133 | $-1.966$ | $-1.865$ |
| 100 | 0.767 | 0.767 | 0.000 | 0.000 | $-2.000$ | $-2.000$ |
| 500 | 0.933 | 0.933 | 0.000 | 0.000 | $-2.000$ | $-2.000$ |
| 2000 | 1.000 | 1.000 | 0.000 | 0.000 | $-2.000$ | $-2.000$ |

## Bullshit score

**Bullshit score: 15%** — Reviewer 2 catches the $\delta = 2$ vs paper's stated $\delta \in (0, 1)$ discrepancy (likely paper sign typo on $x_{c'}$'s second coordinate) and the LCPO oracle-dominance caveat. Substance holds: Theorem 3.1 and Theorem 4.3 reproduce cleanly on the 6-candidate construction with the documented parameter override.
