# Proofread Report: Chapter 8 (RLHF) -- Prose Only (Lines 1-46)

Date: 2026-03-01
Scope: Sections 1-3 (prose). Simulation section (lines 47-77) excluded per request.

## Summary

- 9 citation keys, all confirmed in refs.bib
- 5 equations, all verified against source papers (Rafailov 2023, Korbak 2022)
- 0 factual errors
- 3 style fixes applied
- 2 items flagged but left unchanged

## Section: Learning Rewards from Preferences (lines 3-9)

### Equations
- Eq 1 (BT loss, lines 6-9): Matches Rafailov Eq 2 exactly (notation mapping: s=x, theta=phi).

### Citations
- bradley1952rank, mcfadden:1973, christiano:2017: All correct.

### Factual claims
- Bradley-Terry preference model formulation: correct.
- Connection to McFadden's multinomial logit: correct.

### Flagged (no change)
- F1: "a direct application of discrete choice theory" (line 5). Bradley-Terry (1952) predates McFadden (1974), so calling BT "a direct application of discrete choice theory" is mildly anachronistic. However, the sentence also says "equivalent to a binary logit model," which is correct. The phrasing reads BT as belonging to the class of models studied in discrete choice theory, which is accurate even if the chronology is reversed. Kept as-is.
- F2: $\mathcal{Y}$ not formally introduced (line 5). The reward model signature uses $\mathcal{Y}$ without definition. Context makes clear it is the output space. Acceptable.

## Section: The RLHF Pipeline and Direct Optimization (lines 11-39)

### Equations
- Eq 2 (KL-regularized objective, lines 16-19): Matches Rafailov Eq 3 with lambda_KL = beta. Correct.
- Eq 3 (Bayesian posterior, lines 23-26): Matches Korbak et al. central result and Rafailov Eq 4. Correct.
- Eq 4 (DPO reparameterization, lines 30-33): Follows from Eq 3 by log rearrangement. Matches Rafailov Eq 5. Correct.
- Eq 5 (DPO loss, lines 35-38): Matches Rafailov Eq 7 exactly.

### Citations
- ziegler2019fine, stiennon2020learning, ouyang2022training, korbak2022rl, rafailov2023direct: All correct.

### Factual claims verified
- Three-stage RLHF pipeline (SFT, reward model, RL): correct.
- KL penalty as reward hacking mitigation: correct.
- Korbak's Bayesian interpretation (KL-regularized RL = variational inference): correct.
- DPO reparameterization (partition function cancels): correct.
- DPO as binary cross-entropy on policy log-ratios: correct.

### Style fixes applied
- S1 (line 27): "solid theoretical justification...fundamental component...mere regularizer" -> "theoretical justification...structural component of the inference problem...ad-hoc regularizer". Removed evaluative "solid" and "fundamental"; "mere" -> "ad-hoc" matches setup language.
- S2 (line 29): "Despite this elegance" -> "Despite this closed-form characterization". Replaced aesthetic judgment with factual description.
- S3 (line 27): "from a pure reward-maximization task into a principled probabilistic inference problem" -> "from a reward-maximization task into a probabilistic inference problem". Removed unnecessary "pure" and evaluative "principled".

### Notation
- Consistent throughout: s for context, y for output, pi_phi for policy, lambda_KL for penalty weight.
- Footnote on line 20 correctly explains notation choices (lambda_KL vs beta).

### Cross-references
- All 5 internal equation refs resolve: rlhf_loss, rlhf_objective, rlhf_posterior, dpo_reparameterize, dpo_loss.

## Section: Recent Developments (lines 41-45)

### Citations
- yuan2024self: Correct. Self-Rewarding Language Models description matches paper.

### Factual claims verified
- Identification footnote (reward identified up to additive constant): correct.
- Alignment tax attribution to ouyang2022training: correct.

### No issues found.

## Style Compliance

- No \textbf{}: Confirmed.
- No em dashes: Confirmed.
- No bullet points: Confirmed.
- Footnotes for technical details: Three footnotes present (notation, identification, alignment tax). Appropriate.
- Objective tone: Confirmed after 3 fixes.

## Known Issue (out of scope)

The simulation section (lines 47-77) has a text-results mismatch flagged in prior session work. Excluded from this proofread per user request.
