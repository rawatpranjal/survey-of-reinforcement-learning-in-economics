# Proofread Report: Chapter 6 — RL in Games

**File:** `ch06_games/tex/rl_in_games.tex`
**Date:** 2026-03-01
**Compiled:** `docs/ch06_games.pdf` (7 pages, no warnings)

## Errors Fixed (4)

### Fix 1: Daskalakis2009 scope (line 3)
**Was:** "Computing Nash equilibria in extensive-form games with imperfect information is computationally hard. Daskalakis et al. showed the problem is PPAD-complete"
**Problem:** Daskalakis, Goldberg, and Papadimitriou (2009) prove PPAD-completeness for normal-form (bimatrix) games, not extensive-form games.
**Now:** Correctly states PPAD-completeness for two-player normal-form games. Adds that extensive-form games inherit this hardness.

### Fix 2: Deep CFR exploitability (line 43)
**Was:** "exploitability of 0.036 big blinds per hand in heads-up limit hold'em"
**Problem:** Paper reports 37 mbb/g in heads-up flop hold'em (FHP), not "0.036 big blinds" in HULH. Unit inconsistent with footnote.
**Now:** "exploitability of 37 mbb/g (milli-big-blinds per game) in heads-up flop hold'em." Footnote updated to match (mbb/g).

### Fix 3: Deep CFR loss — missing iteration weighting (line 27)
**Was:** `L_V(θ) = E_{(I,a,r)~M} [(V_θ(I,a) - r)²]`
**Problem:** Brown (2019) Algorithm 1 weights by iteration index t': `L(θ) = E [t' · (V(I,a|θ) - r)²]`. This gives more recent regret estimates higher importance.
**Now:** Loss includes t' weighting. Added sentence explaining the rationale.

### Fix 4: Deep CFR average strategy — KL-divergence wrong (line 30)
**Was:** "approximates the average strategy via KL-divergence minimization"
**Problem:** Brown (2019) uses MSE loss, not KL-divergence. Line 131 of transcription: "One can use any loss function... such as mean squared error loss."
**Now:** "trained via weighted MSE on strategy samples with the same iteration weighting."

## Conceptual Issue Fixed (1)

### C1: Poker-bargaining analogy (line 98)
**Was:** "The poker-bargaining correspondence is exact: buyer valuation maps to hole cards, price acceptance to calling a bet, rejection to folding."
**Problem:** Folding = irrevocable exit (forfeiting future payoffs). Rejection in bargaining = staying in the game and waiting for a better offer. Opposite strategic implications. "Exact" overstates the correspondence.
**Now:** "The poker-bargaining correspondence is structural: buyer valuation maps to hole cards, price acceptance to calling a bet, and both games involve signaling private information through sequential actions."

## Minor Omission Fixed (1)

### O1: Δ undefined in convergence bound (line 17)
**Was:** ε = O(|I|√|A|Δ / √T) with Δ undefined.
**Now:** Added "where Δ is the range of payoffs."

## Flagged but Not Changed

- **F1:** "~10,000 iterations" for exploitability 0.01. Illustrative, problem-specific. Acceptable.
- **F2:** "~10^161 states" for HUNL. Widely cited estimate. Acceptable.
- **F3:** Economics claims (Coase, Stokey, Bulow, Gul). Standard textbook results. No transcriptions needed.
- **F4:** 3-period stress test P₁=120 vs theory 136. Verified against simulation code. Correct.

## Items Verified Correct

- All 14 citation keys confirmed in refs.bib
- CFR theory: counterfactual value equation, regret matching, convergence rate
- CFR+: non-negative clipping + linear averaging description
- NFSP: two-network architecture, DQN loss (γ=1 for poker, standard general form acceptable), SL loss, η mixing
- Coase conjecture theory: all attributions correct (Coase 1972, Stokey 1981, Bulow 1982, Gul 1986)
- Screening price algebra: P*(δ), π* threshold, all verified
- Simulation results: all 17 rows of durable_goods_results.tex match prose
- Stress tests: verified against coase_stress_tests.py
- PPAD footnote: correct characterization
- Information set definition: correct
