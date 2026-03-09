# Proofreading Report: Chapter 2 — Reinforcement Learning Algorithms

**File:** `ch02_rl_algorithms/tex/rl_algorithms.tex` (177 lines)
**Date:** 2026-02-28
**Method:** Systematic claim verification against source paper transcriptions, notation audit, jargon audit for economist audience, style compliance check, cross-reference verification, bibliography verification.

---

## Summary

| Category | Count |
|----------|-------|
| Errors requiring correction | 6 |
| Missing citations (critical) | 3 |
| Notation inconsistencies | 3 |
| Unexplained jargon for economists | 12 |
| Claims that cannot be verified (missing transcription) | 4 |
| Partially verified claims | 3 |
| Verified claims | ~40 |
| Style issues | 0 |

---

## Section-by-Section Findings

### 2.1.1 Sutton (1988) — Lines 11–19

**Verified:**
- Equal probability left/right movement
- Right terminal reward 1, left terminal reward 0 (Sutton uses "outcome" z, not "reward" r; the modern reframing is acceptable)
- True state values 1/6, 2/6, 3/6, 4/6, 5/6
- λ=0 yields pure bootstrapping; λ=1 recovers Monte Carlo

**ERROR 1 — Random walk states (Line 13):**
The chapter says "States A through E are arranged in a line with absorbing terminal states at each end." Sutton's random walk has **seven** states A–G. States B, C, D, E, F are the five nonterminal states; A and G are the absorbing terminals. Writing "States A through E" implies A is nonterminal.

*Fix:* Change to "five nonterminal states B through F, with absorbing terminals A and G at each end" or simply "five nonterminal states arranged in a line with absorbing terminals at each end."

**ERROR 2 — TD(λ) equation attribution (Lines 16–19):**
The TD(λ) update V(s_t) ← V(s_t) + α δ_t e_t(s) with δ_t = r_{t+1} + γV(s_{t+1}) − V(s_t) and e_t(s) = γλ e_{t−1}(s) + 1{s=s_t} is the **modern textbook formulation** from Sutton & Barto (1998), not what appears in Sutton (1988). The 1988 paper works in terms of weight vectors w, observations x_t, and predictions P_t = w^T x_t. The TD error is (P_{t+1} − P_t), not r_{t+1} + γV(s_{t+1}) − V(s_t). There is no discount factor γ and no per-step reward r_{t+1} in the core formulation. The eligibility trace accumulates weight gradients ∇_w P_k, not per-state indicators. The decay is λ alone, not γλ.

*Fix:* Either (a) note that the equation is the modern formulation that generalizes Sutton's original, or (b) present Sutton's original notation first and then map to modern form, consistent with CLAUDE.md style ("use the author's original notation first, then map to modern RL notation").

**Imprecision — "TD(0) converged faster" (Line 13):**
Under repeated presentations, TD(0) achieved the lowest RMS error. Under single presentation (the more practically relevant case), intermediate λ ≈ 0.3 was optimal, not λ = 0. The claim "TD(0) converged faster and with less data" is an oversimplification.

*Suggested fix:* "TD methods with small λ converged faster and with less data than Monte Carlo; in single-presentation experiments, intermediate λ ≈ 0.3 performed best."

---

### 2.1.2 Watkins (1989) — Lines 21–35

**All claims verified:**
- Q-learning formalizes Watkins's 1989 PhD thesis ✓
- Bellman optimality equation (Eq 29) ✓
- Q-learning update rule (Eq 33) ✓
- Off-policy nature via max over a' ✓
- Convergence under standard regularity conditions ✓

**Notation note:** The original paper uses r_n (same-step indexing). The chapter uses r_{t+1} (Sutton-Barto convention). Both are standard; the chapter is internally consistent here.

**Missing citation:** The cross-reference to "Section~\ref{sec:stochastic_approx}" (line 35) resolves correctly to ch03_theory/tex/planning_learning_v3.tex.

---

### 2.1.3 Williams (1992) — Lines 37–45

**Verified:**
- Policy gradient equation (Eq 41) ✓
- G_t definition ✓
- Log-derivative trick ✓
- Variance reduction with baselines ✓

**Note — Robot arm example (Line 45):**
The continuous torque robot arm example is a pedagogical illustration by the chapter author, not from Williams (1992). This is acceptable since it is not attributed to Williams, but the framing "Consider a robot arm..." could mislead a reader into thinking this is from the paper.

**Note — Footnote claim (Line 45):**
"Virtually all continuous-control results in reinforcement learning descend from the policy gradient framework" is the chapter author's editorial assessment, not a claim from Williams. Appropriately placed in a footnote. This is a strong claim that could be softened.

---

### 2.1.4 Tesauro (1994) — Lines 47–59

**Verified:**
- 80 hidden sigmoid units ✓
- Network equation (Eq 51) ✓
- Weight update equation (Eq 55) ✓
- Eligibility trace formulation ✓
- 300,000 self-play games for Version 1.0 ✓
- Version 2.1 near-parity with Bill Robertie ✓
- Robertie was a "former world champion" ✓

**Partially verified:**
- λ = 0.7: This value comes from Tesauro (1992), not the 1994 paper. The 1994 transcription does not state λ = 0.7 explicitly.
- "Discovered novel positional strategies adopted by the human community": Documented in later Tesauro publications, not this specific 1994 paper.
- ~10^20 legal positions: Standard figure, not contradicted, but not stated in the 1994 paper text.

**Note:** "The value function varied smoothly with board position" (line 59) is the chapter author's interpretation. The paper discusses the role of stochastic dice in ensuring exploration but does not use the term "smooth value function."

---

### 2.1.5 SARSA (1994) — Lines 61–71

**Verified:**
- SARSA update rule (Eq 65) ✓
- On-policy nature (bootstraps from a_{t+1} actually taken) ✓
- ε-greedy SARSA converges to Q^{ε-greedy} ✓

**Cannot verify (missing transcription):**
No transcription exists for Rummery & Niranjan (1994). The following claims cannot be verified against the original source:
- Whether Rummery coined the name "SARSA" (the name may have originated with Sutton)
- The specific formulation used in the original paper

**Attribution concern — Cliff-walking example (Lines 69–70):**
The cliff-walking example is from Sutton & Barto (1998, Chapter 6), not from Rummery (1994). It should be attributed or noted as an illustrative example, e.g., "Consider the cliff-walking problem \citep{sutton2018}:..."

**Jargon — GLIE (Line 71):**
The acronym is expanded ("greedy in the limit with infinite exploration") but the conditions are not explained. An economist reader will not know what GLIE requires. At minimum, a footnote should state: "A GLIE policy explores all state-action pairs infinitely often but converges to the greedy policy in the limit."

---

### 2.1.6 Baird (1995) — Lines 73–77

**ERROR 3 — State count (Line 75):**
The chapter says "seven-state star MDP" with "six outer states." Baird's paper defines a **six-state** star MDP: five outer states (1–5) and one inner state (6). The chapter should say "six-state star MDP" with "five outer states."

*Fix:* Change "seven-state" to "six-state" and "six outer states" to "five outer states."

**Verified:**
- Off-policy behavior samples uniformly ✓
- Weights grow without bound ✓
- Three-component deadly triad formulation ✓
- Residual gradient solution proposed ✓
- TD-Gammon contrast explanation ✓

---

### 2.1.7 Actor-Critic Methods (2000) — Lines 79–95

**Verified:**
- Barto (1983) pole-balancing with actor-critic ✓
- Konda (2000) two-timescale convergence, compatibility condition ✓
- A3C: parallel CPU threads, shared parameters, Hogwild-style ✓
- A3C: n-step returns ✓
- A3C: no experience replay buffer ✓
- A2C description ✓

**ERROR 4 — Reward indexing inconsistency (Line 85):**
The actor-critic TD update uses δ_t = r_t + γV(s_{t+1}) − V(s_t). The rest of the chapter (lines 19, 33, 65) consistently uses r_{t+1}. This is an internal inconsistency.

*Fix:* Change `r_t` to `r_{t+1}` on line 85 for consistency.

**ERROR 5 — A3C training time (Line 95):**
The chapter says "training in half a day on a single multi-core CPU." The paper actually says **"half the training time of DQN on a GPU, using a single multi-core CPU."** Table 1 in the paper shows actual training times of 1 day and 4 days for different configurations on 16 cores. "Half a day" is not what the paper claims.

*Fix:* Change to "training in half the time required by DQN on a GPU, using only a single multi-core CPU" or cite actual training times.

**Missing citation — Policy gradient theorem:**
The actor-critic section uses the TD error as an advantage estimate (line 91: "δ_t estimates A^π(s_t, a_t)") without citing \citet{SuttonMcAllester2000}, who proved this is valid under the policy gradient theorem. This is the theoretical foundation for using TD error in the actor update. The key `SuttonMcAllester2000` exists in refs.bib but is not cited.

*Fix:* Add citation, e.g., "The policy gradient theorem \citep{SuttonMcAllester2000} shows that the TD error provides an unbiased sample of the advantage."

---

### 2.2.1 Natural Policy Gradient (2001) — Lines 97–109

**Verified:**
- Fisher information matrix definition (Eq 101) ✓
- Natural gradient formula (Eq 105) ✓
- Invariance to reparameterization ✓
- Tabular softmax: one NPG step = one PI step ✓
- Conjugate gradient for practical implementation ✓

**Citation date note:**
Kakade2001 corresponds to the NIPS 2001 presentation (proceedings published 2002). The bibliography entry has year=2001. Both 2001 and 2002 are defensible; the current choice is consistent.

---

### 2.2.2 Deep Q-Networks (2015) — Lines 115–128

**Cannot verify (transcription issue):**
The file `ch01_history/papers/Human-level_control_through_deep_reinforcement_learning.md` contains the **2013 preprint** ("Playing Atari with Deep Reinforcement Learning," 7 games, 2 conv layers), not the 2015 Nature paper (49 games, 3 conv layers, target network). The following claims from the 2015 paper cannot be verified against available transcriptions:
- 49 Atari games
- Three convolutional layers (the 2013 preprint has only two)
- Target network updated every C=10,000 steps
- 29/49 games exceed human level

These claims are well-established in the literature and almost certainly correct, but the transcription should be corrected.

**Verified (from 2013 preprint):**
- Four consecutive frames ✓
- Replay buffer 10^6 transitions ✓

**Verified (general):**
- "Two decades after TD-Gammon" — 1994 to 2015 is 21 years. "Two decades" is approximately correct. ✓

**Missing citation — Experience replay origin:**
Experience replay is attributed to DQN by context. Lin (1992) introduced experience replay and deserves citation. This reference does not exist in refs.bib.

*Fix:* Add Lin (1992) to refs.bib and cite: "Experience replay \citep{lin1992} stored transitions..."

---

### 2.2.3 TRPO and PPO (2015, 2017) — Lines 130–146

**Verified:**
- TRPO constrained optimization (Eq 136) ✓
- KL divergence constraint ✓
- PPO clipped surrogate (Eq 142) ✓
- Probability ratio definition ✓
- Clipping interval [1−ε, 1+ε] ✓

**Imprecision — PPO "30/49 Atari games" (Line 146):**
The paper reports PPO achieved best average reward on 30/49 games, but the comparison set is A2C and ACER only (not all methods ever tested). The chapter says "among the methods tested," which is technically accurate but could be more precise about the comparison set.

**Notation overload — r_t(θ) (Lines 142, 144):**
The probability ratio r_t(θ) uses the same symbol r as reward (used as r_{t+1} elsewhere). This could confuse readers. Consider using ρ_t(θ) or noting the overload explicitly.

---

### 2.2.4 Soft Actor-Critic (2018) — Lines 148–160

**Verified:**
- Entropy-regularized objective (Eq 152) ✓
- Entropy definition ✓
- Temperature parameter τ ✓
- Softmax policy in Q-values ✓
- Dual Q-networks ✓
- Soft Bellman operator (Eq 158) ✓
- Connection to discrete choice models ✓

**Notation inconsistency — r_t (Line 152):**
The SAC objective uses r_t, but the chapter convention established in lines 19, 33, 65 is r_{t+1}. This is the same inconsistency as in the actor-critic section (line 85).

*Fix:* Use r_{t+1} for consistency, or add a footnote explaining the two conventions.

---

### 2.2.5 AlphaGo and AlphaGo Zero (2016, 2017) — Lines 162–172

**Verified:**
- ~10^170 legal positions, branching factor ~250 ✓
- 30 million positions from human expert games ✓
- 57% accuracy on expert moves ✓
- REINFORCE for self-play RL ✓
- Lee Sedol 4–1, March 2016 ✓
- AlphaGo Zero single network f_θ(s) = (p, v) ✓
- 72 hours on 4 TPUs ✓
- Loss function (Eq 168) ✓
- Igami CCP/CVF interpretation ✓
- Hotz-Miller connection ✓

**Missing citation — Hotz & Miller (Line 172):**
The chapter references "the Hotz-Miller approach" without citing \citet{HotzMiller1993} or \citet{hotz1993}. Both keys exist in refs.bib.

*Fix:* Add citation: "connecting to the \citet{HotzMiller1993} approach in dynamic discrete choice."

**ERROR 6 — θ vs bold θ (Line 168):**
AlphaGo Zero's loss function uses non-bold θ for the full network weight vector. The Tesauro section (line 53) established the convention that bold θ denotes the full vector of network weights, with a footnote explaining this. Line 168 should use bold θ for consistency.

*Fix:* Change `\theta` to `\boldsymbol{\theta}` in the AlphaGo Zero loss function and surrounding text, or at minimum in `\|\theta\|^2`.

---

### 2.2.6 Toward a Unified Framework — Lines 174–177

No claims requiring verification. Cross-reference to Section~\ref{sec:planning_learning} resolves correctly.

---

## Cross-Reference Verification

All five cross-referenced labels resolve to `ch03_theory/tex/planning_learning_v3.tex`:

| Reference | Target | Status |
|-----------|--------|--------|
| `\ref{sec:stochastic_approx}` | Line 42 | ✓ |
| `\ref{sec:deadly_triad}` | Line 81 | ✓ |
| `\ref{sec:actor_critic}` | Line 162 | ✓ |
| `\ref{sec:policy_gradient}` | Line 129 | ✓ |
| `\ref{sec:planning_learning}` | Line 7 | ✓ |

---

## Bibliography Verification

All 18 citation keys exist in `docs/refs.bib`:
sutton1988, WatkinsDayan1992, williams1992, tesauro1994, rummery1994, Baird1995, sutton2018, barto1983neuronlike, konda2000, mnih2016a3c, Kakade2001, mnih2015, Schulman2015, Schulman2017, Haarnoja2018, Silver2016, Silver2017, Igami2020.

---

## Missing Citations (Critical)

| Citation | Location | Reason |
|----------|----------|--------|
| SuttonMcAllester2000 | Actor-critic section (line 91) | Policy gradient theorem is the theoretical basis for using δ_t as advantage estimate. Key exists in bib but is not cited. |
| Lin (1992) | DQN section (line 124) | Experience replay originated with Lin, not Mnih. Not in refs.bib. |
| HotzMiller1993 | AlphaGo section (line 172) | "Hotz-Miller approach" referenced without citing the original paper. Key exists in bib. |

---

## Notation Inconsistencies

| Issue | Lines | Fix |
|-------|-------|-----|
| r_t vs r_{t+1} | 85, 152 use r_t; 19, 33, 65 use r_{t+1} | Standardize to r_{t+1} |
| r_t(θ) overloads reward symbol | 142, 144 | Consider ρ_t(θ) or add footnote |
| θ vs bold θ for network weights | 168 uses θ; 53–57 established bold θ convention | Use bold θ in AlphaGo section |

---

## Unexplained Jargon (Economist Audience)

Terms used without definition or footnote that an economist or social scientist may not know:

| Term | First Use (Line) | Suggested Action |
|------|-------------------|------------------|
| bootstrapping | 19 | Add footnote: "updating from estimated values rather than observed returns" |
| on-policy | 67 | Add explicit parenthetical or footnote (off-policy is defined but on-policy is not) |
| GLIE | 71 | Add footnote explaining the conditions |
| convolutional neural network | 117 | Add footnote describing spatial filter architecture |
| convolutional layers | 119 | Covered by above |
| softmax | 107, 154 | Add footnote with formula: π(a) = exp(z_a) / Σ exp(z_j) |
| learning rate α | throughout | Define on first use (line 17) as step-size parameter |
| feedforward neural network | 49 | Add brief footnote |
| Hogwild-style lock-free writes | 95 | Add footnote or simplify to "asynchronous updates without coordination" |
| L2 regularization | 170 | Add footnote: "penalty on squared magnitude of parameters, preventing overfitting" |
| cross-entropy loss | 170 | Add footnote with connection to log-likelihood |
| advantage function | 91 (used), 138 (defined) | Move definition/footnote to first use at line 91 |

Terms that are adequately explained:
- eligibility trace (footnote, line 15) ✓
- off-policy (footnote, line 35) ✓
- KL divergence (footnote, line 138) ✓
- Gaussian density (footnote, line 45) ✓
- θ^- target network (footnote, line 124) ✓
- bold θ (footnote, line 53) ✓

---

## Style Compliance

| Check | Status |
|-------|--------|
| No em dashes | PASS |
| No \textbf{} | PASS |
| No bullet points | PASS |
| No stub paragraphs (1–2 lines) | PASS |
| Footnotes for technical details | PASS |
| Objective tone | PASS (one editorial footnote about continuous control, line 45) |

---

## Transcription Issues (Not Chapter Errors)

These are issues with the paper transcription files, not with the chapter text:

1. `/ch01_history/papers/A_Natural_Policy_Gradient.md` contains wrong content (a legal contract paper by Daskalopulu & Sergot). The correct Kakade NPG transcription is at `/ch03_theory/papers/kakade2002_natural_policy_gradient.md`.

2. `/ch01_history/papers/Human-level_control_through_deep_reinforcement_learning.md` is mislabeled. The filename matches the 2015 Nature paper, but the content is the 2013 preprint "Playing Atari with Deep Reinforcement Learning" (7 games, 2 conv layers). The 2015 Nature paper should be transcribed to verify the remaining DQN claims.

3. No transcription exists for Rummery & Niranjan (1994). This prevents verification of SARSA-specific claims and the naming attribution.

---

## Consolidated Error List (Action Items)

1. **Line 13:** Change "States A through E" to "five nonterminal states B through F, with absorbing terminals A and G"
2. **Lines 16–19:** Note that the TD(λ) equation is the modern textbook form, not Sutton's 1988 formulation
3. **Line 75:** Change "seven-state" to "six-state" and "six outer states" to "five outer states"
4. **Line 85:** Change r_t to r_{t+1} for consistency with rest of chapter
5. **Line 95:** Change "half a day" to "half the training time of DQN on a GPU"
6. **Line 152:** Change r_t to r_{t+1} for consistency (or add footnote about convention)
7. **Line 168:** Change θ to bold θ in AlphaGo Zero loss function
8. **Line 91:** Add citation for SuttonMcAllester2000 (policy gradient theorem)
9. **Line 124:** Add citation for Lin (1992) experience replay (add to refs.bib first)
10. **Line 172:** Add citation for HotzMiller1993
11. **Lines 142, 144:** Consider renaming r_t(θ) to ρ_t(θ) to avoid overload with reward
12. **Line 69:** Attribute cliff-walking example to Sutton & Barto textbook
13. Add footnotes for 12 unexplained jargon terms (see table above)
