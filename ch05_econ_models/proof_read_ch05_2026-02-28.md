# Proofreading Report: Chapter 5 — RL for Structural Economic Models

**File:** `ch05_econ_models/tex/rl_in_se.tex`
**Date:** 2026-02-28
**Scope:** 5 subsections, 23 citations, 13 equations (12 after removing fabricated eq), 3 theorem blocks, 18 footnotes

## Errors Found and Fixed (10)

### Fix 1: AdusumilliEckardt2022 theorem numbering (line 56)
- **Was:** "Theorem 3" for locally robust PMLE asymptotic normality
- **Correct:** Theorem 5 in the paper (Theorem 3 is about AVI estimation rates)
- **Source:** Transcription line 384

### Fix 2: FershtmanPakes2012 description (line 101)
- **Was:** "use iterative value estimation without a standard RL algorithm"
- **Paper explicitly calls its method a "reinforcement learning algorithm"** (transcription lines 39, 261, 263, 265)
- **Fixed to:** "use a stochastic approximation algorithm to update continuation values from simulated industry trajectories"

### Fix 3: AskerEtAl2020 Q-learning mislabeled (lines 101, 111-114)
- **Was:** "add explicit Q-learning updates" with fixed-alpha tabular Q-learning equation
- **Paper uses:** sample averaging (alpha_k = 1/h_k where h_k is visit count), NOT fixed-alpha. Updates all actions including counterfactual.
- **Fixed to:** "add explicit value-function updates via stochastic approximation" with sample-averaging equation and explanatory footnote

### Fix 4: LomysMagnolfi2024 "reinforcement learning" (line 140)
- **Was:** "agents use reinforcement learning"
- **Paper explicitly distinguishes from RL:** agents use regret-minimizing algorithms (paper keywords: "Regret Minimization")
- **Fixed to:** "agents use learning algorithms (specifically regret-minimizing rules)"

### Fix 5: BreroEtAl2021 informal theorem mischaracterized (lines 160-162)
- **Was:** "For settings with at least two items and correlated agent valuations..."
- **Paper:** Prop 1 uses ONE item and IID agents. Only Prop 3 involves correlated valuations.
- **Fixed to:** Accurate enumeration of what each of the four propositions establishes

### Fix 6: BreroEtAl2021 experimental scale wrong (line 166)
- **Was:** "up to 10 agents and 10 items"
- **Paper:** Tests up to 20 agents and 5 items, notes up to 30 of each
- **Fixed to:** "up to 20 agents and 5 items (with similar results noted for up to 30 of each)"

### Fix 7: BreroEtAl2021 "type class" fabricated (line 164)
- **Was:** "whether agents in each 'type class' have purchased or not"
- **Paper:** Phrase "type class" never appears. For independent valuations: remaining items and agents suffice. For correlated: full allocation matrix needed.
- **Fixed to:** Accurate description of sufficient statistics by valuation type

### Fix 8: RavindranathEtAl2024 section rewritten (lines 171-185)
Five fabrications corrected:
1. **Structure wrong:** Was "sequential auction over T rounds, subset auctioned each round." Actually agents visited one at a time; each selects a bundle from remaining items. MDP state is (i_t, S_t).
2. **"Valuations evolve" wrong:** Was "evolve over rounds." Actually drawn once from distributions V_i, fixed throughout.
3. **Equation fabricated:** REINFORCE + pathwise gradient equation (eq:rav_gradient) removed entirely. Paper uses fitted policy iteration with analytical gradients through softmax relaxation. Paper explicitly avoids REINFORCE: "analytical gradients to overcome sample inefficiency, high variance."
4. **Results fabricated:** Was "15-30% higher revenue." Actual range from paper's Tables 1-3: 0.4-13% over item-wise Myerson.
5. **Baselines wrong:** Was "PPO, DQN." Paper compares PPO and SAC; DQN never mentioned. DDPG tried but unstable.

### Fix 9: AtashbarShi2023 "transition equations" (line 190)
- **Was:** "without requiring knowledge of the transition equations"
- **Paper says:** RL bypasses Euler equation and first-order conditions, NOT transition equations. Transition dynamics ARE used in simulation (transcription line 305).
- **Fixed to:** "without deriving optimality conditions such as the Euler equation"

### Fix 10: Hollenbeck2019 TD(0) label imprecise (lines 130-131)
- **Was:** "The TD(0) update"
- **Paper:** Never uses terms "TD(0)" or "temporal difference." Uses "stochastic algorithm of Pakes and McGuire (2001)" with epsilon-decreasing strategy.
- **Fixed to:** "The value function update" with footnote explaining connection to TD learning and noting original uses visit-count averaging
- Also changed "epsilon-greedy" to "epsilon-decreasing" to match paper

## Flagged Issues (not changed)

### F1: HuYang2025 — entirely unverifiable
No transcription exists anywhere in the repository. Entire subsubsection (lines 67-91) with 3 equations (eq:hy_outer, eq:hy_policy, eq:hy_pg) cannot be verified against the source. The bib entry is minimal ("Working Paper, 2025"). Kept as-is pending paper availability.

### F2: Covarrubias2022 — wrong transcription file
File `Covarrubias2022_collusion_drl.md` contains Eschenbaum, Mellgren, and Zahn (2021) "Robust Algorithmic Collusion," NOT Covarrubias (2022). Chapter claims (deep RL, New Keynesian framework, neural networks, multiple equilibria) match the bib entry title but cannot be verified against a transcription. Kept as-is.

### F3: FernandezVillaverdeHurtadoNuno2023 — wrong transcription file
File `Finding_Equilibrium_in_Heterogeneous_Agent_Models_with_Deep.md` contains Curry et al. (Salesforce), not Fernandez-Villaverde et al. Claims about "financial frictions and endogenous wealth distribution" and "perturbation methods fail" are unverifiable against transcription but are well-known descriptions. Kept as-is.

### F4: Zheng2021, MaliarMaliarWinant2021, FernandezVillaverdeNunoPerla2024
No transcriptions available. Brief mentions with standard descriptions of well-known papers. Kept as-is.

## Items Verified Correct

### AdusumilliEckardt2022 (lines 19-64)
- TD learning for CCP estimation, avoiding transition density estimation
- Recursive equations for h(a,s) and g(a,s) with e(a,s) = gamma_E - ln P(a|s)
- Linear semi-gradient TD(0) fixed-point equation
- AVI pseudo-outcomes construction
- "first DDC estimator compatible with arbitrary ML prediction methods"
- Locally robust PMLE restoring sqrt(n)-convergence
- Theorem 1 convergence rate (omits approximation bias term — acceptable simplification)
- "4- to 11-fold reduction in MSE" — exact quote from paper
- "seven structural parameters and five continuous state variables" — exact match
- Games extension: TD works with joint empirical distribution

### AskerEtAl2020 (lines 99-119)
- Dynamic procurement auctions with serially correlated asymmetric info
- Builds on EBE concept of FershtmanPakes2012
- Repeated first-price sealed-bid auction with two firms
- Private inventory state (unharvested timber)
- Winning increases inventory, harvesting depletes it
- Boundary consistency condition
- Information sharing decreases bids, increases profits
- Myopic benchmark shows negligible effects

### Hollenbeck2019 (lines 122-139)
- Dynamic oligopoly with endogenous mergers, entry, exit, quality investment
- Extends Ericson-Pakes framework
- Industry state as firm quality vector
- Bertrand competition with logit demand
- Pakes-McGuire iterative scheme
- Mergers create entry/investment incentives; firms enter with negative static profits
- Reverses standard static antitrust prediction

### LomysMagnolfi2024 (line 140)
- Asymptotic no-regret condition as minimal rationality requirement
- Identification results for payoff parameters
- Estimation method itself is standard econometrics

### BreroEtAl2021 (lines 148-168)
- RL for optimal sequential price mechanisms
- SPMs generalize serial dictatorship and posted-price mechanisms
- Essentially characterize SOSP mechanisms
- POMDP formulation with remaining items, agents, partial allocation
- Action = agent selection + prices
- Reward r = g(x_T, tau_T; v) for welfare/revenue/fairness
- Trained using PPO
- Improvement largest when valuations are correlated

### AtashbarShi2023 (line 186)
- DDPG applied to real business cycle model

### All 23 original citations + 1 new (bertsekastsitsiklis1996) verified present in refs.bib

## Compilation

Clean build: 15 pages, no undefined citations, no errors.
Output: `docs/ch05_econ.pdf`
