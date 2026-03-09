# Proofread Report: Chapter 4 — RL for Optimal Control

**File:** `ch04_control_problems/tex/applications.tex`
**Date:** 2026-02-28
**Scope:** 209 lines, 14 citations, 6 equations, 8 footnotes, 5 inline tables, 1 figure, 1 `\input` table

## Errors Found and Fixed (3)

### Fix 1: Nevmyvaka execution range (line 143)
- **Was:** "Improvements of 15--20\% over the Almgren-Chriss baseline"
- **Now:** "Improvements of 12--19\% over the Almgren-Chriss baseline"
- **Reason:** Table shows AMZN -14.6%, QCOM -19.3%, NVDA -12.1%. Range is 12.1–19.3%, not 15–20%.

### Fix 2: Chen2023 data description (line 99)
- **Was:** "training on historical booking data from multiple properties"
- **Now:** "training on simulated booking data under parameterized demand models"
- **Reason:** Paper uses a synthetic MDP with Poisson demand. No real historical data, no multiple properties.

### Fix 3: DiDi table (lines 34–37)
- **Was:** Response rate +0.3–0.8%, Fulfillment rate +0.2–0.5%, Avg pickup distance -5–10%
- **Now:** All three remaining metrics (income, response, fulfillment) reported as +0.5–2.0%
- **Reason:** Qin2021 groups all metrics under "0.5%–2%" improvement. No separate breakdowns for response/fulfillment. Pickup distance improvement is not reported in the paper. Row removed.

## Jargon Footnote Added (1)

### J1: Value decomposition (line 44)
Added footnote after "value decomposition architecture" explaining the concept for economists: decomposes global matching objective into individual driver value functions estimable via TD learning.

## Flagged Issue (not changed)

### DiDi pickup distance claim
The "-5–10% avg pickup distance" removed from the table is not in Qin2021dispatch. It may originate from Tang2019cvnet or another DiDi paper. Removing it is the conservative choice.

## Verification Results

### Citation Verification (14/14 confirmed in refs.bib)
All citation keys resolve. No undefined citation warnings in compilation.

### Section-by-Section Findings

#### 4.1 Ride-Hailing Dispatch (lines 7–46)
- ✓ Semi-MDP formulation matches Qin2021dispatch exactly
- ✓ Driver state = (hex zone, time bucket), action = order/idle, reward = fare
- ✓ Value function equation correct (semi-Markov discount γ^τ_k)
- ✓ Edge weight formula w_ij matches paper's advantage formulation
- ✓ CVNet hierarchical coarse-coding correctly attributed to Tang2019cvnet
- ✓ A/B test via time-slice rotation, 3-hour blocks — confirmed
- ✓ Li2019ridesharing mean-field MARL — confirmed (WWW 2019)
- ✓ Han2022lyft dispatching + repositioning — confirmed (KDD 2022)
- ~ Han2022lyft: "value decomposition" is functionally accurate but not the paper's exact term ("driver supply values"). Acceptable paraphrase. Footnote added for economists.

#### 4.2 Data Center Cooling (lines 49–83)
- ✓ 40% cooling energy reduction — Gao2014
- ✓ 15% PUE improvement — Gao2014
- ✓ Additional 12% from autonomous MPC — Lazic2018
- ✓ 0 temperature violations — confirmed
- ✓ Neural network dynamics model equation s_{t+1} = f̂(s_t, a_t) + ε_t — correct
- ✓ 5-minute control interval — confirmed
- ✓ MPC optimization with temperature constraints — equation correct
- ✓ Human override capability — confirmed
- ✓ Table correctly attributes results to both papers

#### 4.3 Hotel Revenue Management (lines 86–118)
- ✓ Bellman equation V_t(I, d) — correct
- ✓ Table values: Fixed=2847, Myopic=2962, EMSR-b=3156, DP=3241, DQN=3198 — exact match
- ✓ 98.7% of DP optimal (3198/3241) — exact
- ✓ DQN training time nearly constant vs DP superlinear — confirmed
- ✓ Gallego1994pricing reference for analytical benchmark — correct
- ✗ "historical booking data from multiple properties" — FIXED (see Fix 2)

#### 4.4 Financial Order Execution (lines 121–143)
- ✓ Tabular Q-learning on NASDAQ LOB data — confirmed
- ✓ State: time remaining, inventory, spread, signed volume imbalance — confirmed
- ✓ Table values: AMZN -14.6%, QCOM -19.3%, NVDA -12.1% vs A-C — exact match
- ✓ 500 trading days of LOB data — confirmed
- ✗ Prose range "15–20%" did not match table (12–19%) — FIXED (see Fix 1)

#### 4.5 Supply Chain Inventory (lines 146–175)
- ✓ K-echelon serial system formulation — correct
- ✓ Cost function equation with holding, backorder, ordering components — correct
- ✓ Clark-Scarf echelon base-stock optimality — confirmed
- ✓ Newsvendor critical fractile S* = F_D^{-1}(b/(b+h)) — correct
- ✓ Table values: K=2 +1.2%, K=3 +6.8%, K=4 +12.3%, K=6 failed — exact match
- ✓ Millions of training transitions — confirmed
- ~ State space sizes (10^2, 10^3, 10^5, 10^8) — reasonable inference from M^K

#### 4.6 Real-Time Bidding (lines 178–181)
- ✓ MDP with budget, time, win rates in state — confirmed online
- ✓ Actions are bid multipliers — confirmed
- ✓ Improvements over rule-based pacing — confirmed (18% over BSLB)

#### 4.7 Bus Engine Simulation (lines 184–209)
- ✓ Monthly replacement decision based on mileage — matches Rust1987
- ✓ Footnote correctly distinguishes simulation's simplified cost from Rust's original
- ✓ State space calculation 6^N correct
- ✓ Figure and table references resolve

### Math Verification (6/6 equations)
1. Semi-MDP value function (eq. 1) — correct
2. Edge weight w_ij (eq. 2) — correct
3. Neural network dynamics (eq. 3) — correct
4. MPC optimization (eq. 4) — correct
5. Hotel Bellman equation (eq. 5) — correct
6. Inventory cost function (eq. 6) — correct

### Style Compliance
- ✓ No em dashes
- ✓ No bullet points
- ✓ No \textbf{}
- ✓ No stub paragraphs
- ✓ Technical details in footnotes (8 footnotes, all appropriate)
- ✓ Objective tone throughout
- ✓ Jargon explained: semi-MDP, coarse-coding, PUE, LOB, implementation shortfall, EMSR-b, echelon inventory, credit assignment, value decomposition (added)

### Compilation
- PDF output: `docs/ch04_control.pdf` (11 pages)
- 0 undefined citation warnings
- 0 missing reference warnings
- DiDi table renders with 3 rows (pickup distance removed)
- Value decomposition footnote renders correctly
