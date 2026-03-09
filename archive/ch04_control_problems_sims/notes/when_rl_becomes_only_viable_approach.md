# When reinforcement learning becomes the only viable approach

**Across operations research and economics, a growing body of literature documents problems where exact dynamic programming fails catastrophically, standard heuristics leave substantial value unrealized, and deep reinforcement learning emerges as the uniquely scalable solution.** This synthesis identifies the strongest examples meeting all three criteria, drawn from peer-reviewed sources in top venues including *Nature*, *Management Science*, *Operations Research*, and major ML conferences.

## The canonical benchmarks: video games as existence proofs

The most rigorous documentation of all three conditions comes from game AI, where state spaces are precisely calculable, pre-RL baselines are well-documented, and RL achievements are unambiguous.

**Go** presents the clearest case. The game's **10^170 legal positions** and branching factor of ~250 moves per turn make brute-force search impossible—more configurations exist than atoms in the universe. Before AlphaGo, the strongest programs reached only amateur level despite decades of research; Monte Carlo Tree Search alone could not compensate for the lack of reliable position evaluation. Silver et al. (2016, *Nature*) demonstrated that deep neural networks trained via RL reduced the effective branching factor enough to achieve superhuman play, defeating world champion Lee Sedol 4-1. AlphaGo Zero (Silver et al., 2017, *Nature*) later achieved 100-0 against the original AlphaGo while training entirely from self-play—no human games required.

**Atari games** established that RL scales to high-dimensional sensory inputs. With raw state spaces of approximately (128)^(210×160) possible frames, tabular methods are infeasible. Mnih et al. (2015, *Nature*) showed DQN achieved human-level performance on 29 of 49 games using identical architecture and hyperparameters—a generality impossible with hand-crafted features. Prior RL methods using engineered features (Sarsa-based approaches, contingency-aware agents) required game-specific tuning and performed substantially worse.

**StarCraft II** extends these results to partial observability and real-time multi-agent settings. AlphaStar (Vinyals et al., 2019, *Nature*) confronted up to **10^26 possible actions per timestep** with imperfect information over thousands of sequential decisions. Scripted bots had competed for a decade without approaching professional play; AlphaStar reached Grandmaster rank (top 0.2% of human players) using multi-agent RL with league-based training.

## Ride-sharing dispatch: production deployment at scale

DiDi's order dispatching system provides the strongest evidence of production RL deployment solving an otherwise intractable problem. Qin et al. (2020, *INFORMS Journal on Applied Analytics*) document matching tens of millions of daily trip requests to drivers across hundreds of Chinese cities.

The **state space explodes** because driver fleet coordination is a cooperative multi-agent system: "The number of agents poses a big challenge...the joint action space quickly becomes intractable." With locations discretized via hexagonal grids, time bucketed into periods, and driver availability states tracked, tabular methods suffer immediately from dimensionality. The authors explicitly note: "As the number of features to represent the agent state increases, the table size for the value function quickly becomes intractable."

**Standard heuristics fail** in documented ways. Nearest-driver matching is "most myopic among all alternatives." Combinatorial optimization with batch windows improves pickup distance but "the myopic nature of the solution means there was still room for further optimization." Pickup distance minimization alone leads to ignoring short trips that reduce total platform income.

DiDi deployed **Deep Value Networks with TD(0) learning**, later enhanced with Double DQN. The system uses a Generalized Policy Iteration framework: offline policy evaluation via temporal-difference learning combined with online policy improvement through linear assignment using learned state values. Mean Field Multi-Agent RL (Li et al., 2019, *WWW*) further scaled coordination by approximating other agents' effects through aggregate distributions, avoiding exponential complexity in agent count.

Supporting evidence comes from MOVI (Oda & Joe-Wong, 2018, *IEEE INFOCOM*), which achieved **76% reduction in unserviced requests** and **20% improvement over receding horizon control**—the best model-based baseline—emphasizing "the benefits of a model-free approach."

## Revenue management and real-time bidding: documented deployments

Network revenue management exhibits classic curse of dimensionality: with n products under multinomial logit choice models, there are **2^n possible offer sets** (Liu & van Ryzin, 2008, *M&SOM*). The state space explodes with resources × fare classes × time periods. Talluri (2014, *INFORMS Journal on Computing*) states explicitly: "The state-space of the DC-NRM stochastic dynamic program explodes."

**EMSR heuristics** (expected marginal seat revenue) remain industry standard but have documented limitations. Belobaba (1989, *Operations Research*) showed EMSRa deviates more than **1.5% from optimal** under certain conditions, while EMSRb stays within 0.5%. Bid-price controls derived from deterministic LP show **5-8% optimality gaps** without frequent reoptimization (University of Mannheim studies). More fundamentally, EMSR assumes demand arrives in increasing fare order—violated by modern booking patterns.

Chen et al. (2023, *Journal of Operations Management*) demonstrated **11.80% improvement in RevPAR** (revenue per available room) using a two-step RL approach in a **production deployment at a budget hotel chain**, validated via synthetic control methods. The RL system overcomes "the challenges of characterizing demand and estimating cancellations" that plague traditional approaches.

**Real-time bidding** shows even stronger production evidence. Alibaba's sponsored search platform uses Robust MDP with deep RL (Zhao et al., 2018, *KDD*), deployed in production. Jin et al. (2018, *CIKM*) deployed multi-agent RL for coordinated bidding. Most recently, Guo et al. (2024, *KDD*) report DiffBid achieved **2.81% GMV increase and 3.36% ROI improvement** in online A/B tests on Alibaba's advertising platform. The key challenge: "state space is represented by the auction information and the campaign's real-time parameters" across millions of auctions, making static optimization infeasible.

## Multi-echelon inventory: rigorous academic evidence

Inventory control offers perhaps the cleanest academic documentation. Powell (2011, *Approximate Dynamic Programming*) identifies **three curses of dimensionality**: state space, action space, and outcome space. For multi-echelon networks, state includes inventory at each location plus pipeline quantities across all echelons; state space grows exponentially with lead times as all in-transit inventory must be tracked.

**Base-stock policies** are optimal only for serial systems under restrictive conditions (Clark-Scarf decomposition). Gijsbrechts et al. (2022, *M&SOM*) found capped base-stock policies achieve **3-6% optimality gaps** in lost sales settings. van Hezewijk et al. (2023, *CEJOR*) documented decomposition heuristics producing **11-16% cost gaps** versus optimal in multi-echelon networks.

Deep RL provides scalable solutions:

- **A3C** matches state-of-the-art heuristics with 3-6% optimality gap while requiring no problem-specific structure (Gijsbrechts et al., 2022)
- **PPO** achieves **16.4% cost reduction** versus benchmark heuristics in linear networks, **11.3%** in divergent networks, and **6.6%** in a real manufacturer case (van Hezewijk et al., 2023)
- **DQN with transfer learning** achieves near-optimal performance in 4-echelon supply chains even when other agents behave irrationally (Oroojlooyjadid et al., 2022, *M&SOM*)
- **Deep Controlled Learning** (2024) reduces optimality gaps to **0.2%**—far better than A3C's 3-6%

The roadmap paper by Boute et al. (2022, *EJOR*) synthesizes the field, noting RL is "especially promising when problem-dependent heuristics are lacking."

## Market making and mechanism design: computational economics

Limit order book market making presents state space dimensionality of **ℤ^P** for P price levels (Gould et al., UCLA survey). Multi-asset market making with d bonds creates HJB PDEs where "classical finite difference methods cannot be used in high dimension" (Guéant & Manziuk, 2019, *Applied Mathematical Finance*). Grid methods require O(N^d) computational cost, infeasible for d>3-4 assets.

**Avellaneda-Stoikov** spreads remain the industry heuristic baseline. Falces Marin et al. (2022, *PLOS ONE*) showed Deep Double DQN (Alpha-AS) achieves substantially higher Sharpe and Sortino ratios on 30 days of BTC-USD L2 tick data. For optimal execution, RL-Exec (2025) demonstrates **+23 basis points improvement** over TWAP at 2-hour horizons using PPO, with statistical significance via Wilcoxon signed-rank tests.

**Automated mechanism design** presents perhaps the starkest computational contrast. Dütting et al. (2024, *Journal of the ACM*) document that LP-based optimal auction computation for just 2 bidders × 2 items × 11 value bins requires **~9×10^5 decision variables and ~3.6×10^6 constraints**, taking **62 hours**. RegretNet—a neural network encoding auction rules trained via gradient descent—solves this in minutes while recovering "essentially all known analytical solutions" and discovering novel mechanisms. RegretFormer (Ivanov et al., 2022, *NeurIPS*) extends this with transformer architectures handling variable bidder/item counts.

## Energy systems and healthcare: emerging frontiers

Battery storage arbitrage exhibits curse of dimensionality through SOC × price × time × degradation state combinations. Sage et al. (2024) achieved **60% improvement in accumulated rewards** using DQN with forecasting on Alberta electricity markets. Multi-service battery systems (energy arbitrage + frequency regulation) involve nested multi-timescale MDPs; TDD-ND algorithms achieve **22.8-32.9% higher revenue** than DQL baselines (MDPI *Energies*, 2021).

Building HVAC control compounds continuous state/action spaces across thermal, occupancy, and weather variables. Deep RL (SAC, DDPG, PPO) achieves **17-35% energy savings** versus rule-based controllers in validated studies (*Applied Energy*, 2024), with real-world deployments showing 26.3% savings versus PI controllers.

**Sepsis treatment** offers the most-cited healthcare example. Komorowski et al. (2018, *Nature Medicine*) used SARSA on MIMIC-III/eICU data (17,000+ sepsis patients, 750 states, 25 actions) and found mortality was lowest when clinicians matched AI recommendations. The AI Clinician agreed with or exceeded human decisions **98% of the time**, generally recommending less fluid and more vasopressors. State space calculation for ICU allocation scales as **2^N for N beds**—"ICUs typically have hundreds of beds (N≥100), making computational demands unattainable" (arXiv 2309.08560). Transformer-based Q-network parametrization addresses this while incorporating fairness objectives.

## Synthesis across domains

The evidence reveals consistent patterns validating RL's unique advantages:

| Domain | State Space Scale | Heuristic Gap | RL Improvement | Production Status |
|--------|------------------|---------------|----------------|-------------------|
| Go | 10^170 positions | Amateur-level MCTS | World champion | DeepMind |
| Ride-sharing | Exponential in agents | 20%+ vs. RHC | DiDi deployment | **Production** |
| Revenue mgmt | 2^n offer sets | 5-8% vs. optimal | 11.8% RevPAR gain | Hotel chains |
| RTB/advertising | Millions of auctions | Static rules fail | 2.8% GMV lift | **Alibaba production** |
| Inventory | O(n^d) | 6-16% cost gaps | 6-16% reduction | Manufacturer cases |
| Market making | ℤ^P LOB states | A-S baseline | +23 bps execution | Research/HFT |
| Mechanism design | O(10^6) LP vars | 62-hour compute | Minutes via NN | Research |
| Battery storage | SOC × price × time | Rule-based limits | 22-60% gains | Emerging |
| Sepsis treatment | 2^N ICU beds | Clinical variability | 98% match/exceed | Awaiting trials |

**Three conditions consistently co-occur**: explicit dimensionality calculations demonstrating DP infeasibility, quantified performance gaps from industry-standard heuristics (typically 5-20%), and RL algorithms achieving substantial improvements while scaling to production. The strongest evidence comes from production deployments at DiDi and Alibaba, with academic validation across all domains.

## Conclusion

This survey identifies a coherent class of economic and operations problems where reinforcement learning is not merely helpful but uniquely necessary. The pattern is consistent: state spaces grow exponentially in problem dimensions, closed-form solutions require restrictive assumptions violated in practice, industry heuristics leave measurable value unrealized, and deep RL provides the only demonstrated path to scalable near-optimal policies. For structural econometrics and inverse RL research, these problems offer ideal settings where forward RL is well-understood, heuristic baselines are documented, and performance gaps are quantified—enabling rigorous study of preference recovery and mechanism identification from observed behavior.

---

## Papers to download

**Production deployments with documented improvements:**

- **Ride-sharing dispatch (DiDi)**: 0.5-5% GMV improvement, serves 30M+ daily rides
  - Qin et al. (2021) "Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning" *INFORMS Journal on Applied Analytics*

- **Ride-sharing dispatch (Lyft)**: $30M+/year incremental revenue, +0.96% fulfillment rate
  - Han et al. (2022) "A Better Match for Drivers and Riders: Reinforcement Learning at Lyft" *KDD*

- **Real-time bidding (Alibaba)**: +2.81% GMV, +3.36% ROI in A/B tests
  - Guo et al. (2024) "DiffBid" *KDD*
  - Cai et al. (2017) "Real-Time Bidding by Reinforcement Learning in Display Advertising" *WSDM*

- **Hotel revenue management**: +11.8% RevPAR in field experiment
  - Chen et al. (2023) "A reinforcement learning approach for hotel revenue management with evidence from field experiments" *Journal of Operations Management*

**Strong academic evidence (approaching production):**

- **Multi-echelon inventory**: 6-16% cost reduction vs. decomposition heuristics
  - Gijsbrechts et al. (2022) "Can Deep Reinforcement Learning Improve Inventory Management?" *M&SOM*
  - van Hezewijk et al. (2023) "Multi-echelon inventory optimization using deep reinforcement learning" *CEJOR*

- **Optimal execution / market making**: +23 bps vs. TWAP
  - Ning et al. (2021) "Double Deep Q-Learning for Optimal Execution" *Applied Mathematical Finance*

- **Battery storage arbitrage**: 22-60% revenue improvement vs. rule-based
  - Jiang & Powell (2015) "Optimal Scheduling of Energy Storage" *INFORMS JOC*

- **Automated mechanism design**: Minutes vs. 62 hours for LP; recovers known optimal auctions
  - Dütting et al. (2024) "Optimal Auctions through Deep Learning" *Journal of the ACM*

**The pattern**: RL wins where (a) decisions are sequential with delayed rewards, (b) state space is too large for exact DP, (c) environment is non-stationary so heuristics can't be pre-tuned. DiDi and Alibaba RTB are the cleanest "killer examples" for the survey—both have production A/B tests and explicit statements about why DP/heuristics failed.
