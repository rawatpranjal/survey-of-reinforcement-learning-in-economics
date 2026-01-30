# Heuristic Decision Rules in Dynamic Optimization: When RL Beats Rules of Thumb

Simple heuristics dominate real-world sequential decision-making across numerous industries—not because practitioners are naive, but because computing optimal policies is computationally intractable. This survey documents **eight major domains** where the curse of dimensionality forces practitioners to use rules of thumb, quantifies the performance gaps, and synthesizes evidence on where reinforcement learning and approximate dynamic programming have demonstrated improvements. The documented "cost of heuristics" ranges from **0.1% to 20%** depending on problem complexity, with aggregate economic stakes in the billions of dollars.

## The curse of dimensionality explains heuristic prevalence

Richard Bellman coined "curse of dimensionality" when analyzing why dynamic programming becomes intractable as problem dimensions grow. For a state space with *d* dimensions and *n* discretization points per dimension, computational cost grows as **O(n^d)**—exponentially in problem dimensionality. This fundamental barrier forces practitioners across industries to adopt simple decision rules that approximate optimal behavior without requiring full state-space enumeration.

Warren Powell's work at Princeton identifies **three distinct curses**: the state space curse (exponential growth in states), the outcome space curse (combinatorial explosion in uncertainty realizations), and the action space curse (continuous or high-dimensional action sets). Modern business problems typically suffer from all three simultaneously.

The domains studied here share common structural features: they can be formulated as Markov Decision Processes, exact solution via backward induction is infeasible, practitioners have converged on interpretable heuristics, and recent advances in RL/ADP have demonstrated measurable improvements over these rules.

---

## Inventory management: (s,S) policies nearly optimal for single products

The **(s,S) inventory policy**—ordering up to level *S* whenever inventory falls to or below reorder point *s*—was proven optimal by Herbert Scarf in 1960 for single-product systems with fixed ordering costs. This theoretical foundation explains its ubiquity in enterprise resource planning systems worldwide.

**Why exact DP fails for multi-product systems**: With *N* products each taking *M* possible inventory levels, the state space grows as M^N. A "simple" 100-product system with 100 inventory levels per product yields **100^100 states**—astronomically beyond computational feasibility. Multi-echelon systems compound this by requiring tracking of pipeline inventory across supply chain stages.

**Performance gaps are surprisingly small for well-tuned heuristics**: Studies consistently find single-stage (s,S) heuristics operate within **0.01-2% of optimal** when properly parameterized. Zhu et al. (2021) report modified (s,S) policies achieve gaps of just 0.05% in over 50% of test cases. However, multi-echelon problems show larger gaps—approximately **5% above cost lower bounds** for modified echelon (r,Q) policies.

**RL has matched and exceeded heuristics in complex settings**: Gijsbrechts, Boute, Van Mieghem, and Zhang (2022) in *Manufacturing & Service Operations Management* found that Asynchronous Advantage Actor-Critic (A3C) algorithms **match state-of-the-art heuristics** with 3-6% optimality gaps in lost-sales problems. More impressively, Deep Controlled Learning (DCL) by Van Jaarsveld et al. achieves gaps of **at most 0.2%**—a significant improvement over both A3C and traditional heuristics for stochastic problems with lost sales, perishability, and random lead times.

**Economic stakes are substantial**: Procter & Gamble reported **$1.5 billion in savings** in 2009 from implementing multi-echelon inventory optimization, shifting from single-echelon heuristics to network-wide optimization. Industry estimates suggest MEIO can reduce inventory by up to **30%** while improving stock availability by **5%**.

---

## Retail pricing reveals large gaps between markup rules and dynamic optimization

Cost-plus pricing—adding a fixed percentage markup to acquisition costs—remains the dominant approach in retail despite decades of dynamic pricing research. Standard retail markups range from **30-50%** for general merchandise to **100% (keystone pricing)** for specialty retailers.

**The computational barrier**: Retailers managing **10,000 to 1,000,000+ SKUs** face impossible optimization problems. Demand depends on prices of substitutes and complements, competitor reactions are unobservable, and demand functions must be estimated from limited data. Koch and Klein (2020) in the *European Journal of Operational Research* explicitly note that "even small instances cannot be solved to optimality due to the curses of dimensionality."

**Revenue lift from dynamic pricing is well-documented**: Talluri and van Ryzin estimate **3-7% revenue increases** and **2-5% profitability improvements** from dynamic pricing. The widely-cited McKinsey finding that "a 1% improvement in price yields an 11.1% improvement in operating profit" underscores the leverage effect. Amazon's dynamic pricing—with approximately **2.5 million daily price updates**—reportedly contributed to a **25% profit increase**.

**Contextual bandits show strong results**: Recent work on contextual bandits for markdown pricing shows substantial improvements. A 2024 ACM CODS-COMAD paper demonstrated **17.24% increase in sales units** and **6.14% margin improvement** using Vowpal Wabbit online cover solutions compared to non-cumulative and rule-based approaches. Deep Q-Learning approaches documented in arXiv (2024) show RL "outperforms traditional models by adapting pricing strategies based on ongoing interactions."

---

## Workforce planning uses threshold rules despite known suboptimality

Hiring and firing decisions in practice follow threshold-based rules: firms hire when productivity or demand exceeds upper thresholds and fire when falling below lower thresholds. The **Booth-Chen-Zoega model** (2002, *Journal of Labor Economics*) formalized this two-threshold structure, showing it emerges optimally under demand uncertainty with adjustment costs.

For call center staffing, the **square-root staffing rule** (s = R + β√R, where R is offered load) provides a simple capacity calculation that performs remarkably well. The SIPP (Stationary, Independent, Period-by-Period) approach treats each time period independently using steady-state queueing formulas—ignoring temporal dependencies but maintaining computational tractability.

**Performance evidence**: Atlason, Epelman, and Henderson (2007, *Management Science*) found their Sample Average Approximation with Cutting Plane Method (SACCPM) outperformed SIPP heuristics in all 16 test cases with shift constraints, producing solutions **0.3-4.6% less costly**. Healthcare scheduling studies show metaheuristics achieve solutions within **0.2% of optimal** for nurse scheduling while heuristic rounding increases costs by approximately **1%**.

**RL applications are emerging**: A 2024 arXiv paper on multi-agent RL for operating room scheduling demonstrated PPO-trained policies that **outperform 6 rule-based heuristics across 7 metrics**. Hospital resource allocation studies (Bushaj et al., 2023) show DRL agents outperform historical allocation strategies while handling legal and economic constraints.

---

## Portfolio rebalancing demonstrates tractable near-optimal linear rules

Investment managers commonly use **calendar-based rebalancing** (monthly, quarterly, annually) or **threshold-based rebalancing** (rebalancing when any asset deviates from target by ±5-10%). Robo-advisors like Betterment use 2-3% drift triggers; Wealthfront uses 4-10% thresholds depending on account type.

**Computational limits are well-quantified**: Academia and Springer studies confirm dynamic programming is limited to a **maximum of five assets** due to the curse of dimensionality. For 10-asset portfolios, over 14 billion calculations would be required. Transaction costs create complex no-trade regions with irregular geometry, and tax considerations (tracking tax lots, holding periods) further explode state dimensions.

**The Moallemi-Saglam linear rebalancing breakthrough**: Research published in the *Journal of Financial and Quantitative Analysis* by Moallemi and Saglam demonstrates that **linear rebalancing rules achieve within 5% of optimal**—where trades are affine functions of return-predicting factors. This computationally tractable approach (convex optimization) achieves up to **18% improvement** over projected Linear-Quadratic Control and up to **72% improvement** in mean-variance settings.

**Vanguard's empirical findings**: Research from Vanguard (2022, 2024) found threshold-based rebalancing generates **15-22 basis points higher annual returns** versus monthly calendar rebalancing, and **5-8 basis points** versus quarterly rebalancing. Their "Rational Rebalancing" paper establishes that annual rebalancing with 1% threshold triggers is most efficient for typical portfolios.

**Deep RL shows promise but with caveats**: Multiple papers (ScienceDirect 2024, MDPI Algorithms 2024) demonstrate DQN and DDPG agents that consistently surpass S&P 500 benchmarks. However, as noted in the MDPI study, "continuous rebalancing leads to higher transaction costs and slippage, making periodic rebalancing a more efficient approach"—validating the intuition behind traditional heuristics.

---

## Airline revenue management: EMSR heuristics remain remarkably effective

Expected Marginal Seat Revenue (EMSR) heuristics, developed by Peter Belobaba at MIT in 1987, remain the industry standard for airline seat inventory control. **EMSR-b is consistently within 0.5% of optimal** for single-leg problems, while EMSR-a deviates up to 1.5% under certain conditions.

**Network problems expose heuristic limitations**: The Deterministic Linear Program (DLP) with frequent re-solving generates approximately **1-2% incremental revenue** over earlier leg-based techniques. However, Farias and Van Roy (2007) show that ADP with separable concave approximations can achieve **up to 8% improvement over DLP** for Markov-modulated demand arrivals. The gap widens substantially when customer choice behavior (buy-up, buy-down) enters the model.

**Industry economic impact**: American Airlines estimated benefits of **$1.4 billion over three years** from their revenue management system (Smith, Leimkuhler & Darrow, 1992, winning the Franz Edelman Award). Industry-wide estimates suggest effective RM systems generate **5-7% incremental revenue**. Modern vendors claim "up to 20% revenue increase in the first year."

**Deep RL is achieving near-optimal performance**: Shihab and Wei (2021) developed Deep RL that earns **more revenue than EMSR-b** and achieves load factors near 100%. Bondoux et al. (2020) at Amadeus demonstrated RL systems that "achieve better revenue performance than state-of-the-art heuristic methods" without requiring separate demand forecasters. A key advantage: RL addresses the earning-while-learning problem inherent in revenue management.

---

## Ride-sharing dispatch shows production-grade RL improvements

**Greedy nearest-driver matching** was the original dispatch approach—simple, fast, and intuitive but entirely myopic. Modern platforms use **batch matching** with combinatorial optimization (Hungarian algorithm) over 2-10 second windows, which enables "global" optimization within each batch but still ignores future demand patterns.

**Why optimal dispatch is intractable**: DiDi's research (Qin et al., *Interfaces* 2021) explicitly states: "The number of agents poses a big challenge for solving such a multiagent problem because the joint action space quickly becomes intractable." With thousands of drivers across hundreds of spatial cells and 96 time buckets daily, the state space grows combinatorially.

**Production RL deployments show consistent gains**:
- **DiDi**: CVNet (Cerebellar Value Network) achieved **0.5-5% improvement in Global GMV** across all cities in China. A/B tests showed **>1.3% improvement in driver income**, with full deployment showing up to **5.3% improvement** via causal inference analysis.
- **Lyft**: Online RL dispatch deployed globally in 2021 generates **>$30 million per year** in incremental revenue and serves **millions of additional riders annually**. Han et al. (KDD 2022) report **+0.96% request fulfillment rate** and **+0.73% profit per passenger session**.

The key insight: RL doesn't replace batch matching but enhances edge weights with learned long-term values, maintaining combinatorial optimization for policy generation while achieving farsighted decision-making.

---

## Energy markets benefit from ADP algorithms near theoretical optimum

Electricity market participants commonly use **cost-based bidding** (bidding at or near marginal cost) or **markup-based strategies** with fixed or utilization-dependent markups. These heuristics are theoretically optimal in perfect competition but fail to capture strategic value in oligopolistic wholesale markets.

**The three curses manifest clearly**: State variables include battery state-of-charge, price history, demand levels, renewable output, and competitor positions. Electricity prices exhibit extreme volatility with irregular spikes and highly non-stationary behavior. Renewable intermittency creates "undispatchable" generation requiring real-time balancing with high penalty costs for deviations.

**ADP algorithms achieve near-optimality**: Jiang and Powell (2015, *INFORMS Journal on Computing*) developed convergent ADP exploiting value function monotonicity, achieving policies within **0.08% of optimal** on deterministic models and **0.86% on stochastic models**. Salas and Powell (2017) extended this to heterogeneous storage portfolios, achieving **1.34% gaps** while scaling to 5+ devices.

**RL improvements are substantial in newer studies**:
- Belgian imbalance market: Distributional Soft Actor-Critic (DSAC) achieved **53.1% higher daily profit** versus DQN
- Multi-agent VPP bidding: MATD3 showed **65% reward improvement** over MADDPG and MAPPO
- Battery-transformer hybrid: **29% profit increase** from joint day-ahead and real-time market participation
- High-frequency trading: Strategies earning **58% more than hourly re-optimization**

---

## Real-time bidding demonstrates RL scalability at billions of decisions

**Linear bidding** (bid proportional to predicted CTR) is the most widely-used RTB heuristic, simple to implement and stable in operation. Constant bidding and MCPC (max cost-per-click) rules serve as common baselines. Budget pacing via probabilistic throttling—randomly sampling which auctions to enter—handles budget constraints without optimization.

**Scale creates the computational barrier**: DSPs process **millions of bid requests per second** with latency requirements under 100 milliseconds. User features are high-dimensional (device, location, time, demographics, behavior), campaign state includes remaining budget and performance metrics, and the competitive landscape shifts continuously. Cai et al. (WSDM 2017) note the state transition probability matrix is "difficult to represent due to huge computational cost."

**RL has been successfully deployed at scale**:
- **Alibaba's DiffBid** (KDD 2024): Online A/B tests showed **+3.36% ROI**, **+2.81% GMV**, and **+2.09% conversions**
- **Zhang et al. ORTB** (KDD 2014): Optimal RTB with nonlinear bid functions "generated highest clicks while maintaining comparable impressions"
- **Google Smart Bidding**: Over **80% of Google advertisers** now use automated bidding (2021)
- **E-commerce MAB study** (OARS-KDD 2021): Batch-updated multi-armed bandits achieved **16.1% relative CVR increase** versus defaults

Industry claims suggest AI-powered bidding can reduce cost-per-acquisition by **up to 30%**, though published academic verification of such large effects remains limited.

---

## Synthesis: patterns across domains

Several patterns emerge across these eight domains that inform both practice and research:

**Heuristic performance correlates with problem structure**: Single-stage problems with well-understood cost structures (inventory, single-leg airline RM) show remarkably small gaps—often under 1%. Multi-stage problems with network effects, customer choice, or strategic interactions show larger gaps of 2-10%, creating greater opportunity for RL improvements.

**State-of-the-art RL achieves 1-5% typical improvements**: Across domains, well-implemented RL systems consistently show low single-digit percentage improvements over optimized heuristics in production settings. Larger improvements (10-50%+) appear in academic simulations but may not translate directly to production.

**Linear approximations offer surprising power**: Moallemi-Saglam's linear rebalancing rules and DiDi's value-weighted batch matching both show that linear approximations to complex value functions can capture most available gains while maintaining computational tractability.

**The "cost of heuristics" depends on environmental stability**: When market conditions, demand patterns, and competitor behavior are stable, well-tuned heuristics perform nearly optimally. RL shows greatest advantage when environments are non-stationary, as explicitly noted in RTB research.

---

## Key papers for academic citation

| Domain | Foundational Paper | Key RL/ADP Paper |
|--------|-------------------|------------------|
| Inventory | Scarf (1960) optimality proof | Gijsbrechts et al. (2022) M&SOM |
| Retail Pricing | Talluri & van Ryzin (2004) | ACM CODS-COMAD (2024) bandits |
| Workforce | Booth, Chen & Zoega (2002) | Atlason et al. (2007) Management Science |
| Portfolio | Merton (1971); Constantinides (1986) | Moallemi & Saglam (JFQA) |
| Airline RM | Belobaba (1987); Gallego & van Ryzin (1994) | Farias & Van Roy (2007) |
| Ride-sharing | Özkan & Ward (2020) | Qin et al. (2021) Interfaces |
| Energy | Cramton (2000); Powell ADP text | Jiang & Powell (2015) INFORMS JOC |
| RTB | Zhang et al. (2014) KDD | Cai et al. (2017) WSDM |

For structural econometrics and inverse reinforcement learning research, these domains provide rich settings where: (1) decision rules are observable, (2) outcome data is abundant, (3) heuristic policies create identifiable deviations from optimality, and (4) revealed preferences through observed decisions can be used to infer underlying cost structures and constraints. The documented performance gaps between heuristics and optimal policies offer natural "behavioral wedges" for identification strategies in IRL estimation.
