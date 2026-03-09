# Reinforcement Learning in Economics and Operations Research: A Survey (2018-2025)

Deep reinforcement learning has emerged as a powerful complement to classical optimization methods in economics and operations research, with over **100 methodological papers** and **significant industrial deployments** at firms including Amazon, Alibaba, DiDi, and major financial institutions. The field has matured from theoretical exercises to production systems, while simultaneously developing formal bridges to structural econometrics—most notably through the equivalence between **Maximum Entropy Inverse RL and dynamic discrete choice models** (Ermon et al., 2015). This survey identifies the key contributions, compares RL approaches with classical methods, and highlights when each dominates.

---

## Structural econometrics meets machine learning

The theoretical foundation connecting reinforcement learning and structural econometrics centers on the **Rust (1987) bus engine replacement problem**—the canonical dynamic discrete choice model that frames equipment replacement as a Markov Decision Process with state-dependent costs. Iskhakov, Rust & Schjerning (2020) in *The Econometrics Journal* provide the authoritative synthesis: structural econometrics evolved from McFadden's (1974) static discrete choice to dynamic models through Wolpin (1984), Rust (1987), and Hotz & Miller (1993), and machine learning now extends these methods to previously intractable settings.

**The critical theoretical result** comes from Ermon et al. (2015, AAAI): for finite-horizon deterministic MDPs with discount factor η=1, Maximum Entropy IRL (Ziebart et al., 2008) is *mathematically equivalent* to logit Dynamic Discrete Choice. This means algorithms transfer directly between communities. Sharma, Kitani & Groeger (2017) extend this by showing inverse RL can be reformulated using **Conditional Choice Probabilities** (Hotz & Miller, 1993), avoiding repeated dynamic programming calls and reducing computational costs by orders of magnitude.

Igami (2020) makes the connection vivid: **Deep Blue** (chess) corresponds to a calibrated value function, **Bonanza** (shogi) to estimated value functions via Rust's NFXP algorithm, and **AlphaGo's policy network** to CCP estimation. Kang et al. (2025) recently propose an ERM-based framework achieving global convergence for DDC/MaxEnt-IRL without explicit transition probability estimation—enabling neural networks for infinite state spaces while maintaining econometric interpretability.

---

## Inventory management: from lost sales to Amazon deployment

RL for inventory management has achieved the strongest industrial validation of any domain. **Gijsbrechts, Boute, Van Mieghem & Zhang (2022)** in *Manufacturing & Service Operations Management* establish that Asynchronous Advantage Actor-Critic (A3C) matches state-of-the-art heuristics for lost-sales, dual-sourcing, and multi-echelon problems, achieving **3-6% optimality gaps**. Their key insight: DRL provides value when problem-dependent heuristics are unavailable, while initial tuning costs amortize across similar problems.

The **most significant deployment** comes from Amazon. Madeka et al. (2022) describe DirectBackprop—a model-based RL approach using differentiable simulators—deployed on **10,000 products over 26 weeks**. Results show **~12% inventory reduction** with no statistically significant revenue loss, outperforming model-free RL and newsvendor baselines. The system handles lost sales, correlated demand, price matching, and random lead times that classical methods struggle with.

| Problem Type | Best Algorithm | Key Paper | Performance vs. Classical |
|--------------|----------------|-----------|---------------------------|
| Lost sales | A3C, DCL | Gijsbrechts et al. (2022) | Matches heuristics at 3-6% gap |
| Multi-echelon | HAPPO | Liu et al. (2024, POMS) | Reduces bullwhip effect |
| Beer Game | Modified DQN | Oroojlooyjadid et al. (2022, M&SOM) | Near-optimal; robust to irrational partners |
| Joint pricing-inventory | PPO, Double DQN | Expert Systems (2022-2023) | Captures reference price effects (~8% profit gain) |
| Perishable | SAC | Yavuz & Kaya (2024) | 4.6% better, 56% faster than DQL |

**Classical methods remain competitive** when problem structure is well-understood (base-stock optimal), demand distributions are stationary and known, and interpretability matters. Deep Controlled Learning (DCL, 2025, EJOR) achieves **≤0.2% optimality gaps** using approximate policy iteration with sequential halving—a hybrid leveraging both RL flexibility and structural insights.

---

## Dynamic pricing: from airlines to algorithmic collusion

Revenue management applications span airlines, hotels, ride-sharing, and e-commerce, with theoretical contributions on algorithmic collusion raising regulatory concerns. **Bondoux, Nguyen, Fiig & Acuna-Agost (2020)** in *Journal of Revenue and Pricing Management* demonstrate RL discovering pricing policies superior to human-designed heuristics *without explicit demand forecasting*—a paradigm shift from traditional revenue management systems requiring demand model specification.

**Ride-sharing** represents the most mature deployment domain. Lei & Ukkusuri (2023) in *Transportation Research Part B* develop TD3-based pricing that scales to real-world systems through offline learning with online deployment, proving existence of deterministic stationary optimal policies under historical versus current price distinctions. DiDi's order dispatching system (documented in *INFORMS Interfaces*) processes **millions of daily orders** using semi-MDP formulations with deep RL.

The **algorithmic collusion literature** carries significant policy implications. Calvano, Calzolari, Denicolò & Pastorello (2020) in the *American Economic Review* show independent Q-learning agents in Bertrand oligopoly converge to **supra-competitive prices through implicit punishment strategies**—without explicit communication. Kastius & Schlosser (2022) confirm with DQN and SAC that RL algorithms can be "forced into collusion" by competitors. Hansen, Misra & Pai (2021) in *Marketing Science* extend this to marketing applications, while recent work (2024-2025) demonstrates single human defectors can drive prices toward competitive levels.

**E-commerce deployment** at Alibaba (Liu et al., 2019) uses DQN and DDPG with a novel reward function—Difference of Revenue Conversion Rates—rather than direct revenue, pre-training from historical data to address cold-start. Amazon changes prices every 15 minutes for **562+ million products** using algorithmic systems, though specific RL architectures are proprietary.

The connection to structural estimation runs through **inverse RL**. Halperin (2017) uses Maximum Entropy IRL to learn customer preferences from observed behavior, providing an alternative to structural models of forward-looking utility-maximizing agents. Besbes & Zeevi (2015, *Management Science*) show simple parametric models achieve near-optimal performance even when misspecified—the "price of misspecification" may be smaller than expected, suggesting hybrid approaches combining structural intuition with RL flexibility.

---

## Portfolio optimization and algorithmic trading

Finance applications emphasize transaction costs, risk management, and the challenge of backtesting in non-stationary markets. **Deep hedging** (Buehler et al., 2019, *Quantitative Finance*) established the paradigm: Monte Carlo Policy Gradient agents outperform Black-Scholes delta hedging under transaction costs. Cao, Chen, Hull & Poulos (2021) extend this with two Q-functions tracking cost and squared cost for mean-variance objectives, finding hybrid approaches using simple valuation models work best.

**Portfolio allocation** research emphasizes risk-adjusted objectives. Jiang, Olmo & Atwi (2024, *Global Finance Journal*) achieve superior performance on DJIA and S&P100 using TD3 with extended Markowitz mean-variance rewards embedding transaction costs. The ART-DRL framework (2025) demonstrates **Sharpe ratio of 4.340** on commodity futures through dynamic agent switching based on rolling performance—DQN for high volatility, PPO for trending markets.

| Application | Algorithm | Key Innovation | Benchmark Comparison |
|-------------|-----------|----------------|----------------------|
| Deep hedging | MCPG, DDPG | Mean-variance cost minimization | Outperforms BS delta with transaction costs |
| Portfolio allocation | TD3, PPO | Extended Markowitz reward | Superior to mean-variance optimization |
| Order execution | DQN, PPO-LSTM | LOB feature extraction | Beats TWAP/VWAP by 3+ basis points |
| Market making | SAC, C-PPO | Attn-LOB for state representation | Outperforms Avellaneda-Stoikov |
| HFT | PPO | Multi-timescale DRL | Sharpe 3.42, 33% improvement |

**Order execution** at scale uses PPO-LSTM for generalization across **50 stocks and 165-380 minute horizons** on Korea Stock Exchange, outperforming market VWAP by **3.282 basis points** across 12,300 executions. Market making research (Gašperov & Kostanjčar, 2022) shows SAC achieves superior risk-reward even under significant transaction costs using Hawkes process LOB simulators.

**Inverse RL applications** in finance include adversarial IRL for automated reward acquisition in market making (ACM ICAIF, 2024) and Transaction-aware IRL (2023, *Applied Intelligence*) introducing a "wait" action addressing reward bias and varying-length transactions. Yang et al. (2018) use Gaussian Process IRL to extract trading signals from sentiment-market interactions.

---

## Auction design and programmatic advertising

Real-time bidding (RTB) represents the highest-volume RL deployment domain, with systems serving **hundreds of billions of bid requests daily**. Zhou et al. (2021, KDD) describe VerizonMedia's deep distribution network for bid shading achieving **+2.4% ROI for CPM/CPC and +8.6% for CPA campaigns**. Alibaba's USCB system (He et al., 2021, KDD) derives unified optimal bidding functions for constrained optimization with RL-based parameter adjustment.

**Mechanism design** has been transformed by neural network approaches. Dütting, Feng, Narasimhan, Parkes & Ravindranath (2019, ICML; 2023, *JACM*) introduce RegretNet and RochetNet—architectures encoding incentive compatibility constraints—that recover known analytical solutions while discovering novel mechanisms for multi-item settings. Neural Auction (KDD 2021) deployed at Alibaba/Taobao jointly optimizes user experience, advertiser utility, and platform revenue, outperforming GSP and VCG in online A/B tests.

Multi-agent RL addresses competitive dynamics directly. Jin et al. (2018, CIKM) model RTB as a multi-agent stochastic game where advertisers learn strategic bidding considering competitor behavior. Understanding Iterative Combinatorial Auction Designs (EC 2024) applies MARL to spectrum auctions with asymmetric bidders, identifying substantially different outcomes from rule modifications that traditional analysis misses.

**Industry deployment** spans all major platforms:
- **Meta**: Lattice system with RL for ad auctions; Variance Reduction System for fairness
- **Google**: Smart Bidding optimizing bids based on conversion likelihood; Performance Max campaigns
- **Alibaba**: Deep GSP Auctions, Neural Auctions, SS-RTB on e-commerce search

---

## Predictive maintenance and the Rust problem revisited

Condition-based maintenance represents the domain with strongest structural econometrics connections. The MDP formulation mirrors Rust (1987) exactly: state = equipment condition, action = repair/replace, reward = negative costs. The computational advantage of RL comes from handling **multi-component systems** where classical dynamic programming faces curse of dimensionality.

Zhang & Si (2020, *Reliability Engineering & System Safety*) demonstrate customized DQN directly mapping degradation to decisions—avoiding threshold-based policies—for multi-component systems under dependent competing risks. Zhou et al. (2022) develop Hierarchical Coordinated RL (HCRL) for large-scale systems, outperforming deep RL benchmarks through agent coordination based on structural importance measures. Fleet maintenance optimization (Yacout et al., 2018, *Journal of Intelligent Manufacturing*) achieves **36.44% improvement** in downtime versus local optimization for military truck fleets.

**Offshore wind turbines** (Cheng et al., 2023, *Ocean Engineering*) and **rail networks** (Mohammadi & He, 2022) represent growing application domains where PPO and DQN optimize dynamic inspection intervals and adaptive repair thresholds. The common finding: RL policies learn "group maintenance" and "opportunistic maintenance" strategies without explicit programming—emergent coordination that mirrors economic principles of economies of scope.

The research gap remains **counterfactual analysis**. While RL excels at prediction and optimization, structural estimation provides the framework for policy analysis under interventions. Hybrid approaches combining offline data with model structure (Kang et al., 2025) represent the frontier.

---

## Other economic applications demonstrate breadth

**Energy markets** use DRL for microgrid management with SAC controlling millions of distributed assets, achieving parity with linear optimization at **<0.1% simulation time** (ScienceDirect, 2022). Graph Neural Networks address scalability for NP-hard Optimal Power Flow problems in distribution networks.

**Healthcare** applications span dynamic treatment regimes for sepsis, glycemic control, and chemotherapy dosing (Yu et al., 2020, *ACM Computing Surveys*). Hospital operating room scheduling (2024) uses cooperative Markov games with PPO, outperforming six rule-based heuristics. The REINFORCE trial demonstrates real-world deployment for type 2 diabetes treatment adherence.

**Labor markets** reveal RL's capacity for bounded rationality modeling. Chen & Zhang (2025, *MDPI Economies*) show DRL in Diamond-Mortensen-Pissarides search models produces **higher volatility than log-linearized DSGE**, better matching the empirical Shimer puzzle while preserving Beveridge curve relationships.

**Monetary policy** research at central banks (Deutsche Bundesbank, Bank of England) applies DDPG to derive optimal reaction functions without assuming Taylor Rule functional forms. Hinterlang & Tänzer (2021) find RL-derived coefficients closer to Fed historical behavior than estimated Taylor Rules. Surprisingly, Reinforcement Learning for Monetary Policy Under Macroeconomic Uncertainty (2025) finds **standard tabular Q-learning outperforms sophisticated deep RL**—simpler approaches may be more robust for macroeconomic control.

---

## Synthesis: when RL outperforms classical methods

Across domains, RL demonstrates consistent advantages under specific conditions while classical methods retain value for interpretability and theoretical guarantees.

**RL dominates when:**
- High-dimensional state spaces make DP intractable (multi-echelon inventory, portfolio allocation)
- Environment models are unknown or non-stationary (e-commerce pricing, real-time bidding)
- Strategic interactions create game-theoretic complexity (competitive pricing, auction design)
- Sequential decisions with delayed rewards require end-to-end optimization (healthcare treatment, maintenance)

**Classical methods remain preferred when:**
- Problem structure is well-understood with known optimal policies (base-stock, mean-variance)
- Interpretability and causal analysis are required (counterfactual policy evaluation)
- Sample efficiency matters and data is limited (traditional structural estimation)
- Theoretical guarantees on optimality are necessary (mechanism design properties)

The most promising research direction synthesizes both approaches: using structural insights to design neural architectures (Structure-Informed Policy Networks), combining IRL with CCP estimation for computational efficiency, and embedding economic constraints directly in RL objectives. The equivalence between MaxEnt-IRL and dynamic discrete choice provides the theoretical foundation; industrial deployments at Amazon, Alibaba, and DiDi demonstrate practical feasibility; and ongoing work on explainability and counterfactual analysis addresses the remaining gaps for adoption in high-stakes economic policy applications.

---

## Conclusion

This survey documents RL's transformation from theoretical curiosity to operational tool across economics and operations research. **Key theoretical advances** include the IRL-DDC equivalence enabling algorithm transfer between communities, CCP-based methods avoiding nested optimization, and multi-agent frameworks capturing strategic interactions. **Key practical advances** include Amazon's inventory system achieving 12% reduction, VerizonMedia's bid shading serving billions daily, and DiDi's dispatch system handling millions of orders.

The research agenda going forward centers on three challenges: (1) developing **hybrid methods** combining structural interpretability with RL flexibility, (2) establishing **counterfactual validity** for policy analysis beyond prediction, and (3) addressing **deployment gaps** between simulation and production. For researchers bridging structural econometrics and machine learning, the theoretical foundations are now solid—the work ahead lies in demonstrating when each paradigm's assumptions hold and how their complementary strengths can be combined.
