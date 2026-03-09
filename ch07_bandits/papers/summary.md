# Bandit Pricing: 10 Papers for Economists

This is a rank-ordered reading list for economists unfamiliar with the bandit pricing literature. The ordering tells a story: theoretical impossibility results first, then the classical baselines that the field builds on, then modern results that sharpen or overturn those baselines, then papers with pure economic content, and finally the strongest real-world deployments.

### 1. Kleinberg & Leighton, "The Value of Knowing a Demand Curve" — FOCS 2003

**Why it matters:** This paper establishes the fundamental limits on how much revenue a seller must sacrifice while learning an unknown demand curve. Every regret bound in the dynamic pricing literature traces back to these results.

**Setting:** Online posted-price auctions over a continuous price space [0,1]. Customers arrive sequentially; the seller observes only buy/no-buy. Three regimes: identical valuations, i.i.d. random valuations, adversarial valuations.

**Key result:** Identical valuations: matching O(log log n) / Omega(log log n). Random valuations: O(sqrt(n) log n) / Omega(sqrt(n)). Adversarial: O(n^{2/3} (log n)^{1/3}) / Omega(n^{2/3}). The random-valuation lower bound is an exponential separation from finite-arm bandits, showing that continuous pricing is qualitatively harder.

**Real-world evidence:** None.

### 2. Broder & Rusmevichientong, "Dynamic Pricing Under a General Parametric Choice Model" — Operations Research, 2012

**Why it matters:** The Operations Research baseline for parametric demand learning. Establishes that under standard econometric demand models (logit, linear, exponential), the regret cost of demand uncertainty is Theta(sqrt(T)) in general, but drops to Theta(log T) when demand curves are "well-separated" (no uninformative prices). Cited 24 times across the corpus.

**Setting:** Monopolist prices to T customers under a general parametric choice model (encompasses logit, linear, exponential demand families). Parameters unknown; MLE-based pricing policy. "Well-separated" condition precludes uninformative prices where all demand curves agree regardless of the parameter.

**Key result:** General case: Omega(sqrt(T)) lower bound and O(sqrt(T)) upper bound via MLE policy. Well-separated case: Omega(log T) lower bound (Cramer-Rao argument) and O(log T) upper bound via greedy MLE. The separation arises because well-separated demand curves permit simultaneous exploration and exploitation at every price.

**Real-world evidence:** None (numerical experiments only).

### 3. Javanmard & Nazerzadeh, "Dynamic Pricing in High-Dimensions" — JMLR, 2019

**Why it matters:** The standard contextual pricing baseline that all high-dimensional and feature-based pricing papers improve on. Shows that sparsity structure in product features can reduce regret from polynomial in d to logarithmic, even when d exceeds T. Cited 28 times across the corpus, the single most-cited paper.

**Setting:** Products described by d-dimensional feature vectors; linear valuation v_t = theta_0 . x_t + alpha_0 + z_t with log-concave noise z_t. Parameter theta_0 has s_0 nonzero entries (sparse). Seller observes only binary buy/no-buy signals. Regularized Maximum Likelihood Pricing (RMLP) runs in episodes of doubling length.

**Key result:** RMLP achieves O(s_0 log d . log T) regret. Lower bound: Omega(s_0 (log d + log T)), so the upper bound is tight up to a log factor. The key insight is that L1-regularized MLE exploits sparsity to learn the demand model from far fewer observations than the ambient dimension suggests.

**Real-world evidence:** None (motivated by Airbnb pricing with hundreds of property features, online ad pricing).

### 4. Misra, Schwartz & Abernethy, "Dynamic Online Pricing with Incomplete Information Using Multi-Armed Bandit Experiments" — Marketing Science, 2019

**Why it matters:** The canonical economics-meets-bandits paper. Demonstrates that imposing the Weak Axiom of Revealed Preference (WARP) on demand learning dramatically reduces exploration costs, and validates the approach in a large-scale field experiment.

**Setting:** K discrete prices, unknown downward-sloping demand, UCB-PI algorithm with WARP partial identification. "Relevant learning" concentrates exploration at profit-relevant prices rather than exploring the full demand curve.

**Key result:** Asymptotically optimal for any weakly downward-sloping demand. Monte Carlo: 95% of optimum vs 66% for balanced experiments. ZipRecruiter field experiment: 43% profit lift over balanced baseline across ~8,000 consumers, 10 price points, deployed at ~2M prices/min.

**Real-world evidence:** ZipRecruiter.com production deployment.

### 5. Xu & Wang, "Logarithmic Regret in Feature-based Dynamic Pricing" — JMLR, 2021

**Why it matters:** Proves a sharp threshold on the value of market knowledge: if the seller knows the noise distribution, regret is O(d log T); if not, regret is Omega(sqrt(T)). This exponential separation quantifies exactly what structural knowledge buys you.

**Setting:** Linear valuation v_t = theta* . x_t + noise; both stochastic (EMLP algorithm) and adversarial (ONSP algorithm) feature arrival.

**Key result:** Known log-concave noise: O(d log T). Unknown noise: Omega(sqrt(T)) lower bound. The gap is exponential in T, making this the sharpest known separation between parametric and nonparametric regimes.

**Real-world evidence:** None.

### 6. Tullii, Merlis, Gaucher & Perchet, "Contextual Dynamic Pricing with Strategic Surplus" — arXiv 2024

**Why it matters:** Closes a long-open gap: the minimax-optimal rate for contextual pricing under minimal assumptions (Lipschitz noise CDF only, no log-concavity) is T^{2/3}. This settles the "what if we don't know the noise?" question from Xu & Wang 2021 with the tightest possible answer.

**Setting:** Linear and nonparametric contextual valuations. Only assumption: noise CDF is Lipschitz continuous.

**Key result:** Linear contextual: matching upper and lower bounds at Theta(T^{2/3}). Nonparametric Holder-beta smoothness: Theta(T^{(d+2beta)/(d+3beta)}). No log-concavity, no parametric form, no known noise distribution.

**Real-world evidence:** None.

### 7. Liu, Yang, Wang & Sun, "Dynamic Pricing with Strategic Buyers" — arXiv 2024

**Why it matters:** Pure economics: if buyers can manipulate their observable features to get lower prices, every standard pricing algorithm suffers linear Omega(T) regret. Ignoring strategic behavior is not just suboptimal; it is catastrophic. The paper provides a matching Theta(sqrt(T)) algorithm that accounts for manipulation.

**Setting:** Buyers observe their own features and strategically distort them before the seller sees them. The seller observes only manipulated features and buy/no-buy outcomes.

**Key result:** Any policy that ignores strategic manipulation: Omega(T) linear regret. Proposed policy with simultaneous estimation of valuation and manipulation cost: Theta(sqrt(T)) regret, matching the lower bound.

**Real-world evidence:** None (but the Omega(T) impossibility result has immediate implications for any marketplace with feature-based pricing).

### 8. Badanidiyuru, Kleinberg & Slivkins, "Bandits with Knapsacks" — FOCS 2013

**Why it matters:** Foundational framework for pricing under supply, budget, or inventory constraints. Proves that dynamic policies strictly dominate the best static price when resources are limited, formalizing why "set a price and forget it" fails with capacity.

**Setting:** K arms with stochastic knapsack constraints (budget B per resource). BalancedExploration and PrimalDualBwK algorithms.

**Key result:** O(sqrt(Bn) + sqrt(nT)) regret, optimal up to polylogarithmic factors. Dynamic policy provably beats the best fixed arm.

**Real-world evidence:** Framework applies to supply-constrained pricing, ad allocation, procurement, and scheduling.

### 9. Cai, Chen, Wainwright & Zhao, "Doubly High-Dimensional Contextual Bandits" (Hi-CCAB) — 2023

**Why it matters:** The strongest empirical paper in the entire corpus. Achieves 3-4x revenue gains over baselines on real company data, with interpretable latent demand factors that match economic intuition.

**Setting:** Joint assortment selection and pricing; high-dimensional products AND contexts; low-rank matrix demand structure with semi-bandit feedback.

**Key result:** O(sqrt(T) polylog) regret; rank-adaptive; non-asymptotic bounds. Interpretable latent factors decompose demand into meaningful dimensions.

**Real-world evidence:** Instant noodles company (3-4x revenue improvement); beauty product startup. Latent factors recovered by the algorithm correspond to recognizable market segments.

### 10. Ganti, "Thompson Sampling for Dynamic Pricing" — Walmart Labs, 2018

**Why it matters:** The only production deployment of a bandit pricing algorithm in large-scale e-commerce. A 5-week field experiment on Walmart.com with statistically significant revenue improvement proves these methods work at scale.

**Setting:** Thompson Sampling for continuous pricing with constant-elasticity demand (MAX-REV-TS algorithm). Infinite-arm bandit structure.

**Key result:** Statistically significant revenue lift in a 5-week Walmart.com experiment. Simulation shows monotonic revenue growth against a stagnating passive baseline.

**Real-world evidence:** Walmart.com production system, 5-week randomized field experiment.

## What this list leaves out

The full corpus contains roughly 50 papers spanning several areas not represented above. Assortment optimization under MNL choice models (Lee & Oh 2024 close the minimax-optimal gap; Erginbas et al. 2024 achieve tight rates at ICLR) is a substantial subfield. Fairness and privacy constraints have produced sharp results: Chen, Simchi-Levi & Wang 2025 prove that fairness raises the regret floor to T^{2/3}, and Chen, Miao & Wang 2021 give rate-optimal locally differentially private pricing. Semiparametric extensions (Fan, Guo & Yu 2024) and shape-constrained methods (Bracale et al. 2025, validated on Welltower Inc. data) generalize the Javanmard baseline. Mueller, Syrgkanis & Taddy 2019 achieve regret independent of the number of products via low-rank demand. Reference price effects and game-theoretic competition are studied by Agrawal & Tang 2024 and Guo, Ying & Shen 2023. Auction reserve pricing (Cesa-Bianchi, Gentile & Mansour 2015, deployed at Yahoo!) and the causal-RL bridge (Liao et al. 2024, with instrumental variables for confounded MDPs) connect to adjacent literatures. For broader context on the field's development, den Boer's 2015 survey "Dynamic Pricing and Learning: Historical Origins, Current Research, and New Directions" remains the standard reference.
