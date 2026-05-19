- we should show RL for single agent before het agent / hank stuffs (separate from MFG)
- separete MFG from Hank if possible, the are related but maybe its possible. I think MFG allows more types of games, maybe we can include another "type" of game e.g. some type of congestion game? so we can showcase 2 MFG variants with 2 papers
- for agent based lets go with that salesforce tax example and rice


for Solver--> SRL is a good anchor but we need a paper before that for simpler RBC/DSGE models
for MFG formulation --> lets get another one
Bounded rationality --> need something truiely behaivoural. regualr non beh models do nto stay here. need one leading example only. 
for agent based, lets put the tax + RICEN22 (Zhang 2022). 





donwload and check if these papers fit? 

Ranked by novelty × credibility, deduplicated by methodological type:

**1. Storch, Timme, Schröder (2020), *Nature Communications*** — *Econophysics phase diagram on surge*
The standout weirdo paper. Combines time-series analysis of Uber prices in 137 cities with a game-theoretic phase diagram showing when driver collective log-off becomes a Nash equilibrium that manufactures surge. Top-tier journal, real Uber data, genuinely unconventional method. https://www.nature.com/articles/s41467-020-18370-3

**2. Castillo, Knoepfle, Weyl (2024), *Management Science*** — *Continuum-driver mean-field equilibrium with multiple equilibria*
Proves the wild-goose-chase equilibrium can co-exist with the high-throughput one and identifies surge as the equilibrium-selection device. Direct Uber-research lineage (Knoepfle was at Uber). Most ops-defensible interpretable rule on the list. https://pubsonline.informs.org/doi/10.1287/mnsc.2022.00096

**3. Garg, Nazerzadeh (2022), *Management Science*** — *Continuous-time HJB on driver incentive compatibility*
Two-state CTMC + Laplace-transform proof that multiplicative surge is not IC for drivers but additive surge is. This is the math behind Uber's current driver-pay architecture. https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4058

**4. Buchholz (2022), *Review of Economic Studies*** — *Empirical dynamic spatial search-and-matching*
City-scale DMP-style equilibrium estimated on the full NYC yellow-cab trip dataset. Recovers the matching function and quantifies $5.7M/day welfare loss from search frictions. Top-5 econ journal. https://academic.oup.com/restud/article/89/2/556/6375457

**5. Allen, Arkolakis, Li (2024), *AER: Insights*** — *Contraction-mapping fixed-point for spatial GE*
The mathematical engine for computing equilibrium across thousands of zones with cross-market spillovers. No platform application exists yet — that gap is the opportunity. https://www.aeaweb.org/articles?id=10.1257/aeri.20230495

**6. Cashore, Frazier, Tardos (2023), *Mathematics of Operations Research*** — *Stochastic spatiotemporal mechanism design*
Proves static prices admit catastrophic equilibria with arbitrarily low welfare under correlated shocks; dynamic re-solving prevents them. The strongest argument that surge cannot be replaced with a clever static schedule. https://pubsonline.informs.org/doi/10.1287/moor.2022.0163

**7. Dütting, Feng, Narasimhan, Parkes, Ravindranath (2024 CACM, ICML 2019)** — *Differentiable mechanism design*
RegretNet: parameterize allocation and payment as neural nets, train under a soft-IC penalty. Bypasses closed-form Myerson for correlated valuations and item bundles. Relevant to Uber Eats sponsored placement and driver-incentive auctions. https://cacm.acm.org/research/optimal-auctions-through-deep-learning/

**8. Oda (2021), *WWW Proceedings*** — *Equilibrium inverse reinforcement learning*
Spatial Equilibrium IRL: recovers a driver reward function consistent with multi-agent passenger-seeking equilibrium, then uses it for counterfactual dispatch. Validated against production data at Mobility Technologies. https://dl.acm.org/doi/fullHtml/10.1145/3442381.3449935

**9. Overwater, Yorke-Smith (2022), *Environment and Planning B*** — *ABM with structural economic theory*
Geographically-accurate Amsterdam Airbnb simulation grounded in Smith's rent-gap hypothesis (not just behavior rules). The only ABM template on the list with real economic structure rather than DES-with-ML. https://journals.sagepub.com/doi/full/10.1177/23998083211000747