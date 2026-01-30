# Theoretical Foundations of Dynamic Programming and Reinforcement Learning

**Sample complexity bounds for online reinforcement learning have been definitively characterized over the past decade**, with minimax optimal rates now established across tabular, linear, and structured MDP settings. This survey synthesizes the landmark results establishing fundamental limits alongside recent 2020-2025 breakthroughs that have closed major theoretical gaps—particularly the surprising proof that model-free Q-learning is provably suboptimal compared to model-based methods by a factor of **1/(1-γ)**.

## Value Iteration converges linearly while Policy Iteration achieves Newton-like quadratic convergence

The fundamental distinction between value iteration (VI) and policy iteration (PI) reflects a **gradient descent versus Newton's method** dichotomy established by Puterman and Brumelle (1979). Value iteration contracts at geometric rate γᵏ, requiring **O(log(1/ε)/(1-γ))** iterations for ε-accuracy. The Bellman operator is a γ-contraction under the supremum norm, guaranteeing convergence via Banach's fixed-point theorem with per-iteration cost O(|S|²|A|).

Policy iteration exhibits **superlinear to quadratic convergence** because each update exactly solves a linear system encoding second-order curvature information. The technical equivalence shows PI updates as Vₙ₊₁ = Vₙ + [I - γPπₙ]⁻¹B(Vₙ), which is precisely Newton's method applied to the Bellman residual operator. Local quadratic convergence satisfies ‖Vₙ₊₁ - V*‖ ≤ C‖Vₙ - V*‖², though per-iteration cost rises to O(|S|³) for matrix inversion.

A critical computational distinction emerges: **VI is not strongly polynomial** while **PI is strongly polynomial**. Feinberg, Huang, and Scherrer (2014) showed MDPs exist where VI requires unbounded iterations for exact optimality. In contrast, Ye (2011) and Scherrer (2016) proved PI finds exact optimal policies in O(|S||A|/(1-γ)) iterations. When discount factors approach 1 (long horizons), PI's quadratic convergence dominates despite higher per-iteration cost—empirically converging in under 20 iterations regardless of problem size.

## Minimax sample complexity is Θ(|S||A|/((1-γ)³ε²)) for the generative model setting

Azar, Munos, and Kappen's 2013 landmark paper established the first **minimax-optimal** sample complexity bound for discounted MDPs with generative model access. Their upper bound of O(N·log(N/δ)/((1-γ)³ε²)) matches the lower bound Θ(N·log(N/δ)/((1-γ)³ε²)) across all parameters: state-action pairs N=|S||A|, accuracy ε, confidence δ, and effective horizon 1/(1-γ). The key innovation exploited variance structure: Var(V*) ≤ V*ₘₐₓ·H, enabling the tight (1-γ)⁻³ dependence versus naive (1-γ)⁻⁴ bounds.

Sidford et al. (2018) extended this to achieve **near-optimal time AND sample complexity** simultaneously: Õ(|S||A|/((1-γ)³ε²)·log(1/((1-γ)ε))) total operations. Their variance-reduced value iteration closed the gap between statistical and computational efficiency—previous algorithms achieved optimal sample complexity with worse running time or vice versa.

For **online episodic learning** without simulator access, the minimax regret is **Ω(√(HSAT))** where H is horizon, S states, A actions, T total timesteps (Osband & Van Roy 2016). The UCBVI-BF algorithm (Azar, Osband, Munos 2017) achieves Õ(√(HSAT)) for sufficiently large T, matching this lower bound. Zhang, Chen, Lee, and Du (COLT 2024) achieved the breakthrough of minimax-optimal regret **min{√(SAH³K), HK}** for all sample sizes K ≥ 1, eliminating the burn-in problem that plagued previous algorithms.

## Q-learning is provably suboptimal: the model-based versus model-free gap

**Li et al. (Operations Research 2024)** definitively resolved the long-standing question of Q-learning's optimality. For synchronous Q-learning with generative model:

| Setting | Sample Complexity | Optimality |
|---------|------------------|------------|
| TD learning (|A|=1) | Θ(|S|/((1-γ)³ε²)) | ✓ Minimax optimal |
| Q-learning (|A|≥2) | Θ(|S||A|/((1-γ)⁴ε²)) | ✗ Strictly suboptimal |
| Model-based | Θ(|S||A|/((1-γ)³ε²)) | ✓ Minimax optimal |

The **1/(1-γ) gap** for Q-learning when |A|≥2 stems from over-estimation bias—rigorously proven unavoidable through hard MDP constructions (4 states, 2 actions). Variance-reduced Q-learning recovers optimal rates by eliminating this bias, achieving Õ(|S||A|/((1-γ)³ε²)). This confirms model-based approaches are fundamentally superior to vanilla model-free methods for sample efficiency.

Jin et al. (2018) answered "Is Q-learning Provably Efficient?" affirmatively for the online setting, proving UCB-augmented Q-learning achieves **Õ(√(H³SAT))** regret—matching model-based methods within a √H factor. This was the first √T regret for model-free RL without simulator access, demonstrating model-free methods can be sample-efficient despite the generative model gap.

## Exploration requires Ω(√HSAT) regret: optimism and Thompson sampling achieve near-optimality

The **optimism in the face of uncertainty (OFU)** principle underlies provably efficient exploration. UCRL2 (Jaksch, Ortner, Auer 2010) established Õ(DS√AT) regret for undiscounted MDPs with diameter D, constructing confidence sets around transition probabilities. UCBVI refined this for episodic settings using Bernstein-type exploration bonuses that exploit empirical variance:

$\text{bonus}(s,a,h) = \beta \cdot \sqrt{\frac{\text{Var}[\hat{V}_{h+1}]}{n(s,a)}} + \frac{\beta}{n(s,a)}$

The key technical insight: applying concentration inequalities directly to value functions (not just transitions) and using the law of total variance recursively improves S-dependence from S to √S.

**Posterior Sampling for RL (PSRL)** offers an alternative Bayesian approach (Osband, Russo, Van Roy 2013). Maintaining Dirichlet priors over transitions and sampling an MDP from the posterior each episode achieves Õ(HS√AT) Bayesian regret—often empirically superior to optimistic methods. Tiapkin et al. (2022) proved optimistic PSRL achieves frequentist regret Õ(√HSAT), matching information-theoretic limits.

**Lower bounds on exploration** are established via Le Cam's method or Fano's inequality on families of statistically indistinguishable MDPs. The Ω(√HSAT) lower bound (episodic) and Ω(√DSAT) (undiscounted) are tight. For discounted MDPs, He, Zhou, and Gu (NeurIPS 2021) proved both upper and lower bounds of Õ(√(SAT)/(1-γ)^1.5)—the first tight minimax characterization for this setting.

## The curse of dimensionality can be overcome only through model-based structural assumptions

**When exponential dependence is unavoidable**: Weisz et al. (2020) proved that if only Q* lies in the linear span of d-dimensional features, sample complexity is **Ω(exp(d))** even with generative model access. Du et al. (2020) showed exponential lower bounds persist even with perfect value function approximation: the structure must be in the dynamics, not just value functions. Wang et al. (2021) extended this to online settings without generative models.

**When polynomial bounds are achievable**: The following structural assumptions enable dimension-free sample complexity:

| Structure | Sample Complexity | Key Property |
|-----------|------------------|--------------|
| Linear MDP | Õ(d³H⁴/ε²) | P, r linear in known features |
| Linear Mixture MDP | Õ(d²H³/ε²) | P = Σᵢθᵢφᵢ(s,a,s') |
| Low-Rank MDP | Õ(d⁴H⁵K/ε²) | Rank-d transition matrix |
| Factored MDP | Õ(Σᵢ|Sᵢ||Aᵢ|/ε²) | Sum vs product of components |
| Low Bellman Eluder | poly(d_BE, H)/ε² | Unifies prior notions |

Jin et al. (2020) introduced LSVI-UCB for **linear MDPs**—the first algorithm with both polynomial runtime and polynomial sample complexity for function approximation. Regret scales as Õ(√(d³H³T)), with lower bound Ω(√(dH²T)) leaving a √(dH) gap. He et al. (ICML 2023) achieved the first **computationally efficient minimax-optimal** algorithm for linear MDPs: Õ(d√(H³K)) regret using weighted linear regression with monotonically decreasing variance estimators.

**Eluder dimension** (Russo & Van Roy 2013) and **Bellman rank** (Jiang et al. 2017) provide unified complexity measures. The **Bellman-Eluder (BE) dimension** (Jin et al. 2021) subsumes both, enabling polynomial sample complexity independent of |S|×|A| for tabular, linear, low-rank, and reactive POMDP settings through the GOLF algorithm.

## Computational complexity distinguishes tractable from intractable RL problems

Papadimitriou and Tsitsiklis (1987) established foundational hardness results:

- **Fully observable MDP**: P-complete (polynomial-time solvable but likely not parallelizable)
- **POMDP**: PSPACE-complete (requires exponential time in general)
- **Decentralized MDP**: NEXP-complete (provably intractable assuming P ≠ NEXP)
- **Binary-encoded horizon MDP**: EXPTIME-complete for value iteration

Standard tabular MDPs admit O(|S|²|A|H) planning via value/policy iteration. Linear MDPs maintain polynomial tractability through LSVI-UCB. However, low Bellman rank algorithms like OLIVE are statistically efficient but computationally intractable for general function classes—the gap between information-theoretic and computational efficiency remains a frontier.

## Recent advances have closed major theoretical gaps

The 2020-2025 period produced decisive results resolving long-standing open problems:

**Instance-dependent bounds** (Dann et al. 2021) move beyond worst-case analysis. With positive sub-optimality gap Δₘᵢₙ, rates improve from minimax Õ(1/√K) to **fast rate Õ(1/K)**—exponentially faster when problems have favorable gap structure.

**Variance-reduced methods** proved essential for matching model-based optimality with model-free algorithms. The reference-advantage decomposition (Li et al. 2021) broke the S⁶A⁴poly(H) burn-in barrier, achieving near-optimal regret when samples exceed SA·poly(H)—an improvement factor ≥ S⁵A³ over prior methods.

**Average-reward MDPs** received their first minimax-optimal bounds (NeurIPS 2024), using span H (always finite for finite MDPs) and novel "transient time" characterization to resolve long-standing open problems about non-episodic infinite-horizon settings.

## Conclusion

Reinforcement learning theory has achieved remarkable maturity since Kakade's 2003 thesis established the PAC framework. The minimax sample complexity Θ(|S||A|/((1-γ)³ε²)) is tight for tabular MDPs with generative models. Model-based approaches provably dominate vanilla Q-learning by 1/(1-γ)—variance reduction is necessary and sufficient to close this gap. The curse of dimensionality is unavoidable for value-based assumptions alone but can be escaped through model-based structural assumptions like linearity, low-rank, or factorization. Policy iteration's Newton-like quadratic convergence explains its empirical dominance on long-horizon problems despite higher per-iteration cost.

The field's trajectory points toward increasingly tight instance-dependent bounds, computationally efficient algorithms for structured function approximation, and unified complexity measures that predict when polynomial sample complexity is achievable. The theoretical foundations are now mature enough to guide algorithm design with precise quantitative predictions.
