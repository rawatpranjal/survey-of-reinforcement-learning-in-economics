# Direct Preference Optimization - Implementation Summary

## Citation
```bibtex
@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={NeurIPS},
  year={2023}
}
```

## Core Contribution

DPO eliminates the need for explicit reward model training in RLHF by showing that the optimal policy under KL-constrained reward maximization has a closed-form solution. This allows reparameterizing the reward function in terms of the optimal policy, converting the RL problem into a supervised classification problem on preference data. The key insight is that learning a reward model and then optimizing against it can be collapsed into a single maximum likelihood objective.

The practical benefit is significant: DPO requires only a reference policy and preference data, avoiding the instabilities of PPO training and the complexity of maintaining separate reward and policy models.

## Key Equations

### Equation 1: Bradley-Terry Preference Model
$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + \exp(-(r(x, y_w) - r(x, y_l)))}
$$
**Notation:**
- $y_w$: preferred (winning) response
- $y_l$: dispreferred (losing) response
- $x$: prompt/context
- $r(x, y)$: reward function
- $\sigma(\cdot)$: sigmoid function

**Implementation notes:** This is the foundation for all preference-based methods. Log-likelihood is numerically stable using `log_sigmoid`.

### Equation 2: KL-Constrained Reward Maximization Objective
$$
\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[r(x, y)] - \beta \mathbb{D}_{KL}[\pi(y|x) \| \pi_{ref}(y|x)]
$$
**Notation:**
- $\pi$: policy being optimized
- $\pi_{ref}$: reference policy (typically SFT model)
- $\beta$: KL penalty coefficient (inverse temperature)
- $\mathcal{D}$: distribution over prompts

**Implementation notes:** The KL penalty prevents the policy from deviating too far from the reference, maintaining generation quality.

### Equation 3: Optimal Policy (Closed Form)
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$
**Notation:**
- $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$: partition function

**Implementation notes:** This is the key theoretical result. The partition function $Z(x)$ is intractable but cancels in the DPO loss.

### Equation 4: Reward Reparameterization
$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$
**Implementation notes:** Substituting this into Bradley-Terry, the $Z(x)$ terms cancel for paired comparisons.

### Equation 5: DPO Loss
$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$
**Implementation notes:**
- Compute log-probabilities under both $\pi_\theta$ and $\pi_{ref}$
- Use `F.logsigmoid` for numerical stability
- Reference policy is frozen (no gradients)

### Equation 6: DPO Gradient
$$
\nabla_\theta \mathcal{L}_{DPO} = -\beta \mathbb{E}\left[\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \left[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right]\right]
$$
where $\hat{r}_\theta(y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$

**Implementation notes:** The sigmoid term acts as an importance weight, downweighting gradients when the model already correctly ranks the pair.

## Testable Claims

### Claim 1: DPO-PPO Equivalence Under Infinite Data
**Statement:** As dataset size $\to \infty$, DPO policy converges to the same policy as PPO with explicit reward model.
**Validation approach:** Train both on same preference data with increasing $N$, measure policy KL divergence.
**Expected result:** KL divergence between DPO and PPO policies decreases as $O(1/\sqrt{N})$.

### Claim 2: Reward-KL Pareto Frontier
**Statement:** DPO achieves points on the Pareto frontier of expected reward vs. KL from reference.
**Validation approach:** Vary $\beta$ in $\{0.01, 0.1, 0.5, 1.0\}$, plot reward vs. KL for DPO and PPO.
**Expected result:** DPO curve should match or dominate PPO curve.

### Claim 3: Implicit Reward Recovery
**Statement:** The implicit reward $\hat{r}_\theta(y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ recovers true reward up to constant.
**Validation approach:** With known ground-truth reward, compute correlation between $\hat{r}$ and $r$.
**Expected result:** Pearson correlation $> 0.9$ after convergence.

### Claim 4: Importance Weighting Reduces Gradient Variance
**Statement:** The sigmoid weighting in the gradient reduces variance compared to unweighted updates.
**Validation approach:** Compare gradient variance with and without sigmoid weighting during training.
**Expected result:** Weighted gradients have $2-5\times$ lower variance.

## Algorithm Pseudocode

```
Algorithm: Direct Preference Optimization (DPO)
Input:
  - Reference policy π_ref (frozen)
  - Preference dataset D = {(x_i, y_w^i, y_l^i)}
  - Temperature β
  - Learning rate α

Output: Optimized policy π_θ

1. Initialize π_θ ← π_ref  # Start from reference
2. For each epoch:
3.   For each batch B ⊂ D:
4.     For each (x, y_w, y_l) in B:
5.       # Compute log-probability ratios
6.       log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
7.       log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
8.       # Implicit reward difference
9.       reward_diff = β * (log_ratio_w - log_ratio_l)
10.    # Compute loss
11.    loss = -mean(log_sigmoid(reward_diff))
12.    # Update
13.    θ ← θ - α * ∇_θ loss
14. Return π_θ
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Temperature | $\beta$ | 0.1 | [0.01, 0.5] | Lower = more aggressive optimization |
| Learning rate | $\alpha$ | 1e-6 to 5e-7 | [1e-7, 1e-5] | Much lower than SFT |
| Batch size | $B$ | 64 | [32, 128] | Per-GPU batch size |
| Epochs | - | 1-3 | [1, 5] | Overfitting is a concern |
| Warmup ratio | - | 0.1 | [0.05, 0.2] | Fraction of steps for LR warmup |

## Connections to Other Papers
- **Christiano et al. 2017**: DPO uses same Bradley-Terry model but avoids explicit RM
- **Korbak et al. 2022**: Provides Bayesian justification for KL penalty that DPO inherits
- **McFadden 1974**: Bradley-Terry is special case of conditional logit
- **Ouyang et al. 2022**: DPO is alternative to InstructGPT's PPO stage

## Simulation Validation Checklist
- [ ] Verify DPO loss decreases monotonically on training data
- [ ] Confirm implicit reward $\hat{r}$ correlates with ground-truth reward
- [ ] Test that $\beta \to 0$ produces more aggressive policy
- [ ] Compare preference accuracy: DPO vs explicit RM + PPO
- [ ] Measure KL from reference at various $\beta$ values
- [ ] Validate gradient importance weighting reduces variance
