# RL with KL Penalties is Better Viewed as Bayesian Inference - Implementation Summary

## Citation
```bibtex
@article{korbak2022rl,
  title={RL with KL penalties is better viewed as Bayesian inference},
  author={Korbak, Tomasz and Perez, Ethan and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2205.11275},
  year={2022}
}
```

## Core Contribution

This paper provides a principled Bayesian interpretation of KL-regularized RL, showing that the KL penalty is not merely a regularization trick but arises naturally from viewing RLHF as approximate Bayesian inference. The optimal policy under KL-constrained reward maximization is exactly the Bayesian posterior when the reference policy is the prior and the reward defines the likelihood. This justifies the ubiquitous KL penalty in RLHF and connects it to variational inference.

The key insight is that maximizing $\mathbb{E}[r(y)] - \beta D_{KL}[\pi \| \pi_{ref}]$ is equivalent to minimizing KL divergence to a Bayesian posterior $p(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$.

## Key Equations

### Equation 1: KL-Regularized Objective
$$
\mathcal{J}(\pi) = \mathbb{E}_{y \sim \pi(\cdot|x)}[r(x, y)] - \beta D_{KL}[\pi(y|x) \| \pi_{ref}(y|x)]
$$
**Notation:**
- $\pi$: policy being optimized
- $\pi_{ref}$: reference (prior) policy
- $r(x, y)$: reward function
- $\beta > 0$: inverse temperature / KL penalty weight

**Implementation notes:** Standard RLHF objective used in PPO with KL penalty.

### Equation 2: Bayesian Posterior Policy
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)
$$
**Notation:**
- $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$: normalizing constant

**Implementation notes:** This is the unique maximizer of $\mathcal{J}(\pi)$. Cannot compute $Z(x)$ in practice.

### Equation 3: Variational Interpretation
$$
\max_\pi \mathcal{J}(\pi) \iff \min_\pi D_{KL}[\pi(y|x) \| \pi^*(y|x)]
$$
**Implementation notes:** KL-regularized RL is equivalent to variational inference with $\pi^*$ as the target posterior.

### Equation 4: Evidence Lower Bound (ELBO)
$$
\log Z(x) \geq \mathbb{E}_{y \sim \pi}[r(x,y)/\beta] - D_{KL}[\pi \| \pi_{ref}] = \frac{1}{\beta}\mathcal{J}(\pi)
$$
**Implementation notes:** The objective $\mathcal{J}(\pi)$ is (up to scaling) the ELBO for the posterior $\pi^*$.

### Equation 5: Optimal Value
$$
\mathcal{J}(\pi^*) = \beta \log Z(x)
$$
**Implementation notes:** At optimum, the objective equals $\beta$ times the log-partition function.

### Equation 6: Temperature Interpretation
$$
\lim_{\beta \to 0} \pi^*(y|x) = \delta_{y^*}(y) \quad \text{where } y^* = \arg\max_y r(x, y)
$$
$$
\lim_{\beta \to \infty} \pi^*(y|x) = \pi_{ref}(y|x)
$$
**Implementation notes:** Low $\beta$ = greedy optimization; high $\beta$ = stay close to reference.

## Testable Claims

### Claim 1: Objective Equivalence
**Statement:** Maximizing $\mathcal{J}(\pi)$ produces the same policy as minimizing $D_{KL}[\pi \| \pi^*]$.
**Validation approach:** Compute both objectives during training, verify they decrease together.
**Expected result:** Correlation between $\mathcal{J}$ and $-D_{KL}[\pi \| \pi^*]$ should be $> 0.95$.

### Claim 2: ELBO Tightness
**Statement:** As $\pi \to \pi^*$, the ELBO gap closes: $\log Z(x) - \mathcal{J}(\pi)/\beta \to 0$.
**Validation approach:** Track ELBO gap during training.
**Expected result:** Gap decreases monotonically and approaches 0.

### Claim 3: Temperature Controls Exploration-Exploitation
**Statement:** Lower $\beta$ increases reward at cost of higher KL; higher $\beta$ reduces KL at cost of reward.
**Validation approach:** Train with varying $\beta$, plot reward vs KL Pareto frontier.
**Expected result:** Monotonic tradeoff curve.

### Claim 4: Posterior Convergence
**Statement:** Policy gradient methods converge to the Bayesian posterior $\pi^*$.
**Validation approach:** In tractable setting, compute exact $\pi^*$ and compare to learned policy.
**Expected result:** Total variation distance $< 0.01$ after convergence.

## Algorithm Pseudocode

```
Algorithm: Bayesian-Interpreted KL-Regularized Policy Gradient
Input:
  - Reference policy π_ref
  - Reward function r(x, y) (or learned reward model)
  - Temperature β
  - Prompts distribution p(x)

Output: Policy π_θ ≈ π*

1. Initialize π_θ ← π_ref
2. For each iteration:
3.   # Sample batch of prompts
4.   x_1, ..., x_B ~ p(x)
5.
6.   # Generate responses from current policy
7.   For each x_i:
8.     y_i ~ π_θ(·|x_i)
9.
10.  # Compute KL-regularized rewards
11.  For each (x_i, y_i):
12.    r_kl = r(x_i, y_i) - β * (log π_θ(y_i|x_i) - log π_ref(y_i|x_i))
13.
14.  # Policy gradient update (e.g., PPO)
15.  advantage = r_kl - baseline
16.  θ ← θ + α * ∇_θ Σ_i log π_θ(y_i|x_i) * advantage
17.
18. Return π_θ
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Temperature | $\beta$ | 0.05 to 0.2 | [0.01, 1.0] | Controls prior strength |
| KL coefficient | $\beta$ | 0.1 | [0.01, 0.5] | Same as temperature in objective |
| Target KL | - | 0.01 to 0.05 | [0.001, 0.1] | Used for adaptive β |
| Entropy bonus | - | 0.01 | [0, 0.1] | Additional regularization |

## Connections to Other Papers
- **Rafailov et al. 2023**: DPO directly optimizes the posterior using the same structure
- **Christiano et al. 2017**: Uses KL penalty without Bayesian justification
- **Ouyang et al. 2022**: InstructGPT uses this objective with PPO
- **Levine 2018**: Control as inference framework, similar variational view

## Simulation Validation Checklist
- [ ] Verify ELBO interpretation holds numerically
- [ ] Confirm policy converges to posterior in tractable case
- [ ] Test temperature sweep produces expected Pareto frontier
- [ ] Compare PPO-KL to DPO under Bayesian interpretation
- [ ] Measure ELBO gap during training
- [ ] Validate that $\beta \to 0$ recovers greedy policy
