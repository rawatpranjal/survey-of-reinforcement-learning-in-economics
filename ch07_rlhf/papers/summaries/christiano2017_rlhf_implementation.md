# Deep Reinforcement Learning from Human Preferences - Implementation Summary

## Citation
```bibtex
@article{christiano2017deep,
  title={Deep reinforcement learning from human preferences},
  author={Christiano, Paul F and Leike, Jan and Brown, Tom and Marber, Miljan and Saunders, Shane and Legg, Shane},
  journal={NeurIPS},
  year={2017}
}
```

## Core Contribution

This paper introduces the foundational framework for learning reward functions from human preference comparisons in deep RL. The key innovation is training a reward predictor from pairwise trajectory comparisons while simultaneously training a policy via RL. This decouples the specification of goals (via human feedback) from the optimization process (via RL), enabling agents to learn complex behaviors that are difficult to specify programmatically.

The approach uses asynchronous training where human comparisons are collected on trajectory clips, a reward model is trained on these comparisons, and a policy is optimized against the learned reward. This was demonstrated on Atari games and MuJoCo locomotion tasks.

## Key Equations

### Equation 1: Preference Model (Bradley-Terry over Trajectories)
$$
P[\sigma^1 \succ \sigma^2] = \frac{\exp\left(\sum_t \hat{r}_\psi(o^1_t, a^1_t)\right)}{\exp\left(\sum_t \hat{r}_\psi(o^1_t, a^1_t)\right) + \exp\left(\sum_t \hat{r}_\psi(o^2_t, a^2_t)\right)}
$$
**Notation:**
- $\sigma^i = (o^i_0, a^i_0, o^i_1, a^i_1, \ldots)$: trajectory segment $i$
- $\hat{r}_\psi(o, a)$: learned reward model with parameters $\psi$
- $\succ$: preference relation ("is preferred to")

**Implementation notes:** Sum of rewards over trajectory acts as score. Trajectories should be similar length for fair comparison.

### Equation 2: Reward Model Loss
$$
\mathcal{L}(\psi) = -\sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \left[\mu(1) \log P[\sigma^1 \succ \sigma^2] + \mu(2) \log P[\sigma^2 \succ \sigma^1]\right]
$$
**Notation:**
- $\mathcal{D}$: dataset of comparisons
- $\mu \in \{(1,0), (0,1), (0.5, 0.5)\}$: label (which trajectory preferred, or tie)

**Implementation notes:** Standard cross-entropy loss. Handle ties by splitting label 50-50.

### Equation 3: Normalized Reward (Practical)
$$
\hat{r}_{norm}(o, a) = \frac{\hat{r}_\psi(o, a) - \mu_r}{\sigma_r}
$$
**Notation:**
- $\mu_r, \sigma_r$: running mean and std of rewards

**Implementation notes:** Normalization stabilizes RL training. Update statistics with exponential moving average.

### Equation 4: Policy Objective
$$
\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t \hat{r}_{norm}(o_t, a_t)\right]
$$
**Implementation notes:** Standard RL objective using learned reward. Use any policy gradient method (A2C, PPO).

### Equation 5: Ensemble Reward Model
$$
\hat{r}(o, a) = \frac{1}{K} \sum_{k=1}^K \hat{r}_{\psi_k}(o, a)
$$
**Implementation notes:** Ensemble of $K$ reward models improves robustness and enables uncertainty estimation.

## Testable Claims

### Claim 1: Sample Efficiency
**Statement:** Learning from ~1% of human feedback compared to total environment interactions is sufficient.
**Validation approach:** Track comparison count vs environment steps, measure policy performance.
**Expected result:** 1000-2000 comparisons sufficient for Atari tasks.

### Claim 2: Reward Model Generalization
**Statement:** Reward model trained on trajectory clips generalizes to full episodes.
**Validation approach:** Train on clips, evaluate preference prediction on full trajectories.
**Expected result:** Generalization accuracy $> 75\%$ on held-out full trajectories.

### Claim 3: Asynchronous Training Stability
**Statement:** Training reward model and policy asynchronously (reward updates less frequently than policy) is stable.
**Validation approach:** Compare synchronous vs asynchronous training, measure reward model accuracy and policy performance.
**Expected result:** Asynchronous achieves similar final performance with lower computational cost.

### Claim 4: Robustness to Noise
**Statement:** System is robust to ~10-20% noise in human labels.
**Validation approach:** Inject random label flips, measure degradation.
**Expected result:** Performance degrades gracefully; 10% noise causes $< 20\%$ performance drop.

## Algorithm Pseudocode

```
Algorithm: RLHF with Asynchronous Reward Learning
Input:
  - Environment E
  - Initial policy π_θ
  - Reward model ensemble {r_ψ_1, ..., r_ψ_K}
  - Comparison buffer D

Output: Trained policy π_θ

# Main loop (runs continuously)
1. In parallel:

   # Thread 1: Policy Training
   While not done:
     Collect trajectory τ using π_θ in E
     Compute rewards: r_t = mean_k(r_ψ_k(o_t, a_t))
     Normalize rewards using running statistics
     Update π_θ using PPO/A2C with normalized rewards

   # Thread 2: Trajectory Sampling for Comparison
   While not done:
     Periodically select pairs of trajectory clips (σ^1, σ^2)
     Add to comparison queue for human labeling

   # Thread 3: Reward Model Training
   While not done:
     Receive labeled comparisons (σ^1, σ^2, μ) from humans
     Add to buffer D
     Sample batch from D
     Update each r_ψ_k via gradient descent on L(ψ_k)

Return π_θ
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Clip length | $T_{clip}$ | 25-100 steps | [10, 200] | Shorter = more comparisons needed |
| Ensemble size | $K$ | 3 | [1, 5] | More = better uncertainty, higher cost |
| Reward model updates | - | Every 100 comparisons | - | Batch updates |
| Comparison buffer size | - | 10,000 | [1000, 50000] | Store all comparisons |
| Reward normalization EMA | - | 0.99 | [0.9, 0.999] | For running mean/std |
| Policy algorithm | - | A2C or PPO | - | Any policy gradient works |

## Connections to Other Papers
- **Rafailov et al. 2023**: DPO avoids the policy training step entirely
- **Korbak et al. 2022**: Provides theoretical justification for KL penalty used here
- **Ouyang et al. 2022**: Scales this approach to language models
- **Stiennon et al. 2020**: Applies to summarization task
- **Ziegler et al. 2019**: Earlier LM application

## Simulation Validation Checklist
- [ ] Verify reward model preference accuracy improves with more comparisons
- [ ] Confirm policy performance correlates with reward model accuracy
- [ ] Test robustness to label noise (10%, 20%, 30%)
- [ ] Compare synchronous vs asynchronous training
- [ ] Validate ensemble improves over single reward model
- [ ] Measure sample efficiency (comparisons per environment step)
