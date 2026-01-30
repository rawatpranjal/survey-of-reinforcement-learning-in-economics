# Proximal Policy Optimization Algorithms (Schulman et al., 2017)

## The Problem (Layperson)

TRPO provided reliable policy improvement but was complicated to implement. It required computing Fisher-vector products, running conjugate gradient optimization, and performing line searches. Could we achieve similar reliability with a simpler algorithm that uses only first-order gradients and standard stochastic gradient descent?

The challenge was that standard policy gradient methods are unstable. Taking multiple gradient steps on the same batch of data leads to destructive updates because the policy changes, making the old data increasingly irrelevant. TRPO solved this with complex constrained optimization. Could a simpler approach work?

## What Didn't Work (Alternatives)

Standard policy gradient with multiple epochs: Using the same batch of data for multiple gradient updates caused catastrophic performance collapse. Each update moved the policy further from the distribution that generated the data, making importance sampling corrections increasingly inaccurate.

Fixed penalty on KL divergence: Adding a term $\beta \cdot D_{KL}$ to the objective was theoretically motivated but practically problematic. No single value of $\beta$ worked across problems, and even within a single problem, the right value changed during training.

Conservative step sizes: Using very small learning rates avoided catastrophe but learned too slowly to be practical. The right step size varied moment to moment in ways that fixed schedules could not capture.

## The Key Insight

Instead of constraining the KL divergence, clip the probability ratio. Define:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

When advantage is positive (action was good), we want to increase the probability. The clip prevents increasing $r$ beyond $1+\varepsilon$, limiting how much we can exploit this sample.

When advantage is negative (action was bad), we want to decrease the probability. The clip prevents decreasing $r$ below $1-\varepsilon$, limiting the policy change.

The min operator takes the worse of the clipped and unclipped objectives, creating a pessimistic (lower bound) estimate that prevents excessive updates in either direction.

## The Method

PPO alternates between sampling and optimization:

```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        Run policy π_θold for T timesteps
        Compute advantage estimates
    end for
    for epoch = 1, 2, ..., K do
        for minibatch in random_permutation(all_data) do
            Compute L^CLIP
            Update θ using Adam
        end for
    end for
    θold ← θ
end for
```

Key hyperparameters:
- $\varepsilon = 0.1$ or $0.2$ (clip range)
- $K = 3$ to $15$ epochs per batch of data
- Minibatch size and learning rate follow standard deep learning practices

The algorithm also learns a value function $V(s)$ to estimate advantages using GAE (Generalized Advantage Estimation). The combined objective includes value function error and entropy bonus:

$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta]$$

## The Result

PPO achieved similar or better performance than TRPO on continuous control tasks while being much simpler to implement. It required only a few lines of code change from vanilla policy gradient.

On MuJoCo locomotion benchmarks, PPO outperformed A2C, TRPO, and other baselines. It learned humanoid running and steering from scratch with a neural network policy.

On Atari games, PPO matched or exceeded prior methods while being significantly simpler than ACER (which includes experience replay and correction terms).

The algorithm was robust to hyperparameter choices. $\varepsilon = 0.2$ worked well across diverse problems. Multiple epochs of updates on the same data improved sample efficiency without causing divergence.

## Worked Example

Consider learning Hopper locomotion. Initial policy produces random joint torques, causing the robot to fall immediately.

Batch collection:
- Run 2048 timesteps across parallel environments
- Most episodes end early (robot falls)
- Record states, actions, rewards, value estimates
- Compute advantages using GAE: good actions that delayed falling get positive $\hat{A}$

Epoch 1 of optimization:
- Sample minibatch of 64 transitions
- For each transition, compute $r = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$
- If $\hat{A} > 0$ and $r > 1.2$, the clip activates, limiting the gradient
- If $\hat{A} < 0$ and $r < 0.8$, the clip activates, limiting the gradient
- Within the clip range, gradients flow normally
- Update parameters with Adam

Epochs 2-10:
- Same data, more gradient steps
- As policy changes, probability ratios drift from 1
- Clips activate more frequently, dampening updates
- Policy improves but cannot change too much from initial policy

After many iterations:
- Hopper learns to hop forward
- Each batch of data is used for 10 epochs of updates
- Clips prevent catastrophic updates while allowing substantial learning per batch

## Subtleties

The clipped objective is not differentiable at the clip boundaries, but this is fine for gradient descent. The gradient is zero when clipped (no learning signal) and nonzero otherwise.

Multiple epochs on the same batch is crucial for sample efficiency but potentially dangerous. The clips prevent the worst outcomes, but learning can still slow if the policy changes too much within one outer iteration. Empirically, 3-15 epochs work well.

Advantage normalization is important in practice. Normalizing advantages to zero mean and unit variance within each minibatch prevents updates from being dominated by outliers.

The value function is typically shared with the policy (same network with different heads). This parameter sharing can cause interference, but it also provides regularization and improves generalization.

PPO is still on-policy: it discards data after each outer iteration. This limits sample efficiency compared to off-policy methods, but the multiple epochs of reuse mitigate this somewhat.

## Critical Debates

PPO versus TRPO: Is the clipping mechanism as theoretically justified as TRPO's KL constraint? Empirically, both achieve similar results, but the theoretical connection between clipping and trust regions is less clear. Some argue PPO works for different reasons than TRPO.

Why does PPO work? The standard explanation is that clipping creates a trust region. But careful analysis shows the effective constraint varies depending on the advantage sign and magnitude. Understanding the true mechanism remains incomplete.

Sample efficiency: PPO is on-policy, so it requires many environment interactions. Off-policy methods like SAC achieve better sample efficiency on some benchmarks. PPO compensates with simplicity and wall-clock efficiency through parallelism.

Robustness claims: While PPO is robust to hyperparameters within each problem class, transferring hyperparameters between domains (e.g., from MuJoCo to Atari) requires adjustment. The "universal" hyperparameters are domain-specific.

PPO's dominance: PPO became the default policy gradient algorithm, used in everything from robotics to RLHF. But this dominance may reflect implementation maturity and community familiarity as much as fundamental superiority.

## Key Quotes

"We propose a novel objective function that enables multiple epochs of minibatch updates." (Abstract)

"The motivation for this objective is as follows. The first term inside the min is L^CPI. The second term modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving r_t outside of the interval [1-ε, 1+ε]." (Section 3)

"We have introduced proximal policy optimization, a family of policy optimization methods that use multiple epochs of stochastic gradient ascent to perform each policy update. These methods have the stability and reliability of trust-region methods but are much simpler to implement." (Section 7)

## Citation

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
