# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Williams, 1992)

## The Problem

Gradient-based optimization of neural networks requires differentiable loss functions. Reinforcement learning provides only a scalar reward signal $r$ that is not a differentiable function of the network weights. The fundamental question: how can we compute $\nabla_w \mathbb{E}[r]$ when $r$ depends on the environment's response to the network's output, and this dependence is unknown or non-differentiable?

Supervised learning computes weight gradients via backpropagation through a known loss function $L(y, y^*)$. In reinforcement learning, the "loss" is the negative reward, but there is no target $y^*$ and no differentiable path from weights to reward. The network must estimate gradients from the statistical relationship between weight changes and reward changes.

## What Didn't Work (Alternatives)

Supervised learning methods like backpropagation required a differentiable error function with known correct outputs. They could not handle evaluative feedback.

Genetic algorithms and evolutionary methods searched weight space through random variation and selection. But they were sample-inefficient and did not exploit gradient information.

Simpler reinforcement learning approaches used finite-difference gradient estimates: perturb each weight slightly, observe the change in reward, estimate the gradient. But this scaled poorly: computing the gradient required O(n) perturbations for n weights.

## The Key Insight

Stochastic units enable gradient estimation without weight perturbation. If network outputs are sampled from probability distributions parameterized by the weights, then the gradient of expected reward with respect to any weight equals:

$$\nabla_w \mathbb{E}[r] = \mathbb{E}\left[ r \cdot \nabla_w \log \pi(y|x,w) \right]$$

This is the policy gradient theorem in modern terminology. The gradient is an expectation over the network's own stochastic outputs. It can be estimated from samples without perturbing weights.

The key mathematical insight is that $\frac{\partial}{\partial w} p(y|w) = p(y|w) \frac{\partial}{\partial w} \log p(y|w)$, so the gradient of expected reward can be written as an expectation under the same distribution. No separate perturbation distribution is needed.

## The Method

A REINFORCE algorithm uses weight updates of the form:

$$\Delta w_{ij} = \alpha (r - b) e_{ij}$$

where:
- $\alpha > 0$ is the learning rate
- $r$ is the received reinforcement
- $b$ is a reinforcement baseline (reducing variance without introducing bias)
- $e_{ij} = \frac{\partial \ln g_i}{\partial w_{ij}}$ is the characteristic eligibility

The **characteristic eligibility** $e_{ij}$ is the gradient of the log-probability of unit $i$'s output with respect to weight $w_{ij}$:

$$e_{ij} = \frac{\partial \ln g_i(y_i, w_i, x)}{\partial w_{ij}}$$

where $g_i$ is the probability density (or mass) function of unit $i$'s output $y_i$ given weights $w_i$ and inputs $x$.

**Bernoulli units.** For a unit with $\mathbb{P}(y_i = 1) = p_i = \sigma(\sum_j w_{ij} x_j)$ where $\sigma$ is the logistic function:

$$e_{ij} = (y_i - p_i) x_j$$

**Gaussian units.** For $y_i \sim \mathcal{N}(\mu_i, \sigma^2)$ with $\mu_i = \sum_j w_{ij} x_j$ and fixed variance $\sigma^2$:

$$g_i(y_i) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - \mu_i)^2}{2\sigma^2}\right)$$

$$e_{ij} = \frac{(y_i - \mu_i)}{\sigma^2} x_j$$

The eligibility is proportional to the deviation of the sampled output from the mean, scaled by the input.

## The Result

**Theorem 1.** For any REINFORCE algorithm, the expected weight update is proportional to the gradient of expected reinforcement:

$$\mathbb{E}[\Delta w_{ij} | W] = \alpha_{ij} \frac{\partial \mathbb{E}[r | W]}{\partial w_{ij}}$$

where expectation is over the stochastic outputs of the network given fixed weights $W$.

**Proof sketch.** The key identity is: $\frac{\partial p(y)}{\partial w} = p(y) \frac{\partial \ln p(y)}{\partial w}$. Therefore:

$$\frac{\partial \mathbb{E}[r]}{\partial w} = \frac{\partial}{\partial w} \sum_y p(y) r(y) = \sum_y r(y) p(y) \frac{\partial \ln p(y)}{\partial w} = \mathbb{E}\left[r \frac{\partial \ln p(y)}{\partial w}\right]$$

The log-derivative $\frac{\partial \ln p(y)}{\partial w_{ij}}$ equals the characteristic eligibility $e_{ij}$ summed over units.

**Baseline invariance.** The baseline $b$ does not affect the expected gradient because $\mathbb{E}[b \cdot e_{ij}] = b \cdot \mathbb{E}[e_{ij}] = 0$ when $b$ does not depend on the output. This follows from $\mathbb{E}[\nabla_w \ln p(y)] = \nabla_w \sum_y p(y) = \nabla_w 1 = 0$.

**Variance reduction.** The optimal baseline minimizing $\text{Var}((r-b)e_{ij})$ is $b^* = \frac{\mathbb{E}[r \cdot e_{ij}^2]}{\mathbb{E}[e_{ij}^2]}$, but in practice any reasonable estimate of $\mathbb{E}[r]$ suffices.

REINFORCE can be combined with backpropagation for networks with deterministic hidden layers and stochastic output layers.

## Worked Example

Consider a single Gaussian unit with one input and one weight. Input $x = 1$. Weight $w = 2$. Variance $\sigma^2 = 1$. The mean output is $\mu = wx = 2$.

The unit samples output $y = 2.5$ (above the mean by 0.5).

Suppose the reinforcement $r = 10$ (a good outcome) and baseline $b = 5$ (expected reward).

Eligibility: $e = \frac{y - \mu}{\sigma^2} x = \frac{2.5 - 2}{1} \cdot 1 = 0.5$

Update: $\Delta w = \alpha (r - b) e = 0.1 \cdot (10 - 5) \cdot 0.5 = 0.25$

The weight increases because:
1. The reinforcement exceeded baseline (good outcome)
2. The eligibility was positive (output was above mean)

The interpretation: the output was above what the weight would predict, and this led to good results, so increase the weight to make such outputs more likely.

If $y = 1.5$ (below mean) with the same $r = 10$:
$e = -0.5$, $\Delta w = -0.25$

The weight would decrease, making above-mean outputs more likely, because the below-mean output (though it led to good reward) should become less likely.

This shows how noise serves as exploration: deviations from the mean are reinforced or suppressed based on outcomes.

## Subtleties

REINFORCE is an unbiased gradient estimator, but it has high variance. Each update is based on a single sample, so estimates are noisy. Variance reduction through baselines is crucial for practical performance, but even with optimal baselines, convergence can be slow.

The algorithm is on-policy: it estimates the gradient of expected reward under the current policy's distribution. Off-policy learning (using data from other policies) requires importance sampling corrections that can increase variance further.

Temporal credit assignment is not addressed directly. For multi-step episodes, REINFORCE can use the total episode reward, but this provides the same reward signal for all actions in the episode. Extensions using eligibility traces or learned value functions (actor-critic methods) improve credit assignment.

The choice of stochastic unit distribution matters. Gaussian units work well for continuous action spaces. Bernoulli units (output 0 or 1 with probability determined by sigmoid of weighted input) work for discrete actions. The distribution determines the form of the eligibility.

## Critical Debates

REINFORCE versus value-based methods: REINFORCE directly optimizes expected reward through policy parameters. Value-based methods (Q-learning, TD) learn value functions and derive policies from them. Policy gradient methods are more flexible (can represent stochastic policies, work with continuous actions) but often have higher variance.

The baseline paradox: the baseline does not affect the expected gradient but dramatically affects variance. Choosing good baselines is crucial but not addressed by the theory. Actor-critic methods learn baselines as value functions.

Sample complexity: REINFORCE requires many samples because each gradient estimate is noisy. Modern policy gradient methods (PPO, A2C) use multiple techniques to reduce variance, but the fundamental sample inefficiency remains compared to value-based methods.

The deeper question is whether there are better gradient estimators. Reparameterization tricks (as used in VAEs) reduce variance for certain distributions by moving stochasticity to an auxiliary random variable. But for discrete actions, the log-derivative trick of REINFORCE remains the standard approach.

## Key Quotes

"These algorithms, called REINFORCE algorithms, are shown to make weight adjustments in a direction that lies along the gradient of expected reinforcement." (Abstract)

"The characteristic eligibility $e_{ij}$ is defined as the partial derivative of the logarithm of $g_i$ with respect to $w_{ij}$." (p. 232)

"On the average the weight updates lie along the gradient of expected reinforcement, and the equality holds when all learning rate parameters and reinforcement baseline parameters are identical." (Theorem 1)

## Citation

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
