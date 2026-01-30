# Residual Algorithms: Reinforcement Learning with Function Approximation (Baird, 1995)

## The Problem (Layperson)

Reinforcement learning algorithms like Q-learning are proven to find optimal policies when each state has its own entry in a lookup table. But real problems have too many states to enumerate. A robot navigating a building cannot store a separate value for every possible position, orientation, and velocity. It must generalize: nearby states should have similar values.

Function approximation, such as neural networks, enables this generalization. But a troubling discovery emerged: algorithms guaranteed to converge with lookup tables could diverge to infinity with function approximation. The guarantees evaporated precisely when they were most needed.

## What Didn't Work (Alternatives)

The direct approach simply replaced lookup tables with function approximation. Q-learning with a neural network: after observing transition $(s, a, r, s')$, use $r + \gamma \max_{a'} Q(s', a')$ as the target and train the network to match it. This worked spectacularly for TD-Gammon but had no convergence guarantee.

Theoretical analysis showed the problem was fundamental. Even with linear function approximation, the simplest possible generalization, Q-learning could diverge. The "deadly triad" of bootstrapping (using own predictions as targets), off-policy learning (learning about a different policy than is followed), and function approximation could cause weights to grow without bound.

Previous convergence proofs for lookup tables relied on each state being updated independently. Function approximation broke this independence: updating one state's value necessarily changed the values of other states through the shared weights.

## The Key Insight

The problem stemmed from treating $r + \gamma V(s')$ as a fixed target when it was actually a function of the same weights being trained. Gradient descent minimizes loss functions, but the "loss" in standard TD learning is not a true loss function because the target depends on the weights.

Baird proposed minimizing the Bellman residual directly:

$$E = \sum_s \left[ V(s) - r(s) - \gamma V(s') \right]^2$$

This is a true loss function: both sides of the Bellman equation depend on the weights, and gradient descent on $E$ is mathematically well-defined. Algorithms minimizing this "residual gradient" are guaranteed to converge.

However, pure residual gradient algorithms can be extremely slow. Baird introduced "residual algorithms" that interpolate between fast-but-unstable direct methods and slow-but-stable residual gradient methods, achieving both speed and convergence.

## The Method

The direct algorithm updates weights as if the TD target were fixed:

$$\Delta w = \alpha (r + \gamma V(s') - V(s)) \nabla_w V(s)$$

The residual gradient algorithm treats both $V(s)$ and $V(s')$ as functions of $w$:

$$\Delta w = \alpha (r + \gamma V(s') - V(s)) [\nabla_w V(s) - \gamma \nabla_w V(s')]$$

The residual algorithm combines them with a mixing parameter $\eta \in [0,1]$:

$$\Delta w = \eta \Delta w_{\text{residual gradient}} + (1-\eta) \Delta w_{\text{direct}}$$

With $\eta = 1$, the algorithm is pure residual gradient (guaranteed convergence, potentially slow). With $\eta = 0$, it is pure direct (potentially divergent, fast when stable). Intermediate values can achieve both stability and speed.

The parameter $\eta$ can be adapted automatically: compute both update directions and choose $\eta$ to maximize progress toward the residual gradient direction while remaining stable.

## The Result

The "star MDP" counterexample demonstrated divergence of direct algorithms with linear function approximation:

Six states with values parameterized by two weights. Each of five "outer" states transitions to a central state. With the right initialization, all weights and values diverge to infinity despite bounded rewards and a simple, linear function approximator.

This counterexample was pathological but illuminating: it showed that divergence was not a rare edge case but a structural property of the algorithm-architecture combination.

Residual algorithms solved this: they converged on the star MDP while retaining fast learning on problems where direct methods worked well. The adaptive $\eta$ automatically found the right balance.

## Worked Example

Consider the "hall" MDP: states $1 \to 2 \to 3 \to 4 \to 5$, with reward only at state 5. Discount $\gamma = 0.95$.

With a lookup table, TD learning propagates values backward: $V(5) = 1$, $V(4) = 0.95$, $V(3) = 0.90$, etc.

Now suppose states 4 and 5 share a weight: $V(4) = V(5) = w$.

Direct update at state 4 $\to$ 5:
$$\Delta w \propto (0 + 0.95 w - w) \cdot 1 = -0.05 w$$
Weight decreases toward 0.

Residual gradient update at state 4 $\to$ 5:
$$\Delta w \propto -0.05 w \cdot (1 - 0.95) = -0.0025 w$$
Also decreases, but 20x slower!

The residual gradient updates both $V(4)$ and $V(5)$ together, causing information to flow "both ways." This is more cautious but much slower when the direct method works fine.

Residual algorithms use $\eta$ to balance: on simple problems like the hall, use $\eta \approx 0$ (nearly direct). On problematic problems like the star, use higher $\eta$.

## Subtleties

The Bellman residual is zero only for the optimal value function. But minimizing residual does not guarantee finding the optimal policy: with limited function approximation capacity, the minimum-residual solution may not be optimal. The residual gradient algorithm finds a minimum of the residual, not necessarily the global minimum or the best approximation to the optimal value function.

For stochastic MDPs, the residual gradient algorithm requires two independent samples of the next state to obtain an unbiased gradient estimate. This is because the gradient involves products of random variables. With a model or with careful sample management, this is achievable; without, the algorithm must use biased estimates.

The deadly triad (off-policy + function approximation + bootstrapping) remains unsolved in full generality. Residual algorithms address the function approximation + bootstrapping part but still struggle with off-policy learning. Modern approaches like fitted Q-iteration and target networks provide additional stability without solving the fundamental problem.

Why did TD-Gammon work despite the deadly triad? Likely explanations include: on-policy learning (self-play), the particular architecture (not pure linear), and the problem structure (smooth value function). TD-Gammon's success was empirical, not guaranteed by theory.

## Critical Debates

Residual versus direct algorithms: The community largely adopted direct methods (DQN, etc.) with heuristic stabilization (target networks, experience replay) rather than the theoretically-grounded residual approach. Why? Residual algorithms require architectural constraints (differentiable function approximation) and can be slow. Practical success trumped theoretical guarantees.

Linear versus nonlinear function approximation: Baird's counterexamples used linear function approximation, the simplest case. Nonlinear approximators like neural networks are even less understood. Modern deep RL works spectacularly without theoretical foundations.

The role of the deadly triad: Subsequent research identified the three-way combination as the core problem. Removing any leg stabilizes learning: use on-policy methods (PPO), avoid bootstrapping (Monte Carlo), or use tabular representations. But all three provide practical advantages that make them hard to abandon.

Target networks, introduced in DQN, provide a different kind of stability: the target $r + \gamma V(s')$ uses a frozen copy of the network, updated periodically. This breaks the direct dependence of target on current weights, providing practical stability without the theoretical elegance of residual algorithms.

## Key Quotes

"A number of reinforcement learning algorithms have been guaranteed to converge to the optimal solution when used with lookup tables. It is shown, however, that these algorithms can easily become unstable when implemented directly with a general function-approximation system." (Abstract)

"The star problem showed an MDP rather than a Markov chain... if the solid Q-values start larger than the dotted Q-values, and the transition from state 6 to itself starts out as the largest of the solid Q-values, then all weights, Q-values, and values will diverge to infinity." (Errata)

"Direct algorithms can be fast but unstable, and residual gradient algorithms can be stable but slow." (p. 4)

## Citation

Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. In *Proceedings of the Twelfth International Conference on Machine Learning* (pp. 30-37). Morgan Kaufmann.
