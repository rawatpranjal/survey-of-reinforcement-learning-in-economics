# Q-Learning (Watkins & Dayan, 1992)

## The Problem

Model-free optimal control requires learning the optimal action-value function $Q^*$ without access to the transition dynamics $P(s'|s,a)$ or reward function $R(s,a)$. Dynamic programming computes optimal policies via the Bellman optimality equation, but this requires a complete model of the MDP. Q-learning achieves the same end using only sampled transitions $(s, a, r, s')$.

The goal is to find a policy $\pi^*$ maximizing expected discounted return $\mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t]$ from any initial state, where $\gamma \in [0,1)$ is the discount factor. With known dynamics, value iteration solves this by repeatedly applying the Bellman operator. Without a model, the agent must learn from experience alone.

## What Didn't Work (Alternatives)

Model-based approaches required learning transition probabilities $P(s'|s,a)$ and expected rewards $R(s,a)$ from experience, then applying dynamic programming. This was computationally expensive and introduced approximation errors at both stages: errors in the learned model propagated into errors in the computed policy.

Temporal difference methods like TD(0) learned value functions $V(s)$, but converting these to actions still required knowing transition probabilities. Given $V(s')$ for all successor states $s'$, the agent needed to know which action led to which successor.

Policy search methods learned policies directly but typically required many samples and could get trapped in local optima. They did not exploit the Bellman optimality structure.

## The Key Insight

Instead of learning $V(s)$, learn $Q(s,a)$: the expected return from taking action $a$ in state $s$ and then behaving optimally. The optimal policy is then simply $\pi^*(s) = \arg\max_a Q^*(s,a)$. No model is needed to act: the agent just picks the action with highest Q-value.

Q-values can be learned incrementally by temporal difference updates:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

This update moves $Q(s,a)$ toward the observed reward plus the best discounted future value. The key insight is that the Bellman optimality equation provides a self-consistency condition: at optimality, $Q^*(s,a) = r + \gamma \max_{a'} Q^*(s',a')$. The learning rule pushes Q-values toward satisfying this condition.

## The Method

Q-learning maintains a table of Q-values, one for each state-action pair. After experiencing transition $(s, a, r, s')$:

1. Compute TD target: $y = r + \gamma \max_{a'} Q(s', a')$
2. Compute TD error: $\delta = y - Q(s,a)$
3. Update: $Q(s,a) \leftarrow Q(s,a) + \alpha \delta$

The Bellman optimality equation provides the fixed-point condition:

$$Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s, a\right]$$

The Bellman operator $\mathcal{T}$ defined by $(\mathcal{T}Q)(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a') | s,a]$ is a contraction in the supremum norm with modulus $\gamma$:

$$\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

This ensures a unique fixed point $Q^* = \mathcal{T}Q^*$.

**Convergence theorem.** Q-learning converges to $Q^*$ with probability 1 under:
1. The state and action spaces are finite
2. $\sum_n \alpha_n(s,a) = \infty$ and $\sum_n \alpha_n^2(s,a) < \infty$ for all $(s,a)$
3. $|\mathcal{R}| \leq R_{\max}$ (bounded rewards)

**The action-replay process (ARP).** The proof constructs an auxiliary MDP where at each step, one of two things happens: (i) with some probability, replay a stored transition by sampling from the history of observed $(s,a,r,s')$ tuples, or (ii) otherwise, act according to a fixed policy. The key insight is that $Q_t$ (the Q-values at step $t$ of learning) equals the optimal Q-function for a particular ARP. As $t \to \infty$, the ARP's transition distribution converges to the true MDP's, so $Q_t \to Q^*$.

## The Result

Q-learning converges to optimal Q-values with probability 1 under the stated conditions. This was a major theoretical result: a model-free algorithm with guaranteed optimality.

The algorithm is off-policy: it learns about the optimal policy regardless of which policy generates the experience, as long as all state-action pairs are sufficiently explored. This separates exploration (how data is collected) from learning (what is learned from the data).

Q-learning is also incremental and computationally simple. Each update requires only the current transition tuple and takes O(|A|) time for the max operation. No model is stored or manipulated.

## Worked Example

Consider a gridworld with 4 states arranged in a line: S1 - S2 - S3 - S4. The agent can move left or right. Reaching S4 gives reward +1; all other rewards are 0. Discount $\gamma = 0.9$.

Initially, all $Q(s,a) = 0$.

Episode 1: Start at S2. Move right to S3 (r=0). Move right to S4 (r=1). Episode ends.

Update from S3 $\to$ S4:
$$Q(S3, R) \leftarrow 0 + 0.5[1 + 0.9 \cdot 0 - 0] = 0.5$$

Now $Q(S3, R) = 0.5$. This encodes: from S3, going right is worth 0.5.

Episode 2: Start at S1. Move right to S2. Update from S1 $\to$ S2:
$$Q(S1, R) \leftarrow 0 + 0.5[0 + 0.9 \cdot 0 - 0] = 0$$
(S2's best Q is still 0)

Move right to S3. Update from S2 $\to$ S3:
$$Q(S2, R) \leftarrow 0 + 0.5[0 + 0.9 \cdot 0.5 - 0] = 0.225$$

Move right to S4. Update:
$$Q(S3, R) \leftarrow 0.5 + 0.5[1 + 0 - 0.5] = 0.75$$

Now knowledge propagates backward. After many episodes, Q-values converge:
- $Q^*(S3, R) = 1$ (immediate reward)
- $Q^*(S2, R) = 0.9$ (discounted)
- $Q^*(S1, R) = 0.81$ (doubly discounted)

## Subtleties

The convergence proof requires visiting all state-action pairs infinitely often, but it says nothing about how quickly convergence occurs. In practice, Q-learning can be very slow when exploration is inefficient. The exploration strategy (e.g., $\varepsilon$-greedy, softmax, UCB) dramatically affects practical performance but is not part of Q-learning itself.

Q-learning is off-policy because it uses $\max_{a'} Q(s',a')$ regardless of which action was actually taken from $s'$. This means the agent learns about the optimal policy even while following an exploratory policy. Contrast with SARSA, which uses $Q(s', a')$ for the action $a'$ actually taken, making it on-policy.

The look-up table representation limits Q-learning to small, discrete state spaces. With continuous states, function approximation is needed. But the convergence guarantee does not extend to function approximation. Baird (1995) showed that Q-learning can diverge with linear function approximation. This "deadly triad" problem (off-policy + function approximation + bootstrapping) remains a challenge.

Q-learning's max operation introduces a positive bias. If Q-values are noisy estimates, $\max_a Q(s,a)$ will overestimate the true value because max selects the noisiest high values. Double Q-learning addresses this by using one Q-function to select actions and another to evaluate them.

## Critical Debates

Q-learning versus SARSA: Off-policy learning (Q-learning) converges to optimal regardless of behavior policy; on-policy learning (SARSA) converges to the policy being followed. Neither dominates: Q-learning is better when the behavior policy is suboptimal, SARSA is safer with function approximation.

Model-free versus model-based: Q-learning is sample-efficient in that it uses each sample to improve Q-estimates. But it discards information: the sample $(s,a,r,s')$ is used once and forgotten. Model-based methods store and reuse samples, potentially achieving better data efficiency at higher computational cost.

Tabular Q-learning versus deep Q-learning: The 1992 theory applies to look-up tables. DQN (2015) combines Q-learning with deep neural networks and experience replay. This works spectacularly in practice but has no convergence guarantee. Understanding why it works remains an open question.

The deepest question is sample complexity: how many samples are needed to learn near-optimal Q-values? Recent theoretical work has characterized minimax-optimal sample complexity, but practical algorithms often exceed these bounds substantially.

## Key Quotes

"Q-learning amounts to an incremental method for dynamic programming which imposes limited computational demands." (p. 279)

"It provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains." (p. 279)

"We show that Q-learning converges to the optimum action-values with probability 1 so long as all actions are repeatedly sampled in all states and the action-values are represented discretely." (Abstract)

## Citation

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
