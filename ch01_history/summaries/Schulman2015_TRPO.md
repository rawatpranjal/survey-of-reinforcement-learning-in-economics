# Trust Region Policy Optimization (Schulman et al., 2015)

## The Problem (Layperson)

How do you update a policy so that it is guaranteed to improve, or at least not get worse? In policy optimization, each update changes the policy's behavior. A small change to the parameters might cause a large change in actions, leading to catastrophic performance collapse. Conversely, being too conservative means learning takes forever.

Standard policy gradient methods lack reliable step sizes. The gradient tells you which direction improves expected reward, but not how far to go in that direction. Take too large a step and performance crashes. Take too small a step and learning stalls. The right step size varies across problems and even within a single problem as training progresses.

## What Didn't Work (Alternatives)

Vanilla policy gradient methods used fixed learning rates that required extensive tuning. Different problems needed different rates; even within one problem, early training might need different rates than late training. This made deployment unreliable.

Natural policy gradient methods used the Fisher information matrix to scale gradients, providing invariance to parameterization. But the resulting "natural gradient" still required a step size, which had to be tuned manually. Too large and it diverged; too small and it crawled.

Black-box optimization methods like CEM and CMA avoided gradients entirely, treating the problem as direct optimization. These worked surprisingly well on small problems but scaled poorly with parameter dimension due to poor sample complexity.

## The Key Insight

There exists a surrogate objective that lower-bounds the true expected return. Maximizing this surrogate guarantees policy improvement. The surrogate is:

$$L_{\pi_{old}}(\pi) - C \cdot D_{KL}^{max}(\pi_{old}, \pi)$$

where $L$ is the expected advantage of the new policy evaluated under the old policy's state distribution, and the penalty term bounds how much the policies can differ.

The key theoretical result shows that if you constrain the KL divergence between old and new policies, the improvement in the surrogate function translates to improvement in the true objective. This provides a principled way to choose step sizes: take the largest step that keeps policies close enough to maintain the bound.

## The Method

TRPO approximately solves:
$$\max_\theta L_{\theta_{old}}(\theta) \quad \text{subject to} \quad \bar{D}_{KL}(\theta_{old}, \theta) \leq \delta$$

where $\bar{D}_{KL}$ is the average KL divergence over states (a tractable approximation to the maximum).

The algorithm proceeds in three steps:

1. Estimate the policy gradient and Fisher information matrix from sampled trajectories

2. Compute the search direction using conjugate gradient:
   - Approximate constraint as quadratic: $D_{KL} \approx \frac{1}{2}(\theta - \theta_{old})^T A (\theta - \theta_{old})$
   - Solve $Ax = g$ where $g$ is the policy gradient and $A$ is the Fisher matrix
   - This gives the natural gradient direction

3. Line search to satisfy constraint:
   - Compute maximum step size: $\beta = \sqrt{2\delta / x^T A x}$
   - Shrink step until surrogate objective improves and constraint is satisfied

Crucially, the Fisher-vector products $Av$ can be computed without forming the full matrix $A$, making the algorithm scalable to neural networks with thousands of parameters.

## The Result

TRPO learned locomotion gaits for simulated robots (swimming, hopping, walking) from scratch, using generic neural network policies and simple reward functions. No hand-designed gaits or domain knowledge were required.

On continuous control benchmarks, TRPO outperformed CEM, CMA, and vanilla policy gradient methods. It reliably made progress where other methods stalled or diverged.

On Atari games with image input, TRPO achieved competitive results with DQN despite being a policy gradient method. This demonstrated that the approach generalized across different problem types.

The key empirical finding: by enforcing the KL constraint rather than using a fixed penalty coefficient, TRPO achieved monotonic improvement across diverse problems without hyperparameter tuning. The constraint bound $\delta = 0.01$ worked well universally.

## Worked Example

Consider learning to balance a pole on a cart. Initial policy is random.

Iteration 1:
- Collect trajectories by running the current policy
- Most trajectories end quickly (pole falls)
- Estimate advantage: actions that briefly delayed falling have positive advantage
- Compute natural gradient direction using conjugate gradient
- Line search finds step size that improves surrogate while satisfying $D_{KL} \leq 0.01$
- Update policy parameters

Iteration 2:
- New policy keeps pole balanced slightly longer on average
- Better trajectories provide more informative advantages
- Natural gradient points toward even better policies
- KL constraint prevents overshooting to a worse policy

After 100 iterations:
- Policy reliably balances the pole
- Each update made small, guaranteed improvements
- No manual learning rate tuning was required

The KL constraint acted as an automatic learning rate. When the surrogate landscape was steep, TRPO took small parameter steps (large policy changes per parameter unit). When flat, it took larger parameter steps (small policy changes per parameter unit).

## Subtleties

The average KL divergence is used instead of the theoretically-justified maximum KL. The maximum is intractable (requires checking all states), while the average can be estimated from samples. Empirically, this approximation works well.

The surrogate objective uses importance sampling to evaluate the new policy using data from the old policy. This becomes inaccurate when policies differ substantially, which is another reason to constrain their divergence.

Fisher-vector products enable scalability. Computing the full Fisher matrix requires $O(n^2)$ space and $O(n^3)$ time for matrix inversion, where $n$ is the number of parameters. Conjugate gradient with Fisher-vector products requires only $O(n)$ space and about 10 matrix-vector products per iteration.

The line search is necessary because the quadratic approximation to KL divergence is only accurate near the current parameters. The search ensures the actual (not approximated) constraint is satisfied.

TRPO is on-policy: it requires fresh trajectories from the current policy for each update. This limits sample efficiency compared to off-policy methods like DQN, but provides stability guarantees.

## Critical Debates

TRPO versus natural policy gradient: The key difference is using a constraint rather than a penalty. With a fixed penalty coefficient, different problems require different values. With a constraint, a universal bound ($\delta = 0.01$) works across problems. This is not just a technical detail but a fundamental shift in how to think about policy optimization.

Sample efficiency: TRPO requires many samples per update because it is on-policy. Each update discards previous experience. Methods like off-policy actor-critic try to reuse old data, but this complicates the theory and can introduce instability.

Trust region versus clipping: PPO (Proximal Policy Optimization) later showed that simple clipping of the probability ratio could achieve similar effects with less computational cost. Whether the full machinery of TRPO is necessary remains debated.

The deeper question: is monotonic improvement the right goal? In practice, some regression might be acceptable if it enables faster eventual progress. Methods that explicitly explore and tolerate temporary performance drops might learn more efficiently.

## Key Quotes

"We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement." (Abstract)

"In practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small. One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint." (Section 4)

"Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input." (Abstract)

## Citation

Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 1889-1897).
