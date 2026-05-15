# Notes for Conclusion — Argument Step by Step

## Paragraph 1: Structure enables identification

1. Economics makes progress by imposing structure on problems that are otherwise underdetermined.
2. What structure does: it constrains the set of possible explanations, converting a search over an infinite hypothesis space into estimation of finitely many parameters.
3. Examples of structure types: exclusion restrictions identify causal effects, functional form assumptions on utility and production enable closed-form solutions, distributional assumptions on unobservables permit likelihood-based estimation.
4. When these assumptions match the data-generating process, structural objects (preferences, technology, beliefs) can be recovered.
5. When they do not match, the result is misspecification, and estimates may be biased.
6. The choice of which assumptions to impose is always specific to the problem.

## Paragraph 2: DP and RL are solution methods for dynamic optimization

1. Dynamic programming characterizes optimal behavior through the Bellman equation: given a transition model and reward function, fixed-point iteration recovers optimal policies.
2. DP imposes no parametric restrictions on transitions or rewards, but it requires enumerating the state space.
3. The cost of enumeration grows exponentially in the number of state variables (curse of dimensionality).
4. RL replaces enumeration with sampling: it estimates value functions from simulated or observed transitions without constructing the full state-action table.
5. This substitution trades convergence rate (geometric for DP, sublinear for RL) for scalability to high-dimensional problems.
6. The cost of this generality is instability: bootstrapping, off-policy learning, and function approximation interact to produce divergence in settings where any two of the three are safe.

## Paragraph 3: Bidirectional benefits, with caveats

1. RL methods can solve economic models in state spaces where enumeration is infeasible, extending the reach of structural analysis to high-dimensional, continuous, and multi-agent settings.
2. In the other direction, economic structure constrains what RL must learn.
3. Structural assumptions on demand, preferences, or equilibrium behavior reduce the hypothesis space that an RL algorithm must search, and this reduction is quantifiable: in dynamic pricing, the gap between structured and unstructured regret rates is not a constant factor but a complexity-class separation (√T vs log T).
4. However, these assumptions are codifications of domain knowledge, not free parameters. They must be assessed problem by problem, and they carry the same misspecification risk as any structural model.

## Paragraph 4: Limitations and current state

1. The convergence theory for deep RL remains incomplete. For economists, this means that RL-based structural estimates lack the finite-sample guarantees that maximum likelihood or dynamic programming provide under correct specification.
2. RL requires either a known transition model or a simulator, and the fidelity of the simulator bounds the quality of the learned policy.
3. Observational data introduces additional identification problems: when unobserved confounders affect both actions and transitions, standard off-policy estimators are biased, and correcting this bias requires causal structure assumptions that may not hold.
4. Results are sensitive to hyperparameters, random seeds, and implementation details.
5. These barriers are real but identifiable. Deployed systems in dispatch, cooling, and pricing demonstrate that the methods surveyed here produce measurable gains in settings where the transition model can be learned or simulated with sufficient accuracy.
