# Conditional Logit Analysis of Qualitative Choice Behavior - Implementation Summary

## Citation
```bibtex
@incollection{mcfadden1974conditional,
  title={Conditional logit analysis of qualitative choice behavior},
  author={McFadden, Daniel},
  booktitle={Frontiers in Econometrics},
  editor={Zarembka, Paul},
  pages={105--142},
  year={1974},
  publisher={Academic Press}
}
```

## Core Contribution

McFadden established the theoretical foundations for discrete choice modeling under random utility maximization (RUM). The conditional logit model arises when unobserved utility components follow independent Type I extreme value (Gumbel) distributions. This work provides the econometric foundation for preference learning: the Bradley-Terry model used in RLHF is a special case of conditional logit with two alternatives.

The key insight is that choice probabilities take the logit form if and only if the random utility errors are Gumbel-distributed, and this functional form enables tractable maximum likelihood estimation with well-understood statistical properties.

## Key Equations

### Equation 1: Random Utility Model
$$
U_{ij} = V_{ij} + \varepsilon_{ij}
$$
**Notation:**
- $U_{ij}$: total utility of alternative $j$ for decision-maker $i$
- $V_{ij}$: deterministic (observed) utility component
- $\varepsilon_{ij}$: random (unobserved) utility component

**Implementation notes:** In RLHF context, $V_{ij} = r(x_i, y_j)$ is the reward.

### Equation 2: Choice Probability (Conditional Logit)
$$
P(y_j | x_i, \mathcal{C}_i) = \frac{\exp(V_{ij})}{\sum_{k \in \mathcal{C}_i} \exp(V_{ik})}
$$
**Notation:**
- $\mathcal{C}_i$: choice set for decision-maker $i$
- This assumes $\varepsilon_{ij} \sim \text{Gumbel}(0, 1)$ i.i.d.

**Implementation notes:** This is the softmax function. For pairwise choices ($|\mathcal{C}| = 2$), reduces to sigmoid.

### Equation 3: Bradley-Terry as Special Case
$$
P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))
$$
**Implementation notes:** The Bradley-Terry model is conditional logit with binary choice sets.

### Equation 4: Log-Likelihood Function
$$
\ell(\theta) = \sum_{i=1}^N \sum_{j \in \mathcal{C}_i} d_{ij} \log P(y_j | x_i, \mathcal{C}_i; \theta)
$$
**Notation:**
- $d_{ij} = 1$ if decision-maker $i$ chose alternative $j$, else 0
- $\theta$: parameters of $V_{ij}(\theta)$

**Implementation notes:** Cross-entropy loss in ML terminology.

### Equation 5: Score Function (Gradient)
$$
\frac{\partial \ell}{\partial \theta} = \sum_{i=1}^N \sum_{j \in \mathcal{C}_i} (d_{ij} - P_{ij}) \frac{\partial V_{ij}}{\partial \theta}
$$
**Implementation notes:** Gradient is difference between observed and predicted choices, weighted by feature gradient.

### Equation 6: Fisher Information Matrix
$$
\mathcal{I}(\theta) = -\mathbb{E}\left[\frac{\partial^2 \ell}{\partial \theta \partial \theta'}\right] = \sum_{i=1}^N \sum_{j \in \mathcal{C}_i} P_{ij}(1 - P_{ij}) \frac{\partial V_{ij}}{\partial \theta} \frac{\partial V_{ij}}{\partial \theta'}
$$
**Implementation notes:** Used for computing standard errors: $\text{Var}(\hat{\theta}) \approx \mathcal{I}^{-1}$.

### Equation 7: Independence of Irrelevant Alternatives (IIA)
$$
\frac{P(y_j | x)}{P(y_k | x)} = \frac{\exp(V_j)}{\exp(V_k)} = \exp(V_j - V_k)
$$
**Implementation notes:** Ratio of choice probabilities depends only on $j$ and $k$, not other alternatives. This is both a feature (tractability) and a limitation (may not hold empirically).

## Testable Claims

### Claim 1: MLE Consistency
**Statement:** $\hat{\theta}_{MLE} \xrightarrow{p} \theta^*$ as $N \to \infty$ under standard regularity conditions.
**Validation approach:** Generate synthetic data from known $\theta^*$, estimate $\hat{\theta}$ for increasing $N$.
**Expected result:** $\|\hat{\theta} - \theta^*\| = O(1/\sqrt{N})$.

### Claim 2: Asymptotic Normality
**Statement:** $\sqrt{N}(\hat{\theta} - \theta^*) \xrightarrow{d} \mathcal{N}(0, \mathcal{I}^{-1})$.
**Validation approach:** Repeat estimation across many samples, compare empirical distribution to theoretical.
**Expected result:** Empirical covariance matrix within 10% of $\mathcal{I}^{-1}/N$.

### Claim 3: Fisher Information Bound
**Statement:** MLE achieves the Cramer-Rao lower bound asymptotically.
**Validation approach:** Compute empirical variance of $\hat{\theta}$ and compare to $\mathcal{I}^{-1}/N$.
**Expected result:** Ratio of empirical to theoretical variance $\to 1$ as $N \to \infty$.

### Claim 4: IIA Property Holds
**Statement:** Adding irrelevant alternatives does not change relative choice probabilities.
**Validation approach:** Estimate model on subset of alternatives, predict on full set, check ratio consistency.
**Expected result:** Predicted ratios match empirical ratios (if data is truly logit-generated).

### Claim 5: Scale Identification
**Statement:** Only utility differences are identified; absolute scale requires normalization.
**Validation approach:** Estimate with and without scale normalization, compare predictions.
**Expected result:** Predictions identical despite different parameter magnitudes.

## Algorithm Pseudocode

```
Algorithm: Maximum Likelihood Estimation for Conditional Logit
Input:
  - Choice data {(x_i, C_i, y_i^*)} where y_i^* is chosen alternative
  - Feature function φ(x, y) → R^d
  - Linear utility V(x, y; θ) = θ^T φ(x, y)

Output: Parameter estimate θ̂, standard errors

1. Initialize θ ← 0
2. For iteration t = 1, ..., max_iter:
3.   # Compute choice probabilities
4.   For each observation i:
5.     For each alternative j ∈ C_i:
6.       V_ij = θ^T φ(x_i, y_j)
7.     P_i = softmax({V_ij : j ∈ C_i})
8.
9.   # Compute gradient
10.  grad = Σ_i Σ_j (d_ij - P_ij) φ(x_i, y_j)
11.
12.  # Compute Hessian (optional, for Newton's method)
13.  H = -Σ_i Σ_j P_ij(1 - P_ij) φ(x_i, y_j) φ(x_i, y_j)^T
14.
15.  # Update (Newton step or gradient ascent)
16.  θ ← θ - H^{-1} grad  # or θ ← θ + α * grad
17.
18.  # Check convergence
19.  If ||grad|| < tol: break

20. # Compute standard errors
21. I = -H  # Fisher information at θ̂
22. se = sqrt(diag(I^{-1}))
23. Return θ̂, se
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Convergence tolerance | tol | 1e-6 | [1e-8, 1e-4] | For gradient norm |
| Max iterations | - | 100 | [50, 500] | Newton usually converges fast |
| Learning rate (if GD) | $\alpha$ | 0.1 | [0.01, 1.0] | Only if not using Newton |
| Scale normalization | - | $\sigma = 1$ | - | Fix error variance to 1 |

## Connections to Other Papers
- **Rafailov et al. 2023**: Bradley-Terry preference model is binary conditional logit
- **Christiano et al. 2017**: Reward model training uses conditional logit likelihood
- **Korbak et al. 2022**: Posterior policy has softmax form (logit structure)
- **Train 2009**: Modern treatment of discrete choice with extensions (mixed logit, nested logit)

## Simulation Validation Checklist
- [ ] Generate data from known parameters, recover via MLE
- [ ] Verify asymptotic normality of estimator
- [ ] Confirm standard errors match empirical variance
- [ ] Test IIA property with synthetic data
- [ ] Compare gradient descent vs Newton convergence speed
- [ ] Validate equivalence of binary logit and Bradley-Terry
