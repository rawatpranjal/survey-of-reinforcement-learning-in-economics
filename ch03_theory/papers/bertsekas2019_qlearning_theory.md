# Bertsekas Q-Learning Theory Reference

**Source:** Bertsekas, D.P. (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.

This document extracts and organizes the Q-learning content from Chapters 4.3, 4.4, and 4.8 of Bertsekas's book, providing a self-contained reference for the theoretical foundations tested in our LQR convergence experiments.

---

## 1. Q-Factor Fundamentals (Section 4.3)

### Definition

For discounted problems, the optimal Q-factors are defined for all state-control pairs (i,u) with u in U(i):

```
Q*(i,u) = E[ g(i,u,j) + α·J*(j) | j ~ p_ij(u) ]
```

where:
- `g(i,u,j)` is the stage cost
- `α` is the discount factor
- `J*(j)` is the optimal cost-to-go from state j
- `p_ij(u)` is the transition probability

### Bellman Equation for Q-Factors

The optimal Q-factors satisfy:

```
Q*(i,u) = Σ_j p_ij(u)·[ g(i,u,j) + α·min_v Q*(j,v) ]     (Eq. 4.50)
```

and are the unique solution of this system.

### The F Operator

Define the operator F by:

```
(FQ)(i,u) = Σ_j p_ij(u)·[ g(i,u,j) + α·min_v Q(j,v) ]    (Eq. 4.51)
```

**Key Property (Proposition 4.3.5):** F is a contraction mapping with modulus α.

This means:
```
||FQ - FQ'||_∞ ≤ α ||Q - Q'||_∞
```

**Consequence:** The algorithm Q_{k+1} = F·Q_k converges to Q* from every starting point Q_0.

---

## 2. Contraction Property and Convergence (Section 4.3)

### Contraction Mapping Theorem

**Proposition 4.3.5:** For discounted problems with discount factor α in (0,1):

1. The operators T (Bellman) and T_μ (policy-specific Bellman) are contractions with modulus α
2. The Q-factor operator F is also a contraction with modulus α

### Convergence Rate

From contraction theory:
```
||J_k - J*||_∞ ≤ α^k ||J_0 - J*||_∞
```

This gives **linear convergence** with rate α.

For the Riccati operator in LQR, the contraction modulus is:
```
ρ = γ·a²·r² / (r + γ·b²·K*)²
```

which is strictly less than 1 for well-posed problems.

---

## 3. Q-Learning Algorithm (Section 4.8)

### Stochastic VI Formulation

Q-learning is a stochastic version of value iteration. An infinitely long sequence of state-control pairs {(i_k, u_k)} is generated. For each pair, a successor state j_k is sampled according to p_{i_k,j}(u_k).

### Update Rule

The Q-factor is updated using:

```
Q_{k+1}(i_k, u_k) = (1 - γ_k)·Q_k(i_k, u_k) + γ_k·(F_k·Q_k)(i_k, u_k)    (Eq. 4.52)
```

where:
```
(F_k·Q_k)(i_k, u_k) = g(i_k, u_k, j_k) + α·min_v Q_k(j_k, v)    (Eq. 4.53)
```

All other Q-factors remain unchanged:
```
Q_{k+1}(i, u) = Q_k(i, u)    for (i,u) ≠ (i_k, u_k)
```

### Convergence Conditions

**From Bertsekas (p. 56):**

> "To guarantee the convergence of the algorithm (4.52)-(4.53) to the optimal Q-factors, some conditions must be satisfied."

**Required conditions:**

1. **Exploration:** All state-control pairs (i,u) must be generated infinitely often within the sequence {(i_k, u_k)}

2. **Independent sampling:** Successor states j must be independently sampled at each occurrence of a given state-control pair

3. **Stepsize (Robbins-Monro conditions):**
   ```
   γ_k > 0,  Σ_{k=0}^∞ γ_k = ∞,  Σ_{k=0}^∞ γ_k² < ∞
   ```

   Example: γ_k = c₁/(k + c₂) for positive constants c₁, c₂

**Convergence proof:** Tsitsiklis [Tsi94], combining stochastic approximation theory with asynchronous DP convergence theory.

---

## 4. Drawbacks of Q-Learning (Section 4.8)

### From Bertsekas (p. 57):

> "In practice, Q-learning has some drawbacks, the most important of which is that the number of Q-factors/state-control pairs (i,u) may be excessive."

### Approximation Issues

When Q-factor approximation is introduced:

> "When Q-factor approximation is used, their behavior is very complex, their theoretical convergence properties are unclear, and there are no associated performance bounds in the literature." (p. 58)

---

## 5. Approximate VI Instability (Section 4.4)

### Example 4.4.1 - Divergence with Function Approximation

Bertsekas provides a 2-state example where approximate VI can diverge even with α < 1.

**Key insight (p. 24):**

> "The difficulty here is that the approximate VI mapping... is NOT a contraction (even though T itself is a contraction)."

For the specific example, divergence occurs when α > 5/6.

### No General Guarantees

> "There is no known general method to guarantee that the iterates r_k remain bounded... the instability illustrated with Example 4.4.1 is avoided."

---

## 6. Error Bounds for Approximate VI (Section 4.4)

### Per-Iteration Error Bound

If the approximation error per iteration satisfies:
```
|J̃_{k+1}(i) - (T·J̃_k)(i)| ≤ δ    for all i, k
```

Then asymptotically:

**Cost function error:**
```
||J* - J̃||_∞ ≤ δ/(1-α)
```

**Policy error:**
```
||J* - J_μ̃||_∞ ≤ 2δ/(1-α)²
```

---

## 7. Policy Iteration Properties (Section 4.5)

### Proposition 4.5.1: Monotonic Improvement

> "The exact PI algorithm generates an improving sequence of policies, i.e., J_{μ_{k+1}} ≤ J_{μ_k}, and terminates with an optimal policy."

### Proposition 4.5.2: Optimistic PI

For optimistic PI with m VI steps for policy evaluation:

> "J* ≤ J_{k+1} ≤ J_k for all k"

Special cases:
- m = 1: equivalent to VI
- m = ∞: equivalent to exact PI

### Quadratic Convergence

PI exhibits quadratic (Newton-like) convergence:
```
|e_{k+1}| ≈ C·|e_k|²
```

This is because PI can be viewed as Newton's method applied to the Bellman equation.

---

## 8. SARSA and On-Policy Methods (Section 4.8)

### SARSA Update

An extreme optimistic scheme using a single sample between policy updates:

1. Simulate transition (i_k → i_{k+1}) using u_k
2. Generate u_{k+1} by greedy policy (with ε-exploration)
3. Update parameter vector via gradient-like step

**From Bertsekas (p. 58):**

> "When Q-factor approximation is used, their behavior is very complex, their theoretical convergence properties are unclear, and there are no associated performance bounds in the literature."

---

## 9. Key Theoretical Results for LQR Validation

### Test 1: Contraction Property

**Prediction:** F is a contraction with modulus ρ
**LQR form:** ρ = γ·a²·r²/(r + γ·b²·K*)²
**Validation:** Verify ||Q_{k+1} - Q*|| / ||Q_k - Q*|| ≈ ρ for VI iterations

### Test 2: Linear Convergence Rate

**Prediction:** log(error) decreases linearly with slope log(ρ)
**Validation:** Plot log₁₀(error) vs iteration, fit linear regression

### Test 3: Stepsize Requirements

**Prediction:**
- Constant stepsize → oscillation around K*
- Diminishing stepsize (Robbins-Monro) → asymptotic convergence

**Validation:** Compare Q-learning with both schedules

### Test 4: Exploration Requirement

**Prediction:** All (s,a) pairs must be visited infinitely often
**Validation:** Track visit counts, verify adequate coverage

### Test 5: V(s) Monotonicity (Observed Phenomenon)

**Observation:** K = V(s)/s² only decreases during learning
**Explanation:**
- V(s) = min_u Q(s,u)
- The min operator filters out upward noise in individual Q(s,a) values
- Most updates affect suboptimal actions, not the greedy action
- Result: V(s) decreases monotonically even though Q(s,a) values fluctuate

---

## 10. Key References

- **[Wat89]** Watkins, C.J.C.H. (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge.
- **[Tsi94]** Tsitsiklis, J.N. (1994). Asynchronous stochastic approximation and Q-learning. *Machine Learning*, 16:185-202.
- **[BeT96]** Bertsekas, D.P. and Tsitsiklis, J.N. (1996). *Neuro-Dynamic Programming*. Athena Scientific.
- **[Ber12]** Bertsekas, D.P. (2012). *Dynamic Programming and Optimal Control, Vol. II*, 4th ed. Athena Scientific.
- **[Ber82]** Bertsekas, D.P. (1982). Distributed dynamic programming. *IEEE Trans. Automatic Control*, 27:610-616.
- **[BeY10], [BeY12], [YuB13a]** Bertsekas and Yu papers on asynchronous optimistic PI and Q-learning.

---

## 11. Summary Table

| Result | Source | Bertsekas Reference |
|--------|--------|---------------------|
| F contraction with modulus α | Prop 4.3.5 | p. 18-21 |
| Q_{k+1} = FQ_k converges | Prop 4.3.5 | p. 54 |
| Q-learning convergence conditions | Section 4.8 | p. 56 |
| Stepsize: Σγ_k = ∞, Σγ_k² < ∞ | Section 4.8 | p. 56 |
| Approximate VI may diverge | Example 4.4.1 | p. 22-25 |
| Error bound δ/(1-α) | Section 4.4 | p. 23 |
| PI monotonic improvement | Prop 4.5.1 | p. 27 |
| SARSA: "no performance bounds" | Section 4.8 | p. 58 |

---

*Document created: 2026-01-29*
*For use with: ch02_planning_learning/sims/lqr_convergence.py*
