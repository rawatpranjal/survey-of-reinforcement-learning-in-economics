# Chapter 02 Rewrite: Planning and Learning for Economists

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite Chapter 02 as a narrative-driven introduction to RL theory for PhD economists, framing RL not as "magic neural networks" but as asymptotic approximations of the Bellman fixed-point operator under uncertainty and computational constraints.

**Architecture:** Six sections that build from economists' existing DP knowledge (Stokey-Lucas-Prescott, Bertsekas) toward modern RL, with each section motivated by a limitation of standard DP and resolved by a specific theoretical contribution. Math is essential but narrative drives structure. No proof-heavy presentation; let formal statements speak through interpretation.

**Tech Stack:** LaTeX with natbib, amsmath/amsthm. Output: `ch02_planning_learning/tex/planning_learning_v3.tex`

---

## Section Structure Overview

| Section | Title | Economist's Question | Key Resolution |
|---------|-------|---------------------|----------------|
| 1 | The Newton Connection | Why does PI converge faster than VI? | PI = Newton's method on Bellman operator |
| 2 | Stochastic Approximation | Can we replace $\mathbb{E}$ with one sample? | Robbins-Monro theory; Q-learning converges w.p.1 |
| 3 | The Deadly Triad | Why does regression inside Bellman loops diverge? | Projection geometry; on-policy fixes it |
| 4 | Policy Gradient | Can we optimize policy directly without $V$? | Policy gradient theorem; gradient domination |
| 5 | Actor-Critic | How to combine low-bias value learning with low-variance policy gradient? | Two-timescale stochastic approximation |
| 6 | Bridging Results | What are the error bounds and complexity guarantees? | Singh-Yee bounds; Kearns sparse sampling |

---

## Task 1: Create New Chapter File with Header and Introduction

**Files:**
- Create: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Step 1: Write the chapter header and introduction**

Write the LaTeX file opening with:
- Comment block identifying chapter purpose and primary references
- Introduction subsection (2-3 paragraphs) that frames the chapter for economists
- Opening paragraph: "Economists solve dynamic programs. RL solves them too, but under uncertainty about the model and with computational constraints that preclude exact solutions."
- Key framing: RL algorithms are asymptotic approximations of the Bellman fixed-point operator
- Preview table summarizing the six sections and their key insights

**Step 2: Verify LaTeX compiles**

Run: `cd /Users/pranjal/Code/rl/docs && pdflatex -shell-escape -jobname=ch02_test "\def\chapterfile{../ch02_planning_learning/tex/planning_learning_v3}\input{compile_chapter}"`
Expected: PDF generates without errors

**Step 3: Commit**

```bash
git add ch02_planning_learning/tex/planning_learning_v3.tex
git commit -m "feat(ch02): begin narrative rewrite with introduction"
```

---

## Task 2: Section 1 — DP Foundations: The Newton Connection

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
Economists are comfortable with VI as fixed-point iteration ($V_{k+1} = TV_k$) and PI as "faster." But why is PI faster? This section formalizes PI as Newton's method, explaining its quadratic convergence.

**Step 1: Write Section 1 content**

Structure:
1. **Opening paragraph:** Economists know VI and PI from Stokey-Lucas-Prescott. VI applies the contraction $T$ repeatedly; PI solves the linearized problem exactly. The difference is Newton vs. fixed-point iteration.

2. **The Bellman Operator as Nonlinear System:** Present $T$ as a nonlinear operator. Value iteration is Picard iteration (linear convergence $\gamma^k$). Policy iteration linearizes at each step.

3. **Puterman-Brumelle (1979) Insight:** At current estimate $\tilde{J}$, the greedy policy $\tilde{\pi}$ defines a linear operator $T^{\tilde{\pi}}$ that is a supporting hyperplane to $T$:
   - $T^{\tilde{\pi}} \tilde{J} = T\tilde{J}$ (tangency at current point)
   - $T^{\tilde{\pi}} J \leq TJ$ for all $J$ (below the nonlinear operator everywhere)

   Policy evaluation solves $J = T^{\tilde{\pi}} J$ exactly—the Newton step.

4. **Convergence Rates Table:** Show iteration counts for VI vs PI at $\gamma = 0.90, 0.95, 0.99$ to achieve 100× error reduction. VI: 44/90/459 iterations. PI: typically 5-10.

5. **Bertsekas Generalization:** The Newton interpretation extends to any monotone contractive operator. This is why the same algorithmic structure (evaluate + improve) works across RL, MPC, and optimal stopping.

6. **Footnote:** Reference Kleinman (1968) for Riccati equations and Pollatschek-Avi-Itzhak (1969) for stochastic games as early instances.

**Key citations:** `\citet{puterman1979}`, `\citet{bertsekas2022newton}`, `\citet{bertsekas2018}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 1 — Newton connection"
```

---

## Task 3: Section 2 — Value Learning: Stochastic Approximation

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
In structural estimation, computing $\mathbb{E}[V(s')]$ requires knowing the transition matrix $P$ or integrating over a high-dimensional distribution. Can we replace the expectation with a single sample and still converge?

**Step 1: Write Section 2 content**

Structure:
1. **Opening:** The Bellman operator $TV = r + \gamma P V$ requires the expectation $\mathbb{E}_{s' \sim P}[V(s')]$. Model-free RL replaces this with a single observed transition. This is stochastic approximation.

2. **Robbins-Monro Framework:** Present the general stochastic approximation result. If we want to find $x^*$ such that $g(x^*) = 0$, and we only observe noisy samples $g(x) + \epsilon$, the iteration $x_{t+1} = x_t - \alpha_t [g(x_t) + \epsilon_t]$ converges under:
   - $\sum_t \alpha_t = \infty$ (sufficient exploration)
   - $\sum_t \alpha_t^2 < \infty$ (diminishing noise)

3. **Q-Learning as Stochastic Bellman Iteration:** The Q-learning update $Q(s,a) \leftarrow Q(s,a) + \alpha_t [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ is Robbins-Monro applied to the Q-factor Bellman operator. The target $r + \gamma \max_{a'} Q(s',a')$ is a noisy sample of $(FQ)(s,a)$.

4. **Jaakkola et al. (1994) Theorem:** State the convergence result. Under (i) every $(s,a)$ visited infinitely often, (ii) Robbins-Monro step sizes, Q-learning converges to $Q^*$ w.p. 1.

5. **SARSA: On-Policy Variant:** SARSA replaces $\max_{a'}$ with the action actually taken: $Q(s,a) \leftarrow Q(s,a) + \alpha_t [r + \gamma Q(s',a') - Q(s,a)]$. This solves the fixed point for the behavior policy $\pi$, not the optimal policy. Singh et al. (2000) prove convergence.

6. **Interpretation for Economists:** Q-learning is "noisy value iteration." SARSA is "noisy policy evaluation." The Robbins-Monro conditions ensure the noise averages out faster than the signal decays.

7. **Rollout and AlphaZero:** Bertsekas (2020) proves that rollout—one step of greedy improvement from base policy $\mu$—is guaranteed to improve: $V^{\tilde{\pi}}(s) \leq V^\mu(s)$. AlphaZero chains many such improvements. Longer lookahead = better Newton approximation.

**Key citations:** `\citet{jaakkola1994}`, `\citet{WatkinsDayan1992}`, `\citet{singh2000}`, `\citet{bertsekas2020rollout}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 2 — stochastic approximation"
```

---

## Task 4: Section 3 — The Deadly Triad: When Geometry Breaks

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
For continuous state spaces, economists use interpolation or regression (Chebyshev polynomials, splines, neural nets) to approximate $V(s)$. In RL, this is "function approximation." But combining regression with Bellman iteration can diverge. Why?

**Step 1: Write Section 3 content**

Structure:
1. **Opening:** Economists use regression constantly. Approximate $V(s)$ by $\hat{V}(s; \theta)$, minimize squared error. But inside a Bellman loop, this can explode. The culprit is the interaction of three elements: function approximation + bootstrapping + off-policy data.

2. **The Geometry of TD Learning:** TD learning doesn't minimize Bellman error directly. It finds a fixed point of the projected Bellman operator $\Pi T$, where $\Pi$ projects onto the function approximation subspace.

3. **Tsitsiklis-Van Roy (1997) Result:** When samples come from the on-policy stationary distribution $d^\pi$, the projection $\Pi$ is non-expansive in the $d^\pi$-weighted norm. The composition $\Pi T$ remains a contraction. TD converges.

4. **Baird's Star Counterexample (1995):** Present the Star MDP. Six states, linear function approximation with two weights, every transition observed equally (off-policy). Result: weights diverge to infinity. The geometric intuition: off-policy sampling breaks the norm under which $\Pi$ is non-expansive. The projected Bellman operator $\Pi T$ is no longer a contraction.

5. **The Deadly Triad:** Sutton's term for the combination that causes divergence:
   - Function approximation (can't store all states)
   - Bootstrapping (TD targets depend on current estimates)
   - Off-policy learning (samples from different distribution than target policy)

   Remove any one element and convergence is restored.

6. **Practical Implications:** This is why DQN uses experience replay (makes samples approximately on-policy) and target networks (breaks the bootstrapping feedback loop). It's why on-policy methods (SARSA, PPO) are more stable than off-policy methods (Q-learning, DQN) with function approximation.

7. **Footnote on Gradient TD:** Sutton et al. (2009) reformulate TD as a saddle-point problem, restoring convergence off-policy. But this adds complexity and is less commonly used.

**Key citations:** `\citet{tsitsiklis1997}`, `\citet{baird1995}`, `\citet{sutton2009gtd}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 3 — deadly triad"
```

---

## Task 5: Section 4 — Policy Learning: Direct Optimization

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
Sometimes the value function is complex but the optimal policy is simple. Can we optimize the policy directly without computing $V$? This is utility maximization, not equation solving.

**Step 1: Write Section 4 content**

Structure:
1. **Opening:** Value-based methods solve the Bellman equation. Policy-based methods maximize expected utility $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t R_t]$ directly. This is familiar to economists: we're doing constrained optimization, not fixed-point iteration.

2. **The Policy Gradient Theorem (Sutton et al. 2000):** The gradient of expected return is:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a) \right]$$

   The "magic": the gradient doesn't require knowing $\nabla_\theta d^{\pi_\theta}$, the derivative of the stationary distribution with respect to policy parameters. This is intractable, but it cancels out.

3. **REINFORCE:** Williams (1992). The simplest policy gradient: sample a trajectory, weight each action's log-probability by the return from that point. Unbiased but high variance.

4. **Natural Policy Gradient (Kakade 2002):** Standard gradients treat all parameter directions equally. But small changes in $\theta$ can cause large changes in the policy distribution. The natural gradient preconditions by the Fisher information matrix:
   $$\tilde{\nabla}_\theta J = F(\theta)^{-1} \nabla_\theta J$$

   **Key insight:** NPG is equivalent to Policy Iteration in the tabular case. The natural gradient direction matches the policy improvement step exactly. This closes the loop to Puterman-Brumelle: we're still doing Newton's method.

5. **Global Convergence Despite Non-Convexity (Agarwal et al. 2021):** Economists fear local optima. The RL objective $J(\theta)$ is non-convex. But it satisfies gradient domination (Polyak-Łojasiewicz condition):
   $$\|\nabla J(\theta)\|^2 \geq \mu (J^* - J(\theta))$$

   This implies every stationary point is global. NPG converges to the global optimum at rate $O(1/T)$.

6. **Performance Difference Lemma:** The "Taylor expansion" of value functions. For any two policies:
   $$V^{\pi'} - V^\pi = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi'}}[A^\pi(s, \pi'(s))]$$

   This is the starting point for TRPO, PPO, and all modern policy optimization analysis.

7. **Entropy Regularization:** Adding entropy accelerates convergence from $O(1/T)$ to $O(e^{-cT})$ (Cen et al. 2022). The soft-max policy is exactly the logit choice probability from discrete choice econometrics.

**Key citations:** `\citet{sutton2000}`, `\citet{williams1992}`, `\citet{kakade2002}`, `\citet{agarwal2021}`, `\citet{cen2022}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 4 — policy gradient"
```

---

## Task 6: Section 5 — Hybrid Learning: Actor-Critic

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
Policy gradient has high variance (noisy trajectory samples). Value iteration has bias (approximation error). Can we combine them?

**Step 1: Write Section 5 content**

Structure:
1. **Opening:** REINFORCE uses the actual return $G_t$ as the signal for policy updates. High variance because one trajectory is one sample. TD uses bootstrapped estimates $r + \gamma V(s')$. Lower variance but biased if $V$ is wrong. Actor-Critic combines both.

2. **Two-Timescale Stochastic Approximation (Konda-Tsitsiklis 2000):** Run two learning processes at different speeds:
   - Critic (fast): estimate $V^\pi$ or $Q^\pi$ for current policy
   - Actor (slow): update policy using critic's value estimates

   If the critic learns fast enough, the actor sees the critic as an oracle providing exact values. The actor's updates become unbiased policy gradient steps.

3. **Convergence Conditions:** The step sizes must satisfy:
   - $\alpha_{\text{critic}} / \alpha_{\text{actor}} \to \infty$ (critic faster than actor)
   - Both satisfy Robbins-Monro conditions

   Under these conditions, actor-critic converges to a local optimum of $J(\theta)$.

4. **Practical Variants:** A2C (synchronous), A3C (asynchronous), SAC (entropy-regularized). All share the two-timescale structure.

5. **Connection to Economics:** This is like estimating a model (critic) while optimizing a policy (actor). The "identification" comes from the critic learning the value function; the "optimization" comes from the actor improving the policy. Separation of concerns.

**Key citations:** `\citet{konda2000}`, `\citet{bhatnagar2009}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 5 — actor-critic"
```

---

## Task 7: Section 6 — Bridging Results: Error Bounds and Complexity

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`

**Context for Economists:**
"So what?" questions: If my approximation is slightly wrong, does the system collapse? Does this scale to high dimensions?

**Step 1: Write Section 6 content**

Structure:
1. **Error Propagation (Singh-Yee 1994):** If you approximate the value function with error $\epsilon$, how bad is the resulting policy? The bound:
   $$\|V^* - V^{\pi_{\text{greedy}}}\|_\infty \leq \frac{2\gamma}{1-\gamma} \|V^* - \hat{V}\|_\infty$$

   Interpretation: a 1% error in $\hat{V}$ with $\gamma = 0.99$ yields at most 200% error in policy value. Not great, but bounded. "Imperfect" neural networks don't cause catastrophic failure.

2. **Breaking the Curse of Dimensionality (Kearns et al. 2002):** Classical DP complexity scales with $|\mathcal{S}|$. For Go, $|\mathcal{S}| \approx 10^{170}$. Impossible.

   Sparse sampling theorem: with a generative model (simulator), near-optimal planning requires samples polynomial in:
   - Horizon $H = O(\log(1/\epsilon) / \log(1/\gamma))$
   - Branching factor $|\mathcal{A}|$
   - Desired accuracy $1/\epsilon$

   But **no dependence on $|\mathcal{S}|$**. This is why AlphaZero can solve Go.

3. **Model-Based vs Model-Free Unification (Sutton 1990, Dyna):** The apparent dichotomy dissolves. Model-based RL learns a simulator $\hat{P}$, then generates synthetic experience for model-free updates. Model-free RL uses real experience directly. In the limit of infinite synthetic samples, they converge to the same solution.

   Dyna-Q: after each real transition $(s, a, r, s')$, perform $k$ simulated transitions using the learned model. This accelerates learning by a factor of $k$.

4. **Summary Table:** Compile the key quantitative results:

   | Result | Bound | Implication |
   |--------|-------|-------------|
   | VI convergence | $\gamma^k$ | 460 iters for $\gamma=0.99$, 100× reduction |
   | PI convergence | Quadratic | ~7 iters for same |
   | Q-learning | $\tilde{O}(1/\sqrt{t})$ | $10^6$ samples for $\epsilon=0.01$ |
   | Value error → policy error | $\frac{2\gamma}{1-\gamma}\epsilon$ | Bounded degradation |
   | Sparse sampling | No $|\mathcal{S}|$ dependence | Curse broken for planning |

**Key citations:** `\citet{singh1994}`, `\citet{kearns2002}`, `\citet{sutton1990dyna}`

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -am "feat(ch02): add Section 6 — bridging results"
```

---

## Task 8: Final Polish and Integration

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v3.tex`
- Modify: `docs/main.tex` (update chapter reference)

**Step 1: Add concluding remarks**

Write 1-2 paragraphs synthesizing the chapter:
- RL is not magic; it's asymptotic approximation of the Bellman operator
- The theoretical framework (contractions, stochastic approximation, gradient domination) provides guarantees
- Remaining open problems: sample complexity gaps, function approximation theory for deep nets

**Step 2: Update main.tex to reference new file**

Change: `\input{../ch02_planning_learning/tex/planning_learning_v2}` → `\input{../ch02_planning_learning/tex/planning_learning_v3}`

**Step 3: Full compilation test**

Run: `cd /Users/pranjal/Code/rl/docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex`
Expected: Full document compiles, Chapter 2 appears correctly

**Step 4: Archive old file**

```bash
cp ch02_planning_learning/tex/planning_learning_v2.tex ch02_planning_learning/tex/backups/2026-01-30-planning_learning_v2.tex
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(ch02): complete narrative rewrite for economists"
```

---

## Verification Checklist

1. [ ] `planning_learning_v3.tex` compiles without errors
2. [ ] All 6 sections present with narrative-driven structure
3. [ ] Key papers cited: Puterman-Brumelle, Jaakkola, Tsitsiklis-Van Roy, Baird, Sutton (PG), Kakade, Agarwal, Konda-Tsitsiklis, Singh-Yee, Kearns, Sutton (Dyna)
4. [ ] No proof-heavy presentation; math with interpretation
5. [ ] Economist framing throughout ("Why does this matter for structural estimation?")
6. [ ] Full document `main.tex` compiles with new chapter
7. [ ] Old version archived to `backups/`

---

## Key Papers Reference

| Paper | Section | Key Contribution |
|-------|---------|------------------|
| Puterman & Brumelle (1979) | 1 | PI = Newton's method |
| Bertsekas (2022) | 1 | Newton interpretation, rollout bounds |
| Jaakkola et al. (1994) | 2 | Q-learning convergence via stochastic approx |
| Singh et al. (2000) | 2 | SARSA convergence |
| Tsitsiklis & Van Roy (1997) | 3 | On-policy TD convergence geometry |
| Baird (1995) | 3 | Star counterexample, deadly triad |
| Sutton et al. (2000) | 4 | Policy gradient theorem |
| Kakade (2002) | 4 | Natural policy gradient = soft PI |
| Agarwal et al. (2021) | 4 | Global convergence, gradient domination |
| Konda & Tsitsiklis (2000) | 5 | Two-timescale actor-critic |
| Singh & Yee (1994) | 6 | Value error → policy error bound |
| Kearns et al. (2002) | 6 | Sparse sampling, curse breaking |
| Sutton (1990) | 6 | Dyna, model-based/free unification |
