# The Deadly Triad: A First-Principles Breakdown

## 1. The Setup (what the reader already knows)

The Bellman operator $T$ is a $\gamma$-contraction in the supremum norm (established in Section 3.1). Q-learning applies $T$ using single samples (established in Section 3.2). In the tabular case, each state has its own value entry, updates are independent, and the contraction property carries through directly. The question is what happens when the state space is too large for a lookup table.

## 2. What Goes Wrong with Function Approximation

With function approximation $V(s) \approx \phi(s)^\top \theta$, parameters are shared across states. Changing $\theta$ to improve the value estimate at state A simultaneously changes the value at state B. You can no longer apply $T$ directly to each state independently. Instead, you apply $T$ to get a target, then *project back* onto the function space (the span of your features).

The composed operator is $\Pi T^\pi$: the Bellman contraction $T^\pi$ followed by projection $\Pi$ onto the function space. The central question is whether $\Pi T^\pi$ is still a contraction.

## 3. On-Policy: It Works (Tsitsiklis & Van Roy 1997)

Define the $d^\pi$-weighted norm: $\|V\|_{d^\pi}^2 = \sum_s d^\pi(s) V(s)^2$, weighting states by how often the policy $\pi$ visits them.

The projection $\Pi$ minimizes error in this same norm: $\Pi V = \arg\min_{\hat{V} \in \text{span}(\Phi)} \|V - \hat{V}\|_{d^\pi}$. Because the projection minimizes distance in the same norm used to measure the Bellman contraction, this is an *orthogonal* projection.

**Why orthogonal projections are non-expansive (Pythagoras).** For any $V$ and its projection $\Pi V$, the residual $V - \Pi V$ is orthogonal to the approximation subspace. By the Pythagorean theorem: $\|V\|_{d^\pi}^2 = \|\Pi V\|_{d^\pi}^2 + \|V - \Pi V\|_{d^\pi}^2$. This means $\|\Pi V\|_{d^\pi} \leq \|V\|_{d^\pi}$. Applied to differences: $\|\Pi V_1 - \Pi V_2\|_{d^\pi} \leq \|V_1 - V_2\|_{d^\pi}$. The projection never inflates distances.

So the composition contracts: $\|\Pi T^\pi V_1 - \Pi T^\pi V_2\|_{d^\pi} \leq 1 \cdot \gamma \|V_1 - V_2\|_{d^\pi} < \|V_1 - V_2\|_{d^\pi}$.

Fixed point exists and TD converges to it. Error bound (Tsitsiklis & Van Roy 1997, Theorem 1):
$$\|\Phi r^* - V^\pi\|_{d^\pi} \leq \frac{1}{\sqrt{1 - \gamma^2}} \|\Pi V^\pi - V^\pi\|_{d^\pi}$$

For TD($\lambda$), the bound becomes:
$$\|\Phi r^* - V^\pi\|_{d^\pi} \leq \frac{1-\lambda\gamma}{1-\gamma} \|\Pi V^\pi - V^\pi\|_{d^\pi}$$

## 4. Off-Policy: It Breaks

Now samples come from a behavior distribution $\mu \neq d^\pi$. The projection minimizes error under $\mu$: $\Pi_\mu V = \arg\min_{\hat{V}} \|V - \hat{V}\|_\mu$. But the Bellman operator $T^\pi$ contracts in the $d^\pi$-norm.

The projection is no longer orthogonal in the $d^\pi$-norm; it is an *oblique* projection. Oblique projections can expand errors. If $\|\Pi_\mu\|_{d^\pi} > 1/\gamma$, the expansion from projection overwhelms the $\gamma$-contraction from the Bellman operator. The fixed-point iteration diverges because the composed operator $\Pi_\mu T^\pi$ has spectral radius greater than 1.

**This is NOT overfitting.** Overfitting occurs when the function approximator memorizes training data at the expense of generalization; more data helps. Divergence occurs when the iterates grow without bound, producing arbitrarily bad value estimates. The deadly triad causes the latter: even with infinite data, the algorithm can diverge. The weights blow up because the algorithm itself is unstable, not because the data is insufficient.

## 5. Baird's Counterexample (the proof that this actually happens)

From Baird (1995). A star MDP with 6 states and zero reward everywhere, so the true value is $V^*(s) = 0$ for all $s$. A lookup table would learn this immediately.

**State structure:** States 1-5 each have a single action transitioning to state 6. State 6 has a single action transitioning back to itself.

**Feature structure:** Linear function approximation with two weights: $V(s) = 2w_1 + w_s$ for $s \in \{1,...,5\}$, and $V(6) = 2w_1 - w_6$ (signs differ on the second component). So $w_1$ is shared across all states, while each state has its own secondary weight.

**Sampling:** All transitions are observed equally often (uniform distribution, not $d^\pi$).

**The feedback loop:**
- States 1-5 transition to state 6. When $V(6)$ is large and positive, the TD target $\gamma V(6)$ exceeds $V(s)$ for $s \in \{1,...,5\}$. The TD error is positive, pushing $w_1$ upward.
- State 6 transitions to itself. Here the TD target is $\gamma V(6) < V(6)$, so the TD error is negative, pushing $w_1$ downward.
- But states 1-5 are visited 5 times for every 1 visit to state 6 (uniform sampling).
- Net effect: $w_1$ gets pushed up 5 times for every 1 time it gets pushed down. It diverges to $+\infty$.

The cross-state coupling through the shared weight $w_1$ creates a positive feedback loop. A lookup table (identity features, no sharing) would simply set everything to 0.

**Why uniform sampling fails:** The on-policy distribution $d^\pi$ concentrates mass on state 6 (the absorbing state), which would counterbalance the upward pressure on $w_1$. Uniform sampling under-weights state 6, creating the 5:1 imbalance.

## 5a. Numerical Walkthrough of Baird's Counterexample

This section works through the numbers step by step using the Sutton & Barto (2018) Example 11.2 formulation, which is slightly different from the original Baird (1995) setup above but cleaner to compute with.

### Setup

Seven states, eight weight parameters $w = (w_1, \ldots, w_8)$.

**Feature vectors.** Each state maps to an 8-dimensional feature vector:
- States $i = 1, \ldots, 6$: $\mathbf{x}(i) = 2\mathbf{e}_i + \mathbf{e}_8$, so $V(i) = 2w_i + w_8$
- State 7: $\mathbf{x}(7) = \mathbf{e}_7 + 2\mathbf{e}_8$, so $V(7) = w_7 + 2w_8$

The shared component $w_8$ appears in every state's value. States 1–6 load on $w_8$ with coefficient 1; state 7 loads on $w_8$ with coefficient 2.

**Transitions.** From every state, two actions are available:
- "Dashed": transitions uniformly to one of states $\{1, \ldots, 6\}$
- "Solid": transitions deterministically to state 7

**Policies.** The behavior policy $b$ selects dashed with probability 6/7 and solid with probability 1/7. The target policy $\pi$ selects solid always.

**Parameters.** $\gamma = 0.99$, all rewards are zero, so $V^*(s) = 0$ for all $s$.

**Initial weights.** $w_1 = w_2 = \cdots = w_7 = 1$, $w_8 = 10$.

### Step 1: Initial values

$$V(1) = V(2) = \cdots = V(6) = 2(1) + 10 = 12$$
$$V(7) = 1 + 2(10) = 21$$

The true values are all zero. Every estimate is already far from correct.

### Step 2: Importance sampling ratios

Since the target policy $\pi$ always plays solid:
- When behavior takes solid (prob 1/7): $\rho = \pi(\text{solid})/b(\text{solid}) = 1/(1/7) = 7$
- When behavior takes dashed (prob 6/7): $\rho = \pi(\text{dashed})/b(\text{dashed}) = 0/(6/7) = 0$

Dashed transitions contribute nothing to the expected update. Only solid transitions matter, weighted by $\rho = 7$.

### Step 3: Stationary distribution under $b$

From any state, the behavior policy sends you to states 1–6 with total probability 6/7 (uniformly, so 1/7 each) and to state 7 with probability 1/7. The stationary distribution is uniform: $d(s) = 1/7$ for all $s$.

### Step 4: Expected update at one state

The semi-gradient TD update for a transition $s \to s'$ is $\Delta w = \alpha \rho \delta \mathbf{x}(s)$, where $\delta = r + \gamma V(s') - V(s)$.

Since $\pi$ always takes solid, every transition goes to state 7. With zero rewards:

$$\delta(s) = \gamma V(7) - V(s)$$

For state 1 at epoch 0:
$$\delta(1) = 0.99 \times 21 - 12 = 20.79 - 12 = 8.79$$

For state 7:
$$\delta(7) = 0.99 \times 21 - 21 = -0.21$$

States 1–6 all have large positive TD errors; state 7 has a small negative one.

### Step 5: Expected weight update

The expected update per epoch combines all states. Since each state is visited with probability 1/7 under $d$, and only solid transitions contribute (with the $b(\text{solid}) = 1/7$ and $\rho = 7$ canceling to give an effective weight of 1):

$$\mathbb{E}[\Delta w] = \alpha \sum_{s=1}^{7} \frac{1}{7} \cdot \delta(s) \cdot \mathbf{x}(s)$$

The critical component is $w_8$, which is shared across all states. Each state contributes to $\Delta w_8$ through its feature coefficient on position 8:

| State $s$ | $\delta(s)$ | $x_8(s)$ | Contribution to $\Delta w_8$ (before $\alpha$) |
|-----------|-------------|-----------|------|
| 1 | +8.79 | 1 | $\frac{1}{7}(8.79)(1) = +1.2557$ |
| 2 | +8.79 | 1 | +1.2557 |
| 3 | +8.79 | 1 | +1.2557 |
| 4 | +8.79 | 1 | +1.2557 |
| 5 | +8.79 | 1 | +1.2557 |
| 6 | +8.79 | 1 | +1.2557 |
| 7 | −0.21 | 2 | $\frac{1}{7}(-0.21)(2) = -0.0600$ |
| **Total** | | | **+7.4743** |

Six large positive pushes versus one tiny negative push. The net update $\Delta w_8 = \alpha \times 7.4743 = 0.0747$ (with $\alpha = 0.01$). The shared weight $w_8$ grows.

### Step 6: The feedback loop, in numbers

With $\alpha = 0.01$, the weight trajectory over the first five epochs:

| Epoch | $w_8$ | $V(1) = \cdots = V(6)$ | $V(7)$ | $\delta(1)$ | $\delta(7)$ |
|-------|-------|-------|-------|-------------|-------------|
| 0 | 10.0000 | 12.0000 | 21.0000 | +8.79 | −0.21 |
| 1 | 10.0747 | 12.1250 | 21.1492 | +8.82 | −0.21 |
| 2 | 10.1497 | 12.2503 | 21.2988 | +8.85 | −0.21 |
| 3 | 10.2248 | 12.3759 | 21.4487 | +8.87 | −0.21 |
| 4 | 10.3001 | 12.5018 | 21.5990 | +8.90 | −0.22 |
| 5 | 10.3756 | 12.6281 | 21.7497 | +8.93 | −0.22 |

Every epoch, $w_8$ grows. This increases $V(7)$ (which has coefficient 2 on $w_8$), which increases $\delta(s)$ for $s = 1, \ldots, 6$, which increases $\Delta w_8$ further. Positive feedback.

After 100 epochs, $w_8 = 18.50$, $V(7) = 37.95$. The values are diverging from the true answer of zero.

### Step 7: Why it diverges — the expected update matrix

Write the expected update as $\Delta w = \alpha A w$ where:

$$A = \frac{1}{7}\sum_{s=1}^{7} \mathbf{x}(s)\bigl(\gamma \mathbf{x}(7) - \mathbf{x}(s)\bigr)^\top$$

The iteration is $w_{t+1} = (I + \alpha A) w_t$. This diverges if and only if $I + \alpha A$ has spectral radius greater than 1, which happens when $A$ has an eigenvalue with positive real part.

The eigenvalues of $A$:

| Eigenvalue | Multiplicity | Interpretation |
|-----------|-------------|----------------|
| +0.2393 | 1 | **Unstable mode** (drives divergence) |
| +0.0222 | 1 | Weakly unstable mode |
| 0 | 1 | Neutral (invariant subspace) |
| −0.5714 | 5 | Stable modes (individual $w_i$ contract) |

Two positive eigenvalues confirm divergence. The dominant unstable eigenvalue 0.2393 gives a per-step growth factor of $1 + 0.01 \times 0.2393 = 1.00239$, meaning exponential blowup.

The five stable eigenvalues at −0.5714 correspond to the individual weights $w_1, \ldots, w_5$ (and $w_6$), which each contract. The instability lives in the shared weight $w_8$ and its interaction with $w_7$.

### Step 8: On-policy contrast

Under the target policy $\pi$ (always solid), the on-policy distribution $d^\pi$ concentrates entirely on state 7 (it is absorbing under solid). Only state 7 gets sampled. No importance sampling needed ($\rho = 1$).

TD error at state 7: $\delta = \gamma V(7) - V(7) = (\gamma - 1)V(7)$. Since $\gamma < 1$, this is always negative. The update shrinks both $w_7$ and $w_8$:

| Epoch | $w_7$ | $w_8$ | $V(7)$ |
|-------|-------|-------|--------|
| 0 | 1.0000 | 10.0000 | 21.0000 |
| 1 | 0.9979 | 9.9958 | 20.9895 |
| 2 | 0.9958 | 9.9916 | 20.9790 |
| 3 | 0.9937 | 9.9874 | 20.9685 |
| 5 | 0.9895 | 9.9790 | 20.9476 |
| 10 | 0.9790 | 9.9581 | 20.8952 |

$V(7)$ contracts toward zero at rate $(1 - \alpha(1-\gamma)) = 0.9999$ per step. No instability because the projection aligns with the sampling distribution. The on-policy sampling "sees" the absorbing state enough to keep $w_8$ in check.

### Step 9: Tabular contrast

Replace the feature structure with a lookup table: each state has its own independent parameter. Now $V(s) = w_s$ with no shared weights. The update to $V(s)$ depends only on $V(s)$ and $V(7)$; there is no cross-state coupling through a shared parameter.

In the expected update, $V(7)$ receives:

$$\Delta V(7) = \alpha \cdot \tfrac{1}{7} \cdot (\gamma - 1) V(7) < 0$$

$V(7)$ contracts toward zero unconditionally. Each $V(s)$ for $s = 1, \ldots, 6$ is pulled toward $\gamma V(7)$, which is itself shrinking. Eventually all values converge to zero.

The instability in the function approximation case arises specifically because the shared weight $w_8$ creates a channel for positive feedback between states. Remove the sharing (go tabular), and each state's estimate evolves independently. The contraction property of the Bellman operator applies state by state.

### Summary: what drives divergence

Three ingredients combine:
1. **Shared weights** ($w_8$ couples all states): a positive TD error at state 1 increases $V(7)$ through $w_8$
2. **Off-policy sampling** (uniform, not $d^\pi$): states 1–6 get 6x the weight of state 7, creating a 6:1 imbalance favoring the positive push
3. **Bootstrapping** (target depends on current estimate): larger $V(7)$ feeds back into larger $\delta(s)$ for $s = 1, \ldots, 6$

Remove any one and the loop breaks.

## 6. The Loss Function Trap (why you can't just "fix the gradient")

The natural fix: minimize the mean-squared Bellman error $\|Q - TQ\|^2$ directly via gradient descent.

**The double-sampling problem:** The gradient of $\|Q - \mathbb{E}[r + \gamma V(s')]\|^2$ requires the product $\mathbb{E}[\cdot] \cdot \nabla\mathbb{E}[\cdot]$. A single sample gives a biased estimate because $\mathbb{E}[XY] \neq \mathbb{E}[X]\mathbb{E}[Y]$. You need TWO independent next-state samples from the same $(s,a)$, which is impractical in most environments.

So practitioners use "semi-gradient" TD: treat the target $r + \gamma V(s')$ as a constant (stop the gradient through the bootstrap target). Semi-gradient finds the $\Pi T^\pi$ fixed point, not the true Bellman fixed point. This is why RL uses "semi-gradient" methods that are inherently unstable off-policy.

Baird (1995) proposed residual gradient algorithms that perform true gradient descent on the MSBE, guaranteeing convergence but requiring the double sample.

## 7. Why Each Component Is Individually Necessary

- **Function approximation** addresses the curse of dimensionality (Go has $10^{170}$ states; storing separate values is infeasible).
- **Bootstrapping** enables learning from single transitions without waiting for episode end, with lower variance than Monte Carlo.
- **Off-policy learning** enables Q-learning's key property: learn about the optimal policy while following an exploratory behavior policy; reuse old data via experience replay.

**Remove any one and convergence is restored:**
- Without function approximation (tabular): projection is the identity, Q-learning's contraction applies directly.
- Without bootstrapping (Monte Carlo returns): targets are independent of current estimates, reducing to supervised learning.
- Without off-policy learning: samples come from $d^\pi$, the Tsitsiklis-Van Roy conditions hold, projection is orthogonal.

## 8. Solutions

Three mechanisms, each attacking a different leg of the triad:

### Target Networks (weaken bootstrapping)

Instead of bootstrapping from $r + \gamma Q(s'; \theta)$, use $r + \gamma Q(s'; \theta^-)$ where $\theta^-$ is a slowly-updated copy. The regression target becomes quasi-static, so the main network's update is just least-squares regression against a fixed target. This converts the coupled fixed-point problem into a sequence of supervised learning problems.

- Zhang (2021): proves convergence to a regularized TD fixed point with two-timescale analysis and ridge regularization. The target network update uses Polyak averaging augmented with two projections.
- Fellows (2023): target networks recondition the TD Jacobian. The spectral radius of the composed update depends on the target network update frequency $k$; for sufficiently large $k$, the spectral radius drops below 1 even in off-policy/nonlinear settings.

### Gradient TD (fix the projection mismatch)

Sutton et al. (2009): reformulate the projected Bellman error as a saddle-point problem $\min_w \max_y L(w, y)$. The resulting algorithms (GTD, GTD2, TDC) perform true stochastic gradient descent on the mean-squared projected Bellman error. No semi-gradient hack.

- Converges off-policy with linear function approximation.
- Cost: auxiliary variables (a second set of parameters), slower per-step, two-timescale learning rates.

### Regularization (shrink the projection)

Lim & Lee (2024): add an $\ell_2$ penalty $-\eta\theta$ to the Q-learning update. This changes the projection from $\Pi = X(X^\top D X)^{-1} X^\top D$ to $\Pi_\eta = X(X^\top D X + \eta I)^{-1} X^\top D$.

As $\eta$ increases, the projection "shrinks" toward the origin. The spectral radius of $\Pi_\eta$ decreases. For sufficiently large $\eta$, $\gamma \|\Pi_\eta\| < 1$, restoring the contraction property. The algorithm converges to a regularized fixed point (biased, but stable).

Analysis uses a switching system framework: the greedy policy changes define a finite set of linear subsystems, and the regularization ensures global asymptotic stability of the switched system.
