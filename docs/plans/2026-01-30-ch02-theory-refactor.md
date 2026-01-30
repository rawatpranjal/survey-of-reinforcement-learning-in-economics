# Chapter 2 Theory Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate 4 overlapping tex files into a single ~10 page theory chapter focused on best convergence results for 8 core algorithms.

**Architecture:** Remove simulations from prose (keep files). Heavy cross-referencing to Chapter 1 (which introduces all algorithms). Two tracks: value-based (Bellman fixed points) and policy-based (direct J(θ) optimization). FRAP table for 8 algorithms only.

**Tech Stack:** LaTeX with natbib, amsmath, amsthm. Compile via pdflatex with chapter compilation command.

---

## Pre-Implementation: Files Audit

**Current state (4 overlapping files):**
- `ch02_planning_learning/tex/planning_learning.tex` - 244 lines, has simulations
- `ch02_planning_learning/tex/planning_learning_alt.tex` - 278 lines, has simulations
- `ch02_planning_learning/tex/planning_learning_theory.tex` - 594 lines, theory-focused
- `ch02_planning_learning/tex/unified_planning_learning.tex` - 241 lines, operator framework

**Target:** Single `planning_learning_v2.tex` (~10 pages)

**Keep unchanged:**
- All files in `sims/` (don't reference in tex)
- All files in `papers/`
- All files in `notes/`

---

### Task 1: Archive Existing Files

**Files:**
- Move: `ch02_planning_learning/tex/planning_learning.tex` → `backups/`
- Move: `ch02_planning_learning/tex/planning_learning_alt.tex` → `backups/`
- Move: `ch02_planning_learning/tex/planning_learning_theory.tex` → `backups/`
- Move: `ch02_planning_learning/tex/unified_planning_learning.tex` → `backups/`

**Step 1: Archive all 4 files with timestamp**

```bash
cd /Users/pranjal/Code/rl/ch02_planning_learning/tex
ts=$(date +%Y-%m-%d-%H%M%S)
for f in planning_learning.tex planning_learning_alt.tex planning_learning_theory.tex unified_planning_learning.tex; do
  if [ -f "$f" ]; then
    cp "$f" "backups/${ts}_${f}"
    echo "Archived $f"
  fi
done
```

Expected: 4 files archived to backups/

**Step 2: Verify archives exist**

```bash
ls -la /Users/pranjal/Code/rl/ch02_planning_learning/tex/backups/ | grep 2026-01-30
```

Expected: 4 new timestamped files

**Step 3: Commit archive**

```bash
git add ch02_planning_learning/tex/backups/
git commit -m "chore: archive ch02 tex files before consolidation"
```

---

### Task 2: Write Section 2.1 - Introduction (0.5 pages)

**Files:**
- Create: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Create file with header and Section 2.1**

```latex
% Chapter 2: Planning and Learning — Theoretical Foundations
% Single consolidated file. Theory only; simulations in Chapter 3.
% Primary references: Bertsekas (2022, 2024, 2025), Agarwal et al. (2021), Cen et al. (2022)

\subsection{Introduction}
\label{sec:ch2_intro}

Chapter~\ref{ch:history} introduced the algorithms that form the backbone of modern reinforcement learning: value iteration, policy iteration, Q-learning, DQN, TRPO, PPO, and AlphaZero. This chapter develops the convergence theory that governs these methods. The goal is not to re-derive the algorithms but to characterize their convergence rates and identify the conditions under which they succeed or fail.

The theoretical literature organizes into two families. Value-based methods seek fixed points of the Bellman operator: value iteration applies the operator directly, policy iteration linearizes and solves, Q-learning performs stochastic approximation, and DQN extends this to neural networks. Policy-based methods optimize the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^\infty \gamma^t R_t]$ directly over policy parameters: natural policy gradient exploits the geometry of the policy space, and entropy regularization accelerates convergence. Soft Actor-Critic bridges both families through the soft Bellman operator.

Table~\ref{tab:ch2_convergence_summary} summarizes the best known convergence results for eight core algorithms. The remainder of this chapter develops these results in detail.

\begin{table}[h!]
\centering
\caption{Convergence rates for core algorithms}
\label{tab:ch2_convergence_summary}
\small
\begin{tabular}{l|l|l|l}
\hline
Algorithm & Best Rate & Source & Practical Meaning \\
\hline
VI & $\gamma^k$ (linear) & Bellman & $\gamma=0.99 \Rightarrow$ 460 iters for 100$\times$ reduction \\
PI & Quadratic & Bertsekas & $\sim$7 iters for same reduction \\
Q-learning & $\tilde{O}(1/\sqrt{t})$ & Even-Dar \& Mansour & $10^6$ samples for $\varepsilon=0.01$ \\
DQN & $\gamma^K + |A|n^{-\alpha^*}$ & Fan et al. & Algorithmic + statistical error \\
AlphaZero & $\frac{2\gamma^H}{1-\gamma}\|V^\pi - V^*\|$ & Bertsekas & $H=50, \gamma=0.99 \Rightarrow 0.6\times$ error \\
NPG & $O((1-\gamma)^{-2}/\varepsilon)$ & Agarwal et al. & No $|S||A|$ dependence \\
NPG+entropy & $O(\log(1/\varepsilon))$ + local quad & Cen et al. & Linear global, quadratic local \\
SAC & Soft PI converges & Haarnoja et al. & Inherits NPG guarantees \\
\hline
\end{tabular}
\end{table}
```

**Step 2: Verify file created**

```bash
wc -l /Users/pranjal/Code/rl/ch02_planning_learning/tex/planning_learning_v2.tex
```

Expected: ~45 lines

---

### Task 3: Write Section 2.2 - Fixed-Point Methods (2.5 pages)

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Append Section 2.2.1 - Convergence Rates**

```latex
\subsection{Fixed-Point Methods for Value Functions}
\label{sec:fixed_point}

\subsubsection{Convergence Rates: What They Mean Practically}

The Bellman optimality operator $T$ acts on value functions via $(TV)(s) = \max_{a} \{ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \}$. Both $T$ and the policy evaluation operator $T^\pi$ are $\gamma$-contractions in the supremum norm: $\|TV_1 - TV_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$. By Banach's fixed-point theorem, value iteration $V_{k+1} = TV_k$ converges geometrically:
\begin{equation}
\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty.
\label{eq:vi_convergence}
\end{equation}

The contraction modulus $\gamma$ determines practical iteration counts. To reduce error by a factor of $10^m$, value iteration requires $k = m / \log_{10}(1/\gamma)$ iterations. Table~\ref{tab:vi_iterations} shows iteration counts for common discount factors.

\begin{table}[h!]
\centering
\caption{Value iteration: iterations to achieve target error reduction}
\label{tab:vi_iterations}
\begin{tabular}{c|ccc}
\hline
$\gamma$ & 100$\times$ reduction & 1000$\times$ reduction & $10^6\times$ reduction \\
\hline
0.90 & 44 & 66 & 131 \\
0.95 & 90 & 134 & 269 \\
0.99 & 459 & 688 & 1376 \\
\hline
\end{tabular}
\end{table}
```

**Step 2: Append Section 2.2.2 - Policy Iteration as Newton's Method**

```latex
\subsubsection{Policy Iteration as Newton's Method}

Policy Iteration achieves faster convergence by exploiting the structure of the Bellman equation. \citet{bertsekas2022} develops the key insight: at the current value estimate $\tilde{J}$, the policy operator $T^{\tilde{\pi}}$ for the greedy policy $\tilde{\pi}$ acts as a supporting hyperplane to the nonlinear operator $T$, satisfying $T^{\tilde{\pi}} \tilde{J} = T\tilde{J}$ and $T^{\tilde{\pi}} J \leq TJ$ for all $J$.\footnote{This linearization interpretation originates in \citet{kleinman1968} for Riccati equations and \citet{pollatschek1969} for stochastic games.}

Policy evaluation solves the linearized fixed-point equation $J = T^{\tilde{\pi}} J$ exactly, just as Newton's method solves the linearized system at each iterate. This yields quadratic convergence near the optimum.

\begin{theorem}[Quadratic Convergence of PI]
\label{thm:pi_quadratic}
For a finite discounted MDP with $|\mathcal{S}|$ states and discount factor $\gamma < 1$, the Policy Iteration sequence $\{\pi_k\}$ satisfies:
\begin{enumerate}
\item[(a)] Monotone improvement: $J^{\pi_{k+1}}(s) \leq J^{\pi_k}(s)$ for all $s$, with strict inequality unless $\pi_k$ is optimal.
\item[(b)] Quadratic convergence: $\|J^{\pi_{k+1}} - J^*\|_\infty = O(\|J^{\pi_k} - J^*\|_\infty^2)$ near the optimum.
\item[(c)] Finite termination: PI terminates in at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations.
\end{enumerate}
\end{theorem}

The practical consequence: PI typically converges in 5--10 iterations regardless of $\gamma$, while VI may require hundreds when $\gamma$ is close to 1.
```

**Step 3: Append Section 2.2.3 - Q-Learning as Stochastic VI**

```latex
\subsubsection{Q-Learning as Stochastic Value Iteration}

Q-learning \citep{WatkinsDayan1992} performs stochastic approximation of value iteration on Q-factors. The Q-factor Bellman operator $F$ acts on action-value functions:
\begin{equation}
(FQ)(s,a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q(s',a').
\end{equation}
This operator is a $\gamma$-contraction: $\|FQ_1 - FQ_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$. The Q-learning update $Q(s,a) \leftarrow Q(s,a) + \alpha_t [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ is a single-sample stochastic approximation of the iteration $Q_{k+1} = FQ_k$.

\begin{theorem}[Q-Learning Convergence, \citet{WatkinsDayan1992}, \citet{tsitsiklis1994}]
\label{thm:q_convergence}
Under the conditions: (i) every $(s,a)$ is visited infinitely often, (ii) step sizes satisfy Robbins-Monro ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$), Q-learning converges to $Q^*$ with probability 1.
\end{theorem}

The finite-time convergence rate is $\tilde{O}(1/\sqrt{t})$ \citep{evendar2003}. To achieve $\varepsilon$-optimal policy, synchronous Q-learning requires $\tilde{\Theta}(|\mathcal{S}||\mathcal{A}|/(1-\gamma)^4\varepsilon^2)$ samples \citep{li2024minimax}. The $(1-\gamma)^{-4}$ dependence exceeds the minimax lower bound $\Omega((1-\gamma)^{-3})$ by a factor of $(1-\gamma)^{-1}$, arising from overestimation bias in the max operator.
```

---

### Task 4: Write Section 2.3 - Deep Value-Based Methods (2 pages)

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Append Section 2.3.1 - DQN Theory**

```latex
\subsection{Deep Value-Based Methods}
\label{sec:deep_value}

\subsubsection{DQN Theory}

Chapter~\ref{ch:history} introduced the DQN architecture: experience replay breaks temporal correlation, and a target network stabilizes the regression target. \citet{fan2020dqn} provide the first rigorous convergence analysis.

\begin{theorem}[DQN Convergence, \citet{fan2020dqn}]
\label{thm:dqn_convergence}
Under Bellman completeness of the function class and bounded concentrability, neural Fitted Q-Iteration with $K$ iterations and $n$ samples per iteration satisfies:
\begin{equation}
\|Q_K - Q^*\|_\infty \leq \gamma^K \|Q_0 - Q^*\|_\infty + C \cdot |\mathcal{A}| \cdot n^{-\alpha^*},
\end{equation}
where $\alpha^* \in (0, 1/2]$ depends on the ReLU network approximation rate, and $C$ is a problem-dependent constant.
\end{theorem}

The bound decomposes into algorithmic error ($\gamma^K$, controlled by iterations) and statistical error ($n^{-\alpha^*}$, controlled by samples). Experience replay provides i.i.d. samples from a fixed distribution, satisfying the statistical assumptions. The target network holds the regression target fixed during gradient steps, making each iteration a well-defined regression problem.
```

**Step 2: Append Section 2.3.2 - Target Networks**

```latex
\subsubsection{Why Target Networks Stabilize Learning}

The deadly triad (function approximation + bootstrapping + off-policy learning) can cause divergence \citep{Baird1995}. \citet{zhang2021target} prove that target networks break this instability.

\begin{theorem}[Target Network Convergence, \citet{zhang2021target}]
\label{thm:target_network}
With Polyak-averaging target updates $\theta^- \leftarrow (1-\tau)\theta^- + \tau\theta$ and projected gradient descent, the iterates converge geometrically:
\begin{equation}
\|\theta_l - \theta^*\|_2 \leq c^l \|\theta_0 - \theta^*\|_2,
\end{equation}
for some $c < 1$, to the regularized TD fixed point.
\end{theorem}

The mechanism: the target network reconditions the TD Jacobian. Without it, the eigenvalues of the update matrix can have positive real parts, causing divergence. With the target network, the effective update matrix has all eigenvalues with negative real parts, ensuring contraction.
```

**Step 3: Append Section 2.3.3 - AlphaZero and Rollout**

```latex
\subsubsection{AlphaZero and Rollout}

Rollout performs a single policy improvement step from a base policy $\mu$: given $V^\mu$, select $\tilde{\pi}(s) = \argmax_a \{ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\mu(s') \}$. \citet{bertsekas2022} proves the cost improvement property: $V^{\tilde{\pi}}(s) \leq V^\mu(s)$ for all $s$, with strict inequality unless $\mu$ is optimal.

\begin{theorem}[Rollout Error Bound, \citet{bertsekas2021lessons}]
\label{thm:rollout_error}
Let $\tilde{\pi}$ be the policy obtained by $H$-step lookahead with terminal approximation $V^\pi$ for base policy $\pi$. Then:
\begin{equation}
\|V^{\tilde{\pi}} - V^*\|_\infty \leq \frac{2\gamma^H}{1-\gamma} \|V^\pi - V^*\|_\infty.
\end{equation}
\end{theorem}

At $H=50$ and $\gamma=0.99$, the amplification factor is $2 \cdot 0.99^{50} / 0.01 \approx 1.2$; at $H=100$, it drops to $\approx 0.07$. Longer lookahead compensates for worse value function approximations. AlphaZero implements this principle: neural networks provide $V^\pi$, and MCTS extends the effective lookahead to $H \gg 1$.
```

---

### Task 5: Write Section 2.4 - Policy Optimization Methods (2.5 pages)

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Append Section 2.4.1 - Why Policy Methods**

```latex
\subsection{Policy Optimization Methods}
\label{sec:policy_optimization}

Value-based methods seek Bellman fixed points. Policy-based methods take a fundamentally different approach: they optimize the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^\infty \gamma^t R_t]$ directly over policy parameters $\theta$. This section develops the convergence theory for natural policy gradient methods.

\subsubsection{Why Policy Methods for Continuous Control}

Q-learning requires computing $\max_a Q(s,a)$ at every update. When $\mathcal{A}$ is continuous, this becomes a nested optimization problem. Policy gradient methods sidestep this: parameterize the policy as $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$, sample an action, observe the return, and update $\theta$ by gradient ascent.\footnote{This is why virtually all continuous-control results use policy gradient methods rather than Q-learning.}

The policy gradient theorem \citep{SuttonMcAllester2000} establishes:
\begin{equation}
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a) \right],
\end{equation}
where $d^{\pi_\theta}$ is the discounted state visitation distribution.
```

**Step 2: Append Section 2.4.2 - Natural Policy Gradient**

```latex
\subsubsection{Natural Policy Gradient}

Standard gradient descent treats all parameter directions equally. \citet{Kakade2001} observed that the policy space has non-Euclidean geometry: small changes in $\theta$ can cause large changes in the policy distribution. The natural gradient accounts for this curvature.

\begin{definition}[Natural Policy Gradient]
The natural gradient is $\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)$, where $F(\theta) = \mathbb{E}_{s,a}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top]$ is the Fisher information matrix.
\end{definition}

\citet{Kakade2001} proved that NPG is equivalent to Policy Iteration in the tabular setting: the natural gradient direction exactly matches the policy improvement step. This explains why NPG inherits PI's fast convergence.

\begin{theorem}[NPG Global Convergence, \citet{agarwal2021theory}]
\label{thm:npg_convergence}
For softmax policies, NPG with step size $\eta = (1-\gamma)/4$ achieves:
\begin{equation}
J(\pi^*) - J(\pi_k) \leq \frac{4}{(1-\gamma)^2} \cdot \frac{1}{k+1}.
\end{equation}
The iteration complexity to achieve $\varepsilon$-optimal policy is $O((1-\gamma)^{-2}/\varepsilon)$, with no dependence on $|\mathcal{S}|$ or $|\mathcal{A}|$.
\end{theorem}

The dimension-free convergence is remarkable: for lifecycle models with large or continuous state spaces, the computational cost of NPG does not scale with state space size.
```

**Step 3: Append Section 2.4.3 - Entropy Regularization**

```latex
\subsubsection{Entropy Regularization and Fast Convergence}

Adding entropy to the objective smooths the optimization landscape. The entropy-regularized objective is:
\begin{equation}
J_\tau(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^\infty \gamma^t \left( R_t + \tau \mathcal{H}(\pi_\theta(\cdot|s_t)) \right)\right],
\end{equation}
where $\mathcal{H}(\pi) = -\sum_a \pi(a) \log \pi(a)$ is the entropy and $\tau > 0$ is the temperature.

\begin{theorem}[Fast NPG Convergence, \citet{cen2022fast}]
\label{thm:npg_entropy}
NPG with entropy regularization achieves:
\begin{enumerate}
\item[(a)] Global linear convergence: $J_\tau(\pi^*) - J_\tau(\pi_k) \leq (1 - c\tau)^k \cdot [J_\tau(\pi^*) - J_\tau(\pi_0)]$ for constant $c > 0$.
\item[(b)] Local quadratic convergence: once $\|\pi_k - \pi^*\|$ is small, convergence becomes quadratic.
\end{enumerate}
Iteration complexity: $O((1-\gamma)^{-2} \log(1/\varepsilon))$ to achieve $\varepsilon$-optimal policy.
\end{theorem}

The $\log(1/\varepsilon)$ dependence (versus $1/\varepsilon$ for unregularized NPG) makes entropy regularization essential for practical convergence. The mechanism: entropy prevents the policy from collapsing to a deterministic distribution, avoiding the vanishing gradients that slow convergence.
```

**Step 4: Append Section 2.4.4 - Trust Region Methods (brief)**

```latex
\subsubsection{Trust Region Methods}

TRPO \citep{Schulman2015} and PPO \citep{Schulman2017} constrain the KL divergence between successive policies: $\bar{D}_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$. This prevents the catastrophic policy collapses that plague unconstrained policy gradient.\footnote{PPO dominates practice despite lacking the formal guarantees of TRPO. The clipped surrogate objective $\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t)$ provides a simple, effective heuristic.}

Trust region constraints are equivalent to KL regularization with adaptive $\tau$. \citet{shani2020} prove that adaptive trust region methods achieve $\tilde{O}(1/\sqrt{N})$ convergence to global optima, connecting the practical success of PPO to the theoretical framework of regularized policy optimization.
```

---

### Task 6: Write Section 2.5 - Soft Actor-Critic (2 pages)

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Append Section 2.5.1 - Soft Bellman Operator**

```latex
\subsection{Hybrid Methods: Soft Actor-Critic}
\label{sec:sac}

Soft Actor-Critic \citep{Haarnoja2018} bridges value-based and policy-based methods through the maximum entropy framework. The key object is the soft Bellman operator.

\subsubsection{Soft Bellman Operator}

\begin{definition}[Soft Bellman Operator]
The soft Bellman operator $T^\pi_{\text{soft}}$ for policy $\pi$ is:
\begin{equation}
(T^\pi_{\text{soft}} Q)(s,a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a) \left[ \sum_{a'} \pi(a'|s') Q(s',a') + \tau \mathcal{H}(\pi(\cdot|s')) \right].
\end{equation}
\end{definition}

The soft optimality operator replaces the hard max with a soft-max:
\begin{equation}
(T_{\text{soft}} Q)(s,a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a) \tau \log \sum_{a'} \exp\left(\frac{Q(s',a')}{\tau}\right).
\end{equation}

\begin{lemma}[Soft Contraction]
\label{lem:soft_contraction}
The soft Bellman operator is a $\gamma$-contraction in the supremum norm: $\|T_{\text{soft}} Q_1 - T_{\text{soft}} Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$.
\end{lemma}

The optimal soft policy has the Boltzmann form:
\begin{equation}
\pi^*(a|s) = \frac{\exp(Q^*(s,a)/\tau)}{\sum_{a'} \exp(Q^*(s,a')/\tau)}.
\end{equation}
This is exactly the logit choice probability from discrete choice econometrics \citep{Rust1987}.
```

**Step 2: Append Section 2.5.2 - Soft Policy Iteration**

```latex
\subsubsection{Soft Policy Iteration}

\begin{lemma}[Soft Policy Evaluation]
\label{lem:soft_eval}
For any policy $\pi$, the sequence $Q_{k+1} = T^\pi_{\text{soft}} Q_k$ converges to the unique soft Q-function $Q^\pi_{\text{soft}}$ satisfying $Q^\pi_{\text{soft}} = T^\pi_{\text{soft}} Q^\pi_{\text{soft}}$.
\end{lemma}

\begin{lemma}[Soft Policy Improvement]
\label{lem:soft_improve}
Let $\pi_{\text{new}}$ be the Boltzmann policy with respect to $Q^\pi_{\text{soft}}$. Then $Q^{\pi_{\text{new}}}_{\text{soft}}(s,a) \geq Q^\pi_{\text{soft}}(s,a)$ for all $(s,a)$.
\end{lemma}

\begin{theorem}[Soft PI Convergence, \citet{Haarnoja2018}]
\label{thm:soft_pi}
Soft Policy Iteration, alternating soft policy evaluation and soft policy improvement, converges to the entropy-regularized optimal policy $\pi^*_\tau$ and optimal soft Q-function $Q^*_\tau$.
\end{theorem}

The proof follows the same structure as classical PI: soft evaluation finds the fixed point of a contraction, soft improvement increases the value at every state, and finite policy space guarantees termination.
```

**Step 3: Append Section 2.5.3 - Synthesis and Robustness**

```latex
\subsubsection{Synthesis and Robustness}

SAC combines three components: Q-learning (critics estimate $Q^\pi_{\text{soft}}$), policy gradient (actor maximizes expected soft Q-value), and entropy regularization (exploration + convergence acceleration). The algorithm inherits convergence guarantees from both traditions.

\citet{geist2019regularized} show that regularized MDPs form a unified framework encompassing KL-regularized policy improvement, entropy-regularized value iteration, and soft Q-learning. The regularized Bellman operator $T_\Omega$ with strongly convex $\Omega$ is a $\gamma$-contraction, preserving all fixed-point convergence guarantees.

A key property is robustness. Maximum entropy RL maximizes a lower bound on the robust RL objective: the optimal soft policy performs well across a set of perturbed MDPs, with larger $\tau$ expanding the robustness set.\footnote{This connects to the robust control literature: entropy regularization is equivalent to minimizing regret against an adversarial perturbation of the dynamics.}
```

---

### Task 7: Write Section 2.6 - Unified View and FRAP Table (1 page)

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex`

**Step 1: Append Section 2.6 with FRAP table**

```latex
\subsection{Unified View: Algorithm Taxonomy}
\label{sec:taxonomy}

\citet{moerland2022unifying} propose the FRAP framework (Full-model, Approximation, Representation, Anticipation, Policy) for classifying RL algorithms. Table~\ref{tab:frap} positions the eight core algorithms along three key dimensions: model access (full knowledge vs. learned vs. model-free), update depth (one-step bootstrap vs. multi-step vs. full trajectory), and optimization target (value function vs. policy vs. both).

\begin{table}[h!]
\centering
\caption{Algorithm classification: Model, Depth, Target, and Best Convergence Rate}
\label{tab:frap}
\begin{tabular}{l|ccc|l}
\hline
Algorithm & Model & Depth & Target & Best Rate \\
\hline
VI & Full & 1 & $V$ & $\gamma^k$ (linear) \\
PI & Full & 1 & $V, \pi$ & Quadratic \\
Q-learning & Free & 1 & $Q$ & $\tilde{O}(1/\sqrt{t})$ \\
DQN & Free & 1 & $Q$ & $\gamma^K + O(n^{-\alpha})$ \\
AlphaZero & Learned & $H$ & $V, \pi$ & $\gamma^H$ \\
NPG & Free & 1 & $\pi$ & $O(1/T)$ \\
NPG+entropy & Free & 1 & $\pi$ & $e^{-cT}$ \\
SAC & Free & 1 & $Q, \pi$ & Linear \\
\hline
\end{tabular}
\end{table}

The table reveals two convergence hierarchies. Among value-based methods, PI dominates VI due to Newton-like steps, and model-based methods dominate model-free by a factor of $(1-\gamma)^{-1}$ in sample complexity. Among policy-based methods, entropy regularization accelerates convergence from $O(1/T)$ to $O(e^{-cT})$. SAC achieves the best of both: value-based sample efficiency with policy-based continuous-action capability.
```

---

### Task 8: Add Bibliography Entries and Finalize

**Files:**
- Modify: `ch02_planning_learning/tex/planning_learning_v2.tex` (closing)
- Modify: `docs/refs.bib` (add missing entries)

**Step 1: Check for missing bibliography entries**

Required citations to verify exist in refs.bib:
- `evendar2003` - finite-time Q-learning
- `li2024minimax` - Q-learning sample complexity
- `bertsekas2021lessons` - AlphaZero rollout bounds
- `shani2020` - adaptive trust region

```bash
grep -E "(evendar|li2024minimax|bertsekas2021|shani2020)" /Users/pranjal/Code/rl/docs/refs.bib
```

**Step 2: Add any missing entries to refs.bib**

```bibtex
@inproceedings{evendar2003,
  author    = {Even-Dar, Eyal and Mansour, Yishay},
  title     = {Learning Rates for Q-learning},
  booktitle = {Journal of Machine Learning Research},
  volume    = {5},
  pages     = {1--25},
  year      = {2003}
}

@article{li2024minimax,
  author    = {Li, Gen and Wei, Yuting and Chi, Yuejie and Chen, Yuxin},
  title     = {Is Q-Learning Minimax Optimal? A Tight Sample Complexity Analysis},
  journal   = {Operations Research},
  year      = {2024}
}

@article{bertsekas2021lessons,
  author    = {Dimitri P. Bertsekas},
  title     = {Lessons from {AlphaZero} for Optimal, Model Predictive, and Adaptive Control},
  journal   = {Athena Scientific Reports},
  year      = {2021}
}

@inproceedings{shani2020,
  author    = {Shani, Lior and Efroni, Yonathan and Mannor, Shie},
  title     = {Adaptive Trust Region Policy Optimization: Global Convergence and Faster Rates for Regularized {MDPs}},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2020}
}
```

**Step 3: Remove old tex files (keep backups)**

```bash
cd /Users/pranjal/Code/rl/ch02_planning_learning/tex
rm planning_learning.tex planning_learning_alt.tex planning_learning_theory.tex unified_planning_learning.tex
```

**Step 4: Commit the new chapter**

```bash
git add ch02_planning_learning/tex/planning_learning_v2.tex docs/refs.bib
git commit -m "feat: consolidate ch02 into single theory chapter (planning_learning_v2.tex)"
```

---

### Task 9: Compile and Verify

**Files:**
- Read: `docs/ch02_planning_learning.pdf` (output)

**Step 1: Compile chapter**

```bash
cd /Users/pranjal/Code/rl/docs && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/planning_learning_v2}\input{compile_chapter}"
```

**Step 2: Run bibtex**

```bash
cd /Users/pranjal/Code/rl/docs && bibtex ch02_planning_learning
```

**Step 3: Recompile twice**

```bash
cd /Users/pranjal/Code/rl/docs && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/planning_learning_v2}\input{compile_chapter}" && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/planning_learning_v2}\input{compile_chapter}"
```

**Step 4: Verify page count**

```bash
pdfinfo /Users/pranjal/Code/rl/docs/ch02_planning_learning.pdf | grep Pages
```

Expected: Pages: 8-12 (target ~10)

**Step 5: Check for undefined references**

```bash
grep -i "undefined" /Users/pranjal/Code/rl/docs/ch02_planning_learning.log | head -20
```

Expected: No critical undefined references

**Step 6: Final commit**

```bash
git add docs/ch02_planning_learning.pdf
git commit -m "docs: compiled ch02 theory chapter (~10 pages)"
```

---

## Verification Checklist

After completion, verify:

- [ ] 4 old tex files archived in `backups/` with timestamps
- [ ] Single `planning_learning_v2.tex` exists
- [ ] Old tex files removed from main directory
- [ ] Chapter compiles without errors
- [ ] Page count is ~10 pages (8-12 acceptable)
- [ ] All 8 core algorithms appear in FRAP table
- [ ] Cross-references to Chapter 1 resolve
- [ ] No simulation content in prose
- [ ] Bibliography entries exist for all citations

---

## Post-Implementation

After all tasks complete:

1. Update `CLAUDE.md` tasklist to mark item 5 as complete
2. Update `changelog.md` with entry for chapter consolidation
