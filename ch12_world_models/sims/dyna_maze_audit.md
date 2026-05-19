# Dyna Maze Sim — 7-Point Audit

Audit for `ch12_world_models/sims/dyna_maze.py`. Sutton's blocking-maze experiment with five agents on a $6 \times 9$ gridworld. The sim is the chapter's first empirical demonstration that the 1990 Dyna idea works. Updated 2026-05-18 after a stress-test pass: Dyna-Q+ now initializes untried actions at every visited state (faithful Sutton-Barto §8.3); the Schmidhuber baseline now learns reward from observed transitions rather than hard-coding it from the goal location.

## 1. Algorithm Identity

The chapter's framing in §3 introduces Dyna-Q in proper notation. The sim takes the algorithm box at face value and runs it.

### Q-learning (K = 0): the direct-RL baseline

Watkins-style tabular Q-learning with $\varepsilon$-greedy exploration. The update rule is
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)].$$
This is "Dyna with the planning step removed" and serves as the no-model baseline.

### Dyna-Q (K = 5 and K = 50): planning amplification

Sutton 1990 Algorithm box: after each real step, sample $K$ previously-visited $(s, a)$ pairs, query the tabular model $\widehat r(s, a)$ and $\widehat P(\cdot \mid s, a) = \delta_{s'}$ (the last-observed transition), and apply the Q-update to each. $K = 5$ and $K = 50$ are the headline values from Sutton's original gridworld paper. The model is deterministic because the env is deterministic; the planning step amplifies the information content of each real transition by replaying it with the latest Q estimates.

### Dyna-Q+ (K = 50): primitive curiosity for environment change

Sutton & Barto Ch 8.3 extension. Planning targets receive an exploration bonus
$$\widehat r_\text{bonus}(s, a) = \widehat r(s, a) + \kappa \sqrt{\tau(s, a)},$$
where $\tau(s, a)$ is the number of time steps since $(s, a)$ was last visited and $\kappa$ is a small constant. Encourages revisiting $(s, a)$ pairs whose model has gone stale. When the wall flips at $t = N_1$, Dyna-Q+ explores the previously-closed corridor and discovers the new shortcut, while plain Dyna-Q continues to plan into the closed corridor using the now-stale model.

Sutton & Barto §8.3 specifies a second detail: when state $s$ is first visited, *all* actions $a'$ enter the model with $\tau(s, a') = 0$. Otherwise the bonus only modulates the value of already-tried $(s, a)$ pairs and cannot drive discovery of untried actions. The implementation registers all four actions per state with default $(r = 0, s' = s, t_\text{last} = 0)$ on first visit; the bonus $\kappa \sqrt{t_\text{step}}$ then grows monotonically for untried actions and eventually pulls the greedy policy toward them.

This is the chapter's first empirical handshake between the Dyna line and the Schmidhuber 1990 / Pathak 2017 curiosity line.

### Schmidhuber 1990 (controller + model neural networks)

Two small MLPs trained jointly. The model $M_\theta$ maps a one-hot state-action pair to (next-state logits, scalar reward); both heads are trained from the replay buffer with cross-entropy on next state and MSE on reward. The controller $C_\phi$ maps a one-hot state to action logits; it is trained by REINFORCE on imagined rollouts of horizon $H_\text{plan}$ from random replay states under the learned model, with a moving-average baseline and entropy regularization. Reward in the rollout comes from the learned reward head — not from the known goal location. The agent does not see the goal during training.

## 2. Environment / MDP Fidelity

Sutton-Barto blocking maze, $6 \times 9$ grid, deterministic, episodic.

- **State space.** Cells $\{(r, c) : 0 \le r \le 5, 0 \le c \le 8\}$ minus the wall cells.
- **Start.** $S = (5, 3)$.
- **Goal.** $G = (0, 8)$, reward $+1$ on arrival; episode terminates.
- **Actions.** Up, down, left, right. Deterministic; attempting to enter a wall or step off the grid is a no-op (agent stays in place, episode continues).
- **Reward.** $0$ on every step except $+1$ on goal arrival.
- **Discount.** $\gamma = 0.95$.
- **Phase 1 wall.** Row $r = 2$, columns $c \in \{0, 1, 2, 3, 4, 5, 6, 7\}$ are blocked. Column $8$ is open. The agent must go up-right around column 8 to reach the goal. Phase 1 runs for $N_1 = 1000$ environment steps.
- **Phase 2 wall.** Row $r = 2$, columns $c \in \{1, 2, 3, 4, 5, 6, 7, 8\}$ are blocked. Column $0$ is open. Phase 2 runs for an additional $N_2 = 2000$ environment steps, for a total of $N_1 + N_2 = 3000$ steps. The flip is automatic at $t = N_1$.
- **Episode termination.** On goal, or after a step cap of $200$ per episode (cap is generous; the optimal path is $\le 14$ steps).

This is the canonical Sutton-Barto Ch 8 blocking-maze; matches Figure 8.4 of the textbook.

## 3. Data Integrity

The headline metric is *cumulative reward over the cumulative environment-step axis*, which is the metric Sutton plots in Figure 8.4. Each real env step contributes either $0$ or $+1$ to the cumulative reward; planning steps do not count toward the env-step axis even though they contribute Q-updates. Per-step cumulative reward is logged per (agent, seed, t), and the curves plotted are means and standard errors across seeds. The table reports the cumulative reward at $t = N_1 + N_2 = 3000$ per agent.

## 4. Comparison Fairness

- All four agents see the same env, same seed for env transitions (the env is deterministic so this is a tie; seeds affect only $\varepsilon$-greedy and the planning-sample selection).
- All four agents have the same env-step budget ($N_1 + N_2 = 3000$).
- The Q-learning baseline ($K = 0$) does no planning between real steps; Dyna-Q and Dyna-Q+ do $K$ planning updates per real step. Planning compute is not bounded; this is the same convention as Sutton 1990 and is justified pedagogically (the comparison is *information per real interaction*, not wall-clock).
- All tabular agents share $\varepsilon = 0.1$ greedy parameter, $\alpha = 0.1$ learning rate, $\gamma = 0.95$ discount, initial $Q \equiv 0$. Dyna-Q+ adds the bonus parameter $\kappa = 10^{-4}$. Schmidhuber 1990 uses neural networks with its own optimizer; the comparison is meaningful only at the level of cumulative reward per env step.

## 5. Theoretical Sanity Checks

Before declaring trustworthy:

1. **Q-learning ($K = 0$) reaches a stable cumulative-reward slope only after many real steps.** Expected: slope approaches the optimal-path reward rate (one goal per ~14 steps, so slope $\to 1/14 \approx 0.07$ asymptotically) over the course of Phase 1. Sutton 1990 reports near-optimal in $\sim 25$ episodes for $K = 0$ vs $\sim 3$ episodes for $K = 50$.
2. **Dyna-Q $K = 50$ reaches near-optimal slope within the first $\sim 50$ real steps**, an order of magnitude faster than $K = 0$. This is the headline of Sutton 1990 Figure 4.
3. **Both Dyna-Q variants stall after the wall flips.** Dyna-Q ($K = 50$) without the bonus has a learned model that still places the path through the (now-blocked) corridor; the planner reinforces this stale path. Dyna-Q+ with the bonus eventually drives the agent through the (now-open) opposite corridor and recovers a positive slope.
4. **The final cumulative-reward gap between Dyna-Q and Dyna-Q+ in Phase 2** is the key visual signature of the experiment. Sutton & Barto Figure 8.5 shows Dyna-Q+ continuing to gain while Dyna-Q plateaus.

If any of these fail, the audit captures the deviation; the prose reports what is observed, not what was hoped.

## 6. No Information Leakage

- All agents see only the current state and the most recent observed $(s, a, r, s')$ tuple. None of them sees the wall pattern, the start/goal location, the phase index, or the time step.
- The phase switch is administered by the environment, not announced to the agents. From the agent's perspective, a transition that was open at $t = N_1 - 1$ is closed at $t = N_1$; the only signal is the next-state observation differing from the model's prediction.
- Dyna-Q+'s curiosity bonus is computed entirely from $\tau(s, a)$, the agent's own visit counts; it does not use the env's phase.

## 7. Seed and Reproducibility

- $N_{\text{seeds}} = 30$. Each seed sets `np.random.default_rng(seed)` once at agent reset; the env is deterministic so its only randomness is the initial agent placement (always at $S$). The seeds drive $\varepsilon$-greedy choice and the planning-step's $(s, a)$ sampling.
- All curves report mean and standard error of the mean across the 30 seeds.

## Hyperparameter Reference

| Item            | Value      | Source / rationale                                     |
| --------------- | ---------- | ------------------------------------------------------ |
| Grid            | 6 × 9      | Sutton & Barto Ch 8.2                                  |
| $S, G$          | $(5,3), (0,8)$ | Sutton & Barto Fig 8.4                             |
| Phase 1 wall    | row 2, c=0..7  | "Original maze"                                    |
| Phase 2 wall    | row 2, c=1..8  | "Blocking maze" flip                               |
| $N_1, N_2$      | 1000, 2000 | Total interaction budget 3000 steps                    |
| $\gamma$        | 0.95       | Standard                                               |
| $\alpha$        | 0.1        | Sutton & Barto Ch 8                                    |
| $\varepsilon$   | 0.1        | Sutton & Barto Ch 8                                    |
| $K$             | 0, 5, 50   | Sutton 1990 headline values                            |
| $\kappa$ (bonus) | $10^{-4}$ | Sutton & Barto Ch 8.3 (small value; aggressive bonuses poison exploitation) |
| Schmidhuber hidden | 32        | Two MLPs, model + controller; one ReLU layer       |
| Schmidhuber $H_\text{plan}$ | 10 | Imagined rollout horizon                          |
| Schmidhuber $K_\text{plan interval}$ | 10 | Real steps between controller updates    |
| Episode cap     | 200 steps  | Loose upper bound on a single episode length           |
| $N_{\text{seeds}}$ | 30      | Above project minimum, accommodates higher variance in tabular control |

## Sign-off

Written before implementation. If implementation deviates, update this doc and document the deviation in the §3 prose.
