# Audit: ch12_world_models/sims/dyna_maze.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch12_world_models/tex/s03_dyna_q.tex`
**Cited paper PDFs read:** `papers/SCHMIDHUBER1990_FKI.pdf` (present), Sutton-Barto 2018 §8.2-8.3 (not in `papers/` directly; referenced via `\citet{sutton2018}`), Sutton 1990 (not in `papers/`).

## 1. Algorithm Identity

Dyna-Q reference (Sutton 1990 / Sutton-Barto 2018 §8.2, Algorithm box "Tabular Dyna-Q"):

1. e-greedy action selection from Q,
2. observe (r, s'),
3. Q-update on (s, a, r, s'),
4. Model[s,a] <- (r, s'),
5. n planning loops: sample previously-observed (s,a), retrieve (r, s') from model, apply Q-update.

`DynaAgent` (lines 90-158):
- `Q = np.zeros((n_states, n_actions))` — tabular Q. Correct.
- `act()` — e-greedy with uniform tie-break. Correct.
- `_q_update()` — `r + gamma * max(Q[s']) - Q[s,a]` then `Q[s,a] += alpha * td`. Correct standard Q-learning update.
- `observe()` — step counter, direct Q-update, register (s,a) to visited_set+visited_sa, model[(s,a)] = (r, s', t_step). Correct.
- Planning loop (lines 148-158): samples K indices uniformly from `visited_sa`, applies Q-update on cached (r, s'). Correct.

Dyna-Q+ (Sutton-Barto §8.3):
- Augmented reward in planning: `r + kappa * sqrt(tau)` where tau = time since (s,a) last observed.
- Sutton-Barto specifies "actions that had never been tried before from a state were allowed to be considered in the planning step" — i.e. for every visited state, register *all* four actions with `tau` measured from start. Lines 140-146 do exactly this: when bonus=True, registers all actions at s and s_next with `t_last=0`, model entry `(0.0, state, 0)` (self-loop, zero reward). This is the Sutton-Barto §8.3 convention.

Notable detail: `bonus_kappa = 1e-4` (Sutton-Barto uses 1e-3 to 1e-4 range; 1e-4 is on the low end but in the literature range). For deterministic shortest-path maze with reward 1, kappa * sqrt(2000) ≈ 4.5e-3, which is small relative to the goal reward but persistent. Acceptable.

Schmidhuber 1990 controller-model (`SCHMIDHUBER1990_FKI.pdf`): the original Schmidhuber paper proposes two coupled networks — a *model* network predicting (s', r) from (s, a), and a *controller* network selecting a from s, with the controller trained via gradients flowing through the differentiable model. The script's variant uses REINFORCE on imagined rollouts rather than direct backprop-through-model, which is a legitimate stochastic-policy variant but is not strictly the 1990 formulation. The reward inside imagined rollouts comes from `r_pred` (the learned reward head). The agent is labeled "Schmidhuber 1990 (NN)" — calling it "faithful realization" (as the tex does on line 57 of s03_dyna_q.tex) is slightly aspirational; it's REINFORCE on imagined trajectories under a learned model, which is closer to a modern policy-gradient-with-world-model setup. The tex text does note "a REINFORCE policy-gradient estimator," so the description matches the code.

Verdict: Dyna-Q / Dyna-Q+ identity is correct including the §8.3 untried-action initialization. Schmidhuber agent is a defensible REINFORCE-style instantiation rather than the exact 1990 backprop-through-model variant, but the tex describes it accurately.

## 2. Environment / MDP Fidelity

Sutton-Barto 2018 Ex 8.2 blocking maze (Figure 8.4):
- 6 rows x 9 cols, deterministic gridworld.
- Wall in one row (in S-B the wall is drawn on the third row from the top); opening on the right in Phase 1, opening on the left in Phase 2.
- Reward = 0 elsewhere, +1 on goal arrival. gamma = 0.95.
- "After 1000 time steps the wall is shifted" — total ~3000 steps.

Script (`dyna_maze_env.py`):
- `N_ROWS=6, N_COLS=9`, `START=(5,3)`, `GOAL=(0,8)`. Matches S-B layout.
- `WALL_ROW=2`, Phase1 wall cols (0..7) leaving col 8 open; Phase2 wall cols (1..8) leaving col 0 open. Matches S-B figure exactly.
- Reward: 0 everywhere, +1 on goal; deterministic transitions; bumping wall is no-op. Correct.
- `t_switch=1000, t_total=3000, episode_cap=200` — matches S-B Ex 8.2 setup.
- Phase-switch is automatic in `_is_wall` based on `t_global`. Agent not signaled. Correct.

One minor concern: episode_cap=200 is not specified in S-B; S-B simply lets the agent run with episodes terminating only on goal-reach within the 3000-step budget. The cap is a safety net to prevent a truly stuck e-greedy Q-learning agent from wasting the entire budget in one episode. With reward = 0 on truncation, this matters: an early-truncated episode for plain Q-learning is functionally identical to one that would have eventually reached the goal — except that the partial value backups during that episode are now lost since the agent didn't complete the path. This *may* slightly disadvantage plain Q-learning relative to the un-capped S-B variant, but it applies identically to all agents (they all share the same env), so cross-agent comparisons remain fair. The cap is not declared in the tex setup paragraph (line 54 cites "per-episode step cap is two hundred") — it IS disclosed, so fine.

## 3. Data Integrity

`compute_data()` (lines 391-402) loops `AGENT_ORDER`, dispatching to `compute_or_load(...)` with per-agent cache keys constructed from the agent name. Each call invokes `compute_agent(cfg, name)` which runs `rollout()` for each seed and stacks `cum_reward_curve` into `curves[seed]`. Means / SEs / phase-1-end / phase-2-gain / final are all computed *from the curves array*, not hardcoded.

`generate_outputs()` reads from the `data` dict and writes the .tex table directly from `data[name]['phase1_final']` etc. No hardcoded "expected" values. The stdout in `dyna_maze_stdout.txt` shows cache hits for three agents and fresh computes for two (Dyna-Q+ K=50 and Schmidhuber), consistent with config-keyed caching.

Per-component config decomposition (ENV_CONFIG -> SHARED_CONFIG -> AGENT_CONFIGS) follows the CLAUDE.md modularity pattern. Changing only `BONUS_KAPPA` would invalidate Dyna-Q+ but not Dyna-Q (which has bonus=False). This is correct provided `compute_or_load` hashes the full config dict — and the cache hits in stdout confirm it does.

## 4. Comparison Fairness

- Same env class (`BlockingMaze`) instantiated identically per rollout (same `t_switch`, `t_total`, `episode_cap`).
- Same total step budget: 3000 environment steps per seed for all five agents.
- Same seeds (`for seed in range(N)`), same `N_SEEDS=30`.
- Same alpha=0.1, gamma=0.95, epsilon=0.1 for all four Dyna variants.
- Schmidhuber agent shares gamma=0.95 and N_SEEDS=30 with the others; it has different hyperparameters (LR, hidden dim, planning interval) because it's a different algorithm — this is expected and standard. It does NOT use epsilon-greedy (it samples from controller softmax with entropy regularization), which is the algorithm's own native exploration mechanism. Fair.

One subtle issue: the Dyna variants count "planning steps" as K per real step, so total compute scales linearly with K. The Schmidhuber agent does one model-network gradient step per real env step PLUS a REINFORCE policy update every 10 steps using 16 imagined trajectories of length 10. Total compute per real env step is roughly: 1 forward+backward through M plus, every 10 steps, 16*10=160 model forward passes + 1 controller update. This is not comparable to Dyna-Q K=50 in wall-clock or in any clean "planning steps" sense. The tex paper acknowledges this by separating the comparison as a 1990-architectures qualitative point rather than a fair head-to-head on compute. The plot shares an x-axis of "environment steps," which is the budget-of-interest and is identical across agents. Fair on sample budget, intentionally not fair on compute. That's defensible.

## 5. Theoretical Sanity Checks

Sutton-Barto Figure 8.5 (blocking maze, K=50, 30 seeds): Dyna-Q+ recovers within ~500 steps post-flip; Dyna-Q is slower to recover but eventually does. Both reach ~14-15 goal-reaches by step 3000. Dyna-Q tends to slightly outperform Dyna-Q+ in cumulative reward at this scale because the bonus dilutes greedy exploitation; this matches the script's results.

Script numerics:
- Dyna-Q (K=50): total 52.0 ± 4.2, Phase 1 end 42.2 ± 5.3, Phase 2 gain 9.8 ± 4.0.
- Dyna-Q+ (K=50): total 47.0 ± 4.4, Phase 1 end 37.4 ± 3.7, Phase 2 gain 9.6 ± 2.8.
- Dyna-Q (K=5): total 39.2, Phase 1 end 34.4, Phase 2 gain 4.8.
- Q-learning (K=0): 3.5 ± 0.5 total. Plain Q-learning barely reaches the goal a few times.
- Schmidhuber: 4.0 ± 0.4, comparable to plain Q-learning.

Hostile-reviewer perspective: the *absolute* numbers (52 vs 14 in S-B Fig 8.5) are higher than canonical because the script uses `gamma=0.95` (helps backups) and 30 seeds with this specific maze, and goal-reaches reset the episode immediately so the cumulative count can climb fast once the path is learned. The *relative ordering* matches theory: K=50 > K=5 >> K=0 in pre-flip, K=50 Dyna-Q ≈ Dyna-Q+ post-flip. The Phase-1 cost for Dyna-Q+ (37.4 vs 42.2, ~5 reward units below plain Dyna-Q) matches the S-B observation that the bonus dilutes exploitation.

One genuine sanity issue: the Phase 2 *gain* for Dyna-Q and Dyna-Q+ are statistically indistinguishable (9.8 ± 4.0 vs 9.6 ± 2.8). S-B Figure 8.5 visually shows Dyna-Q+ recovering faster than Dyna-Q. The script's tex prose acknowledges this on line 60: "the recovery rates are statistically indistinguishable here." This is consistent with the wider literature observation that the Dyna-Q+ vs Dyna-Q recovery gap on the blocking maze is small and seed-noisy; on the *shortcut maze* (S-B Ex 8.3) the gap is larger. With kappa=1e-4 (low end of literature range) and only 30 seeds, indistinguishable Phase-2 gains are believable. A hostile reviewer might push for kappa sweep or shortcut-maze comparison, but the current implementation is not wrong.

The Schmidhuber agent performing near plain Q-learning is theoretically defensible: gradient-based fit of a 32-dim hidden network on 3000 transitions from a sparse-reward 54-state task will not specialize meaningfully before the budget expires. This matches the tex narrative.

## 6. Information Leakage

- `DynaAgent` observes (s, a, r, s') only; model is `dict[(s,a)] -> (r, s', t_last)` populated from observed transitions. No access to true transition kernel. Correct.
- Dyna-Q+ untried-action initialization registers `(state, action) -> (0.0, state, 0)` — a zero-reward self-loop placeholder, NOT the true transition. This is the Sutton-Barto §8.3 trick and is not leakage (the agent is hallucinating "untried actions look like self-loops with zero reward and were observed at t=0", which is a deliberate exploration prior, not privileged information about the env). Correct.
- Schmidhuber agent: `WorldModel` learned by SGD on buffer of observed transitions. Reward inside rollouts is `r_pred` from the learned model. `goal_state_id` is stored but explicitly commented as "for diagnostics only. Never used in training" (line 213) — verified by grep: not referenced anywhere in `_train_m_step`, `_train_c_reinforce`, or `act`. Clean.
- Caching: each seed runs a fresh `BlockingMaze.reset_global()` and a fresh agent (`make_agent(...)`). No state bleed across seeds.

No leakage.

## 7. Seed and Reproducibility

- `N_SEEDS = 30` for all agents. Exceeds the ≥10 threshold.
- Seeds set explicitly: `seed=seed` passed through `make_agent`. `DynaAgent` uses `np.random.default_rng(seed)`. `Schmidhuber1990Agent` calls `torch.manual_seed(seed)` AND uses `np.random.default_rng(seed + 31415)` for numpy randomness. Both PyTorch and numpy seeded.
- Mean and SE computed via `curves.std(ddof=1) / sqrt(N)`. Correct.
- One residual concern: `torch.manual_seed(seed)` is global; if multiple Schmidhuber agents were constructed concurrently in the same process, the second's init would see a different RNG state. But agents are constructed sequentially in `compute_agent`'s seed loop, and `torch.manual_seed` is called at the *start* of `Schmidhuber1990Agent.__init__`, so each seed begins with a deterministic torch RNG state. Acceptable.
- The Schmidhuber agent does NOT call `torch.use_deterministic_algorithms(True)` and uses default CUDA/MPS-related nondeterminism if those backends are active. On CPU PyTorch with default settings this is mostly deterministic but not strictly guaranteed across machines. For audit purposes (30 seeds, modest deviations), this is fine.

## Hostile-Reviewer Summary

The implementation is materially correct: Dyna-Q and Dyna-Q+ follow Sutton-Barto §8.2-8.3 including the untried-action-initialization detail of §8.3. The blocking maze layout (6x9, wall on row 2, start (5,3), goal (0,8), wall flip at t=1000) matches Ex 8.2. Comparison is fair on the sample-budget axis. The Schmidhuber 1990 agent is a defensible REINFORCE-on-imagined-rollouts instantiation, though calling it "faithful" to the 1990 paper is a small stretch (the original used backprop-through-model). 30 seeds, both numpy and torch RNG seeded, mean+SE reported, results-in-rank-order in the table. The Phase-2 recovery gap between Dyna-Q+ and Dyna-Q is statistically indistinguishable (the tex acknowledges this explicitly), so the "Dyna-Q+ recovers faster" headline is muted rather than reproduced — but the tex prose handles this honestly, attributing the small gap to seed noise and the low kappa rather than over-claiming. The episode_cap=200 truncation is a minor deviation from the S-B setup that the tex discloses.

A hostile reviewer would push for: (i) shortcut-maze companion experiment where Dyna-Q+ should clearly beat Dyna-Q, to firm up the "exploration bonus aids recovery" claim that the blocking-maze panel only weakly supports; (ii) a kappa sweep showing the Phase-2 gap as a function of bonus strength; (iii) tighter language around the Schmidhuber agent being a REINFORCE-style variant rather than the original FKI-148 backprop-through-model formulation. None of these are correctness errors; they are scope/strength complaints that a major-revision letter would catch.

**Bullshit score: 15%** — Reviewer 2 catches the Phase-2 statistically-indistinguishable recovery rate undermining the Dyna-Q+ narrative and the slight overstatement of "faithful Schmidhuber 1990," but the algorithms and environment are implemented correctly, seeds and SE protocol are clean, and the tex prose is unusually honest about what the numbers do and don't show.
