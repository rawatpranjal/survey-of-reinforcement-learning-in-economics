# Offline RL Pricing Sim — 7-Point Audit

Audit for `ch08_offline_rl/sims/offline_rl_pricing.py`. Seven offline-RL methods on a perishable inventory pricing MDP with demand regime switching. Updated 2026-05-18 after refactor to per-paradigm `compute_or_load` caching plus the addition of Decision Transformer (Chen 2021) and return-conditioned supervised learning (Emmons 2022). The earlier monolithic-cache version comparing FQI/CQL/IQL/BCQ/BC remains the empirical anchor; the two new supervised-conditioning methods are the supervised-alternative-to-pessimism family that the new §sec:dt_rvs subsection of `offline_rl.tex` introduces.

## 1. Algorithm Identity

The chapter compares two algorithmic families on the same offline dataset. The first family, pessimism-based value learning, holds the value function as the central object and modifies the Bellman backup to be conservative outside data support. The second family, return-conditioned supervised learning, drops the value function entirely and treats the policy as a supervised model conditioned on a target return.

### DP Oracle (knows true MDP)

Exact backward-induction value iteration on the tabular MDP. Returns $V^\star, \pi^\star$ for all $(i, d, t)$. The DP oracle is the upper bound for all offline methods.

### Behavioral Cloning (BC)

Cross-entropy supervised learning of $\pi(a \mid s)$ from the logged $(s, a)$ pairs. No value learning, no return conditioning. The BC baseline reproduces the behavioral policy and is the floor that any genuine offline-RL method must beat to claim it learned something beyond imitation.

### FQI (Ernst et al. 2005)

Fitted Q-iteration with neural-network function approximation. Update rule: $Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$ regressed on the logged dataset. No pessimism, no behavioral constraint. Serves as the unconstrained baseline that exhibits the overestimation cascade when offline coverage is incomplete.

### CQL (Kumar et al. 2020)

Conservative Q-Learning with $\alpha = 0.1$. Adds a regularizer pushing down $Q$-values at out-of-data actions while pulling up $Q$-values at data actions: $\mathcal{L}_{\text{CQL}} = \alpha (\mathbb{E}_{s \sim \mathcal{D}}[\log \sum_a \exp Q(s, a)] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)])$. Soft-target Q-network with EMA rate 0.005.

### IQL (Kostrikov et al. 2022)

Implicit Q-Learning with expectile $\tau = 0.7$. Avoids querying $Q$ at OOD actions entirely by learning a separate $V(s)$ via expectile regression of $Q(s, a) - V(s)$. The Q-network is then fit with $r + \gamma V(s')$ as the target. Soft-target Q-network with EMA rate 0.005.

### BCQ (Fujimoto et al. 2019)

Batch-Constrained Q-Learning with threshold $\tau = 0.3$. Restricts the policy to actions $a$ with behavioral probability above $\tau$ times the most likely action. Pre-trains a behavior cloning model, then runs FQI with the action set masked to the constraint at both the Bellman target and the policy step.

### DT (Chen et al. 2021)

Decision Transformer. Tokenizes each trajectory as $(\widehat R_t, s_t, a_t)$ triples with $\widehat R_t = \sum_{t' \geq t} r_{t'}$ the return-to-go. A small causally-masked transformer (2 layers, 4 heads, $d_{\text{model}} = 64$, context length $K = 10$) predicts the next action token given the last $K$ context tokens. Trained by cross-entropy on action labels from random subwindows of the trajectory data. Adam lr $3 \times 10^{-4}$, 300 gradient steps to match the budget of FQI's $N_{\text{FQI}} = 200$ iterations $\times$ inner steps.

### RvS (Emmons et al. 2022)

Return-conditioned supervised learning. An MLP with 4-dim input $(s_{\text{normalized}}, \widehat R_{\text{normalized}})$ mapped to action logits via two hidden layers of width 128. Cross-entropy on action labels. Same training budget as BC and DT.

**Deployment protocol shared by DT and RvS.** At test time, the operator specifies a target return $R^\star$. The model is primed with $\widehat R_0 = R^\star$ and the current state. After each env step, the realized reward is subtracted from $R^\star$ and the next state appended. Target return chosen as `dp_init_val` (the oracle return at the start state $(\text{MAX\_INVENTORY}, d_{\text{init}}, H)$). This is an extrapolation request because the behavioral policy's typical return is far below the oracle. The choice is the strongest possible stress test of return-conditioning, and matches the spirit of the DT paper which uses near-optimal target returns in their D4RL evaluation.

## 2. Environment / MDP Fidelity

Perishable inventory pricing MDP per `offline_rl_pricing.py` lines 29-50. No changes from the pre-refactor version.

- **State.** $(i, d, t)$ with inventory $i \in \{0, \ldots, 30\}$, demand regime $d \in \{0, 1, 2, 3\}$, time remaining $t \in \{0, \ldots, 20\}$.
- **Action.** Price $p \in \{1, 2, \ldots, 10\}$.
- **Demand.** $Q \sim \text{Poisson}(\lambda_0[d] \cdot e^{-0.15 p})$, $\lambda_0 = (1.5, 3.0, 5.0, 8.0)$.
- **Reward.** $r_t = p_t \cdot \min(Q_t, i_t)$ during episode; terminal salvage cost $-2.00$ per unsold unit.
- **Transition.** Inventory decrements by units sold; demand regime evolves under a $4 \times 4$ Markov chain with diagonal persistence 0.6.
- **Horizon.** $H = 20$, $\gamma = 1.0$ (finite horizon).

**Behavioral policy.** Maximum price $p = 10$ with probability $0.85$, uniform random over all prices with probability $0.15$. Concentration at $p = 10$ ensures distributional shift since the optimal policy adapts price to inventory and time remaining.

## 3. Data Integrity

Per-paradigm caching via `sims.sim_cache.compute_or_load`. Cache keys: `'shared'`, `'BC'`, `'FQI'`, `'CQL'`, `'IQL'`, `'BCQ'`, `'DT'`, `'RvS'`. Each is hashed against its own config dict that inherits from `SHARED_CONFIG`. Changing `CQL_ALPHA` invalidates CQL's cache only. Changing `N_SEEDS` cascades to all caches.

`compute_shared()` returns `{'dp_policy': ..., 'dp_value': ..., 'dp_init_val': ..., 'offline_datasets': [per-seed list of episodes]}` where each episode preserves the $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ sequence. The Q-methods flatten this to single transitions; DT and RvS consume it as trajectories.

The path from compute to table is `compute_data()` → per-paradigm `compute_or_load` → `{'returns_per_seed': np.array(N_SEEDS,), 'mean': float, 'se': float, 'pct_optimal': float}` → table writer reads from this dict. No hardcoded numbers anywhere downstream of `compute_data`.

## 4. Comparison Fairness

- **Same offline dataset across all methods.** For seed $s$, all seven trained methods (BC, FQI, CQL, IQL, BCQ, DT, RvS) see the exact same 500 episodes generated by `generate_offline_data(N_OFFLINE_EPISODES, np.random.RandomState(s))`.
- **Same evaluation episodes.** Each method is evaluated for 1000 episodes against the live environment with `np.random.RandomState(seed + 10000)` to fix the eval-time noise sequence per seed.
- **Same number of seeds.** All seven methods run on 20 seeds.
- **Comparable training budgets.** FQI/CQL/IQL/BCQ use `N_FQI_ITERATIONS = 200` outer loops × ~3 inner gradient steps. BC/DT/RvS use 300 gradient steps. Direct comparison of wall time is not the comparison object; the comparison is final policy value at convergence.
- **DT/RvS target return.** Fixed at `dp_init_val` for all seeds. The choice is documented and treated as a hyperparameter, not as test-time access.

## 5. Theoretical Sanity Checks

Predictions to verify after the run, in rank order by expected mean return:

- **DP Oracle** at 100% by construction. If anything beats it, there is a bug (most likely in evaluation).
- **CQL, IQL** above BC (~88%) and above FQI. Pessimism-based methods exploit the 15% behavioral noise to discover state-adapted pricing.
- **BCQ** near BC. BCQ's action constraint masks out everything except $p = 10$ most of the time, so it cannot improve on the behavioral.
- **FQI** below BC. The overestimation cascade under sparse coverage compounds geometrically; this is the chapter's motivating result.
- **DT, RvS.** Expected band: $40$–$95\%$ of optimal. Both methods are asked to extrapolate to a return ($\approx 280$ at start state) far above the typical behavioral return ($\approx 100$ from always playing $p = 10$). The `Brandfonbrener2022` critique applies: in stochastic environments, conditioning on a high return tilts toward high-tail actions rather than high-expectation actions. Expect DT and RvS to land between BC and the pessimism methods, possibly underperforming both, possibly competitive with CQL/IQL.

If DT outperforms the DP Oracle, the eval protocol leaked the target return into the reward stream. If RvS exactly matches BC, the return-conditioning input was ignored (verify `RVS_HIDDEN_DIM` is large enough and the return is included in the input vector).

## 6. No Information Leakage

The agent never has access to the DP oracle's policy, $V^\star$, or the true MDP transition / reward parameters during training. The only training input is the offline dataset.

The target return $R^\star = \text{dp\_init\_val}$ used by DT and RvS at deployment is a hyperparameter, not test-time access. It is computed once from the DP oracle and stored in `shared`. The agent uses it as a number the operator supplies, the same way an operator would specify "I want to achieve at least $X$ profit" in production. The chapter writeup documents this transparently.

No future states or rewards are used in any update rule. Each $Q$-update at $(s, a, r, s')$ uses the data point's own $s'$ and the current $Q$-network estimate at $s'$, not the true continuation value. DT and RvS train on the realized returns from the logged trajectory, which is fair use of the offline data.

## 7. Seed and Reproducibility

- `np.random.seed(seed)` and `torch.manual_seed(seed)` set at the top of each training function for every seed in `range(N_SEEDS) = range(20)`.
- Dataset RNG: `np.random.RandomState(seed)`. Eval RNG: `np.random.RandomState(seed + 10000)`. These are separate streams to avoid the eval randomness depending on training randomness.
- Twenty seeds, mean ± standard error reported per method.
- Config hashes in cache files identify which hyperparameter version produced each cached result.

The shared and per-paradigm config dicts are version-tagged via a `version` key. Bumping the version forces cache invalidation on first run.

## Risks and known limits

- **DT and RvS may match or trail BC on this small env.** Per literature and the Brandfonbrener critique, these methods shine on long-horizon, high-dimensional, partially-observable tasks. Perishable inventory pricing is short-horizon, discrete-state, fully-observed. The chapter prose must report the result honestly.
- **The transformer in DT is small (≈50k parameters).** Fine for a 20-step horizon and 4-dim per token. A bigger model is unlikely to help on this env and would risk overfitting the 10k-transition dataset.
- **The DT context window $K = 10$ is half the horizon.** Long enough to see one regime transition typically but short enough that the transformer doesn't trivially memorize trajectories.
- **The DT/RvS target return is fixed at `dp_init_val`.** Sensitivity to this choice is not explored in the main result. A natural follow-up is to sweep $R^\star \in \{\text{mean behavioral return}, \text{max observed return}, \text{dp\_init\_val}\}$. Deferred.
