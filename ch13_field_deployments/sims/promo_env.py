# ch13 capstone: the hidden true DGP for the OPE-reliability study.
#
# High-dimensional dynamic targeted-promotions MDP. Each step one consumer arrives
# with a K-dim context c; the firm sets a targeted discount d on a fixed base price;
# buy/no-buy is a binary logit whose price sensitivity alpha(c) is a low-dim index
# buried among noise features (so the optimal policy is high-dimensional and no
# tabular DP applies). A market-level reference-discount r evolves as a running mean
# of offered discounts (agrawal2024 averaging reference), so over-promoting trains
# customers to expect discounts and myopic promotion is strictly suboptimal: the
# dynamics are what make this an RL problem rather than a contextual bandit.
#
# numpy + gymnasium only (no torch / scope-rl), so the env stays fast and unit-testable.
# The demand equation lives in ONE place (buy_prob_vec); the scalar gym-API path and
# the batched rollout both call it, so there is no second copy to drift.
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Action index -> discount depth on the fixed base price. Index 0 is no-promo.
DISCOUNTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
N_ACTIONS = len(DISCOUNTS)


@dataclass
class PromoConfig:
    K: int = 8  # context dimension (swept for the curse-of-dim panel)
    n_signal: int = 2  # context coords that actually drive price sensitivity
    alpha_base: float = 1.5  # baseline price sensitivity
    theta_scale: float = (
        1.5  # signal strength of the buried index (context heterogeneity)
    )
    p_base: float = 1.0  # fixed base price (normalized)
    cost: float = 0.30  # marginal cost -> full-price margin 0.70
    delta: float = 1.0  # buy-utility intercept
    beta_ref: float = 2.5  # reference sensitivity (offered d vs reference r)
    loss_aversion: float = 1.5  # shortfall (d<r) hurts more than surplus (d>r) helps
    r_init: float = 0.10  # initial reference discount consumers expect
    gamma: float = 0.95  # discount factor (LTV horizon)
    horizon: int = 40  # consumers per episode
    ctx_clip: float = 4.0  # context support bound

    def theta(self):
        # Sparse unit-scaled index weights: only the first n_signal coords matter.
        # Fixed to the DGP (seeded by K so each dimension setting is self-consistent).
        rng = np.random.default_rng(1234 + self.K)
        th = np.zeros(self.K)
        th[: self.n_signal] = rng.normal(size=self.n_signal)
        nrm = np.linalg.norm(th)
        return th / nrm if nrm > 0 else th


class PromoEnv(gym.Env):
    """Gymnasium env; the observation is [context (K), reference r, time t/T]."""

    metadata = {"render_modes": []}

    def __init__(self, config: PromoConfig | None = None, seed: int = 0):
        super().__init__()
        self.cfg = config or PromoConfig()
        self._theta = self.cfg.theta()
        self.action_space = spaces.Discrete(N_ACTIONS)
        c = self.cfg.ctx_clip
        low = np.concatenate([np.full(self.cfg.K, -c), [0.0, 0.0]]).astype(np.float32)
        high = np.concatenate([np.full(self.cfg.K, c), [DISCOUNTS[-1], 1.0]]).astype(
            np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._c = None
        self._r = self.cfg.r_init
        self._t = 0

    # --- true demand model (canonical, batched; the single source of truth) ---
    def buy_prob_vec(self, C, r, d):
        """C: (B,K) contexts, r: (B,) references, d: (B,) discount depths -> (B,) probs."""
        C = np.atleast_2d(C)
        r = np.atleast_1d(r).astype(float)
        d = np.atleast_1d(d).astype(float)
        z = (C @ self._theta) / np.sqrt(self.cfg.n_signal)
        alpha = self.cfg.alpha_base * np.exp(self.cfg.theta_scale * z)
        p_eff = self.cfg.p_base * (1.0 - d)
        gap = d - r
        ref = self.cfg.beta_ref * np.where(gap >= 0, gap, self.cfg.loss_aversion * gap)
        u = self.cfg.delta - alpha * p_eff + ref
        return 1.0 / (1.0 + np.exp(-u))

    def buy_prob(self, c, r, d):
        return float(self.buy_prob_vec(np.asarray(c)[None, :], [r], [d])[0])

    def margin(self, d):
        return self.cfg.p_base * (1.0 - np.asarray(d)) - self.cfg.cost

    # --- gym API (scalar; used for Scope-RL logging and gym-compliance) ---
    def _obs(self):
        return np.concatenate([self._c, [self._r, self._t / self.cfg.horizon]]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._r = self.cfg.r_init
        self._t = 0
        self._c = np.clip(
            self._rng.normal(size=self.cfg.K), -self.cfg.ctx_clip, self.cfg.ctx_clip
        )
        return self._obs(), {}

    def step(self, action):
        d = float(DISCOUNTS[int(action)])
        p_buy = self.buy_prob(self._c, self._r, d)
        buy = self._rng.random() < p_buy
        reward = float(self.margin(d)) if buy else 0.0
        self._r = (self._t * self._r + d) / (self._t + 1)  # averaging reference
        self._t += 1
        terminated = self._t >= self.cfg.horizon
        if not terminated:
            self._c = np.clip(
                self._rng.normal(size=self.cfg.K), -self.cfg.ctx_clip, self.cfg.ctx_clip
            )
        return (
            self._obs(),
            reward,
            terminated,
            False,
            {"p_buy": p_buy, "buy": bool(buy)},
        )


# ---------------------------------------------------------------------------
# Batched policies: obs_batch (B, K+2) -> actions (B,). Used for logging behavior,
# the deployment baseline, and vectorized true-value rollouts. A trained d3rlpy
# candidate plugs in via a thin wrapper calling model.predict(obs_batch).
# ---------------------------------------------------------------------------
def uniform_batch(env, obs, rng):
    """A/B behavior: uniform over discounts (propensity 1/N per action)."""
    return rng.integers(N_ACTIONS, size=obs.shape[0])


def myopic_batch(env, obs, rng=None):
    """Model-based myopic incumbent: argmax immediate expected margin, ignoring
    the effect of today's discount on tomorrow's reference. The simple baseline."""
    C = obs[:, : env.cfg.K]
    r = obs[:, env.cfg.K]
    er = np.stack(
        [
            env.buy_prob_vec(C, r, np.full(obs.shape[0], d)) * env.margin(d)
            for d in DISCOUNTS
        ],
        axis=1,
    )
    return er.argmax(axis=1)


def constant_batch(action_idx):
    def pol(env, obs, rng=None):
        return np.full(obs.shape[0], int(action_idx))

    return pol


def softened_batch(base_pol, epsilon):
    """Epsilon-greedy softening of a batched policy: with prob 1-epsilon play
    base_pol's action, else uniform. Matches EpsilonGreedyHead's action law, so
    idealized coverage/occupancy rollouts describe the SAME softened policy OPE
    evaluates (not its deterministic greedy core)."""

    def pol(env, obs, rng):
        greedy = np.asarray(base_pol(env, obs, rng)).astype(int)
        B = obs.shape[0]
        explore = rng.random(B) < epsilon
        random_a = rng.integers(N_ACTIONS, size=B)
        return np.where(explore, random_a, greedy)

    return pol


def vec_rollout_value(env, batch_policy, n_episodes, seed):
    """True discounted-return field value of a batched policy in the hidden env
    (Monte Carlo over n_episodes run in lockstep; no oracle). Returns (mean, se)."""
    cfg = env.cfg
    B, K, T, g = n_episodes, cfg.K, cfg.horizon, cfg.gamma
    rng = np.random.default_rng(seed)
    C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
    r = np.full(B, cfg.r_init)
    total = np.zeros(B)
    for t in range(T):
        obs = np.concatenate([C, r[:, None], np.full((B, 1), t / T)], axis=1).astype(
            np.float32
        )
        a = np.asarray(batch_policy(env, obs, rng)).astype(int)
        d = DISCOUNTS[a]
        p = env.buy_prob_vec(C, r, d)
        buy = rng.random(B) < p
        total += (g**t) * np.where(buy, env.margin(d), 0.0)
        r = (t * r + d) / (t + 1)
        if t < T - 1:
            C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
    return float(total.mean()), float(total.std(ddof=1) / np.sqrt(B))
