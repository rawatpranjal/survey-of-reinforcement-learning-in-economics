# Cobweb with adjustment cost: eight learning paradigms.
# Chapter: World Models and Model-Based Reinforcement Learning. Section: dual
# simulation (cobweb panel).
# Compares oracle, naive, RLS, Q-learning, Arifovic GA, model-based LQ, and two
# model-based linear-Gaussian policy-gradient learners (MB-LG-REINFORCE with a
# score-function gradient, MB-LG-Pathwise with an analytic forward-sensitivity
# gradient) on the self-referential cobweb across three stability regimes.

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import (
    apply_style,
    COLORS,
    BENCH_STYLE,
    FIG_TRIPLE,
)
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()
import matplotlib.pyplot as plt

from cobweb_env import CobwebEnv, solve_oracle_lq, expected_reward  # noqa: F401 (expected_reward is the monkeypatch anchor for the no-leak tests)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "cobweb_paradigms"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGIMES = {
    "stable": dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1),
    "borderline": dict(a=4.0, b=1.0, c=1.0, phi=0.2, sigma=0.1),
    "unstable": dict(a=4.0, b=2.0, c=1.0, phi=0.2, sigma=0.1),
}
SHARED_CONFIG = {
    "GAMMA": 0.95,
    "T_EPISODE": 500,
    "N_SEEDS": 20,
    "Q_MIN": 0.0,
    "Q_MAX": 4.0,
    "REGIMES": REGIMES,
}

RLS_CONFIG = {**SHARED_CONFIG, "INIT_VAR": 100.0}
QL_CONFIG = {
    **SHARED_CONFIG,
    "G_S": 20,
    "G_A": 25,
    "P_MIN": -4.0,
    "P_MAX": 8.0,
    "ALPHA": 0.1,
    "EPS_HI": 0.3,
    "EPS_LO": 0.01,
}
GA_CONFIG = {
    **SHARED_CONFIG,
    "N_POP": 30,
    "L_BITS": 10,
    "P_CROSS": 0.6,
    "P_MUT": 0.0033,
    "GEN_LEN": 10,
}
MBLQ_CONFIG = {**SHARED_CONFIG, "EXPLORE_STD": 0.15, "FIT_EVERY": 1, "WARMUP": 5}
MB_LG_REINFORCE_CONFIG = {
    **SHARED_CONFIG,
    "EXPLORE_STD": 0.15,
    "WARMUP": 5,
    "ENSEMBLE_SIZE": 5,
    "ROLLOUT_HORIZON": 5,
    "N_ROLLOUTS": 10,
    "POLICY_LR": 0.005,
}
MB_PATHWISE_CONFIG = {
    **SHARED_CONFIG,
    "EXPLORE_STD": 0.15,
    "WARMUP": 5,
    "ENSEMBLE_SIZE": 5,
    "ROLLOUT_HORIZON": 5,
    "N_ROLLOUTS": 20,
    "POLICY_LR": 0.05,
}
NAIVE_CONFIG = {**SHARED_CONFIG}
ORACLE_CONFIG = {**SHARED_CONFIG}

# ---------------------------------------------------------------------------
# Paradigm base
# ---------------------------------------------------------------------------


class Paradigm:
    """Each paradigm exposes act(state, t) and observe(state, action, reward, next_state).

    Optional methods used by the deep-analysis pass:
      - get_params(): dict of current parameter estimates (a_hat, b_hat, ...).
                      Returns {} for paradigms without an explicit demand model.
      - greedy_action(q_ref, p_ref): noise-free action at the reference state.
                                      Used to compute policy distance to oracle.
    """

    name = "base"

    def reset(self, regime_params, seed=0):
        pass

    def act(self, state, t):
        raise NotImplementedError

    def observe(self, state, action, reward, next_state):
        pass

    def get_params(self):
        return {}

    def greedy_action(self, q_ref, p_ref):
        return np.nan


# ---------------------------------------------------------------------------
# Oracle (knows true parameters)
# ---------------------------------------------------------------------------


class OraclePolicy(Paradigm):
    name = "Oracle"

    def __init__(self, gamma):
        self.gamma = gamma
        self.lq = None
        self._true_params = None

    def reset(self, regime_params, seed=0):
        self.lq = solve_oracle_lq(
            a=regime_params["a"],
            b=regime_params["b"],
            c=regime_params["c"],
            phi=regime_params["phi"],
            gamma=self.gamma,
        )
        self._true_params = dict(
            a=regime_params["a"],
            b=regime_params["b"],
            c=regime_params["c"],
            phi=regime_params["phi"],
        )

    def act(self, state, t):
        q_prev = state[0]
        return self.lq["K0"] + self.lq["Kq"] * q_prev

    def get_params(self):
        return dict(self._true_params) if self._true_params else {}

    def greedy_action(self, q_ref, p_ref):
        if self.lq is None:
            return np.nan
        return self.lq["K0"] + self.lq["Kq"] * q_ref


# ---------------------------------------------------------------------------
# Naive (myopic, ignores adjustment cost)
# ---------------------------------------------------------------------------


class NaivePolicy(Paradigm):
    """Constant rule: q_t = q_naive across all periods and regimes.

    True no-learning baseline. The fixed value is chosen as the midpoint of
    the optimal steady-state actions across the three regimes (roughly 1.4)
    so the rule is moderately wrong in every regime without being a
    pathological choice. The role is to provide an absolute lower bound for
    'how much does learning buy you?'.
    """

    name = "Naive"

    Q_FIXED = 1.4

    def reset(self, regime_params, seed=0):
        pass

    def act(self, state, t):
        return self.Q_FIXED

    def greedy_action(self, q_ref, p_ref):
        return self.Q_FIXED


# ---------------------------------------------------------------------------
# RLS adaptive learning (Marcet-Sargent 1989)
# ---------------------------------------------------------------------------


class RLSPolicy(Paradigm):
    """Recursive least squares on (a, b) of p = a - b q. Cost params (c, phi)
    assumed known. Each period, the agent solves the LQ-Bellman with point
    estimates and applies the resulting optimal policy.

    This is the most generous version of RLS-as-learner: the agent has
    correct functional form, knows c, phi, and only needs to estimate (a, b).
    """

    name = "RLS"

    def __init__(self, gamma, init_var):
        self.gamma = gamma
        self.init_var = init_var
        self.theta = None  # (a_hat, b_hat); b stored as positive number
        self.R = None  # information matrix
        self.c = None
        self.phi = None
        self.q_min = 0.0
        self.q_max = 4.0

    def reset(self, regime_params, seed=0):
        self.theta = np.array([1.0, 0.5], dtype=np.float64)  # (a_hat, b_hat)
        self.R = (1.0 / self.init_var) * np.eye(2)
        self.c = regime_params["c"]
        self.phi = regime_params["phi"]
        self.q_min = 0.0
        self.q_max = 4.0

    def act(self, state, t):
        q_prev = state[0]
        a_hat, b_hat = self.theta
        # Guard against b_hat <= 0 (would break LQ): cap below.
        b_hat_safe = max(b_hat, 0.05)
        try:
            lq = solve_oracle_lq(
                a=a_hat,
                b=b_hat_safe,
                c=self.c,
                phi=self.phi,
                gamma=self.gamma,
                max_iter=200,
                tol=1e-8,
            )
            q = lq["K0"] + lq["Kq"] * q_prev
        except (ValueError, ZeroDivisionError):
            q = a_hat / (2 * (b_hat_safe + 0.5 * self.c))
        return float(np.clip(q, self.q_min, self.q_max))

    def observe(self, state, action, reward, next_state):
        # Regressor x = (1, -q_t) targeting p_t
        q_t = action
        p_t = next_state[1]
        x = np.array([1.0, -q_t])
        self.R = self.R + np.outer(x, x)
        # Solve linearly: theta_new = theta + R^{-1} x (p - x' theta)
        try:
            R_inv = np.linalg.inv(self.R)
            innov = p_t - x @ self.theta
            self.theta = self.theta + R_inv @ x * innov
        except np.linalg.LinAlgError:
            pass

    def get_params(self):
        return dict(a_hat=float(self.theta[0]), b_hat=float(self.theta[1]))

    def greedy_action(self, q_ref, p_ref):
        a_hat, b_hat = self.theta
        b_hat_safe = max(b_hat, 0.05)
        try:
            lq = solve_oracle_lq(
                a=a_hat,
                b=b_hat_safe,
                c=self.c,
                phi=self.phi,
                gamma=self.gamma,
                max_iter=200,
                tol=1e-8,
            )
            return lq["K0"] + lq["Kq"] * q_ref
        except (ValueError, ZeroDivisionError):
            return a_hat / (2 * (b_hat_safe + 0.5 * self.c))


# ---------------------------------------------------------------------------
# Tabular Q-learning (Watkins 1989)
# ---------------------------------------------------------------------------


class QLearningPolicy(Paradigm):
    """Discrete-state-action Q-learning with epsilon-greedy. Uses bucketed
    (q_prev, p_prev) state and bucketed action."""

    name = "Q-Learning"

    def __init__(
        self,
        gamma,
        g_s,
        g_a,
        p_min,
        p_max,
        alpha,
        eps_hi,
        eps_lo,
        q_min=0.0,
        q_max=4.0,
        T_episode=500,
    ):
        self.gamma = gamma
        self.g_s, self.g_a = g_s, g_a
        self.p_min, self.p_max = p_min, p_max
        self.alpha = alpha
        self.eps_hi, self.eps_lo = eps_hi, eps_lo
        self.q_min, self.q_max = q_min, q_max
        self.T = T_episode
        self.Q = None
        self.actions = None
        self.rng = None

    def reset(self, regime_params, seed=0):
        self.Q = np.zeros((self.g_s, self.g_s, self.g_a), dtype=np.float64)
        self.actions = np.linspace(self.q_min, self.q_max, self.g_a)
        self.rng = np.random.default_rng(seed + 12345)

    def _bucket(self, state):
        q_prev = np.clip(state[0], self.q_min, self.q_max)
        p_prev = np.clip(state[1], self.p_min, self.p_max)
        i = int((q_prev - self.q_min) / (self.q_max - self.q_min) * (self.g_s - 1))
        j = int((p_prev - self.p_min) / (self.p_max - self.p_min) * (self.g_s - 1))
        return min(max(i, 0), self.g_s - 1), min(max(j, 0), self.g_s - 1)

    def _action_idx(self, q):
        i = int((q - self.q_min) / (self.q_max - self.q_min) * (self.g_a - 1))
        return min(max(i, 0), self.g_a - 1)

    def _eps(self, t):
        frac = t / max(1, self.T)
        return self.eps_hi + (self.eps_lo - self.eps_hi) * min(1.0, frac)

    def act(self, state, t):
        i, j = self._bucket(state)
        eps = self._eps(t)
        if self.rng.random() < eps:
            a_idx = self.rng.integers(0, self.g_a)
        else:
            a_idx = int(np.argmax(self.Q[i, j]))
        self._last_idx = (i, j, a_idx)
        return float(self.actions[a_idx])

    def observe(self, state, action, reward, next_state):
        i, j, a_idx = self._last_idx
        ni, nj = self._bucket(next_state)
        td = reward + self.gamma * self.Q[ni, nj].max() - self.Q[i, j, a_idx]
        self.Q[i, j, a_idx] = self.Q[i, j, a_idx] + self.alpha * td

    def greedy_action(self, q_ref, p_ref):
        if self.Q is None:
            return np.nan
        state = np.array([q_ref, p_ref])
        i, j = self._bucket(state)
        a_idx = int(np.argmax(self.Q[i, j]))
        return float(self.actions[a_idx])


# ---------------------------------------------------------------------------
# Arifovic 1994 GA with election operator
# ---------------------------------------------------------------------------


class ArifovicGAPolicy(Paradigm):
    """Population of binary-encoded production rules. One chromosome plays
    per period in round-robin order. After GEN_LEN periods, evolve the
    population: elitism (top-2 by realized fitness survive), then
    fitness-proportional selection, single-point crossover, and bit-flip
    mutation. The original Arifovic 1994 election operator required the true
    demand and cost parameters to score hypothetical offspring; it is
    omitted here so the agent uses only realized observed profit. Expect a
    larger regret as the price of that honesty.
    """

    name = "Arifovic GA"

    def __init__(self, n_pop, L_bits, p_cross, p_mut, gen_len, q_min=0.0, q_max=4.0):
        self.n_pop, self.L = n_pop, L_bits
        self.p_cross, self.p_mut = p_cross, p_mut
        self.gen_len = gen_len
        self.q_min, self.q_max = q_min, q_max
        self.pop = None  # (N, L) binary
        self.fitness = None  # (N,) running mean of realized rewards
        self.n_obs = None  # (N,) plays since last evolve
        self.rng = None
        self.regime_params = None
        self.idx_cycle = 0

    def _decode(self, chrom):
        n_bits = self.L
        bits = chrom.astype(int)
        weights = 2 ** np.arange(n_bits - 1, -1, -1)
        val = (bits * weights).sum()
        return self.q_min + (val / (2**n_bits - 1)) * (self.q_max - self.q_min)

    def reset(self, regime_params, seed=0):
        self.rng = np.random.default_rng(seed + 54321)
        self.pop = self.rng.integers(0, 2, size=(self.n_pop, self.L))
        self.fitness = np.zeros(self.n_pop)
        self.n_obs = np.zeros(self.n_pop, dtype=int)
        self.regime_params = regime_params
        self.idx_cycle = 0
        self._last_active = 0

    def act(self, state, t):
        self._last_active = self.idx_cycle % self.n_pop
        self.idx_cycle += 1
        return float(self._decode(self.pop[self._last_active]))

    def observe(self, state, action, reward, next_state):
        i = self._last_active
        self.fitness[i] = (self.fitness[i] * self.n_obs[i] + reward) / (
            self.n_obs[i] + 1
        )
        self.n_obs[i] += 1
        if self.idx_cycle % self.gen_len == 0:
            self._evolve()

    def _evolve(self):
        # Selection probabilities from realized fitness (no parametric eval).
        fit = self.fitness.copy()
        fit_shift = fit - fit.min() + 1e-6
        probs = fit_shift / fit_shift.sum()
        # Elitism: top-2 chromosomes by realized fitness survive intact.
        n_elite = min(2, self.n_pop)
        elite_idx = np.argsort(-fit)[:n_elite]
        new_pop = [self.pop[i].copy() for i in elite_idx]
        elite_fitness = [self.fitness[i] for i in elite_idx]
        elite_nobs = [self.n_obs[i] for i in elite_idx]
        # Remaining slots: crossover + mutation of selected parents.
        n_offspring = self.n_pop - n_elite
        for _ in range((n_offspring + 1) // 2):
            parents_idx = self.rng.choice(self.n_pop, size=2, replace=True, p=probs)
            p1, p2 = self.pop[parents_idx[0]].copy(), self.pop[parents_idx[1]].copy()
            if self.rng.random() < self.p_cross:
                pt = self.rng.integers(1, self.L)
                p1[pt:], p2[pt:] = p2[pt:].copy(), p1[pt:].copy()
            mask1 = self.rng.random(self.L) < self.p_mut
            mask2 = self.rng.random(self.L) < self.p_mut
            p1 = np.where(mask1, 1 - p1, p1)
            p2 = np.where(mask2, 1 - p2, p2)
            new_pop.append(p1)
            new_pop.append(p2)
        new_pop = np.array(new_pop[: self.n_pop])
        self.pop = new_pop
        # Preserve elite fitness so they remain competitive next cycle;
        # zero out the offspring slots (they need to play to earn fitness).
        new_fitness = np.zeros(self.n_pop)
        new_nobs = np.zeros(self.n_pop, dtype=int)
        for j, (f, n) in enumerate(zip(elite_fitness, elite_nobs)):
            new_fitness[j] = f
            new_nobs[j] = n
        self.fitness = new_fitness
        self.n_obs = new_nobs

    def greedy_action(self, q_ref, p_ref):
        if self.pop is None:
            return np.nan
        if self.n_obs.sum() == 0:
            decoded = np.array([self._decode(c) for c in self.pop])
            return float(decoded.mean())
        i_best = int(np.argmax(self.fitness))
        return float(self._decode(self.pop[i_best]))


# ---------------------------------------------------------------------------
# Parametric LQ learner (closed-form Riccati on point estimates)
# ---------------------------------------------------------------------------


class ParametricLQLearner(Paradigm):
    """Learn (a, b, c, phi) by least squares on accumulated data, plan via
    the LQ-Bellman with current point estimates, act with Gaussian
    exploration noise. This is a model-based learner that exploits the
    linear-Gaussian structure to skip branched rollouts entirely; the
    MB-LG-REINFORCE baseline (with ensemble rollouts and REINFORCE) lives
    below.
    """

    name = "Model-Based LQ"

    def __init__(self, gamma, explore_std, warmup, q_min=0.0, q_max=4.0):
        self.gamma = gamma
        self.explore_std = explore_std
        self.warmup = warmup
        self.q_min, self.q_max = q_min, q_max
        self.data_q = None
        self.data_qprev = None
        self.data_p = None
        self.data_r = None
        self.rng = None
        self.a_hat = 1.0
        self.b_hat = 1.0
        self.c_hat = 1.0
        self.phi_hat = 0.2

    def reset(self, regime_params, seed=0):
        self.data_q = []
        self.data_qprev = []
        self.data_p = []
        self.data_r = []
        self.rng = np.random.default_rng(seed + 98765)
        self.a_hat = 1.0
        self.b_hat = 1.0
        self.c_hat = 1.0
        self.phi_hat = 0.2

    def _refit(self):
        if len(self.data_q) < 4:
            return
        q = np.array(self.data_q)
        qprev = np.array(self.data_qprev)
        p = np.array(self.data_p)
        r = np.array(self.data_r)
        # Fit p = a - b q via OLS on (1, q) -> p
        X = np.column_stack([np.ones_like(q), q])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        a_h, neg_b = coef
        self.a_hat = float(a_h)
        self.b_hat = max(0.05, float(-neg_b))
        # Fit r = p q - (c/2) q^2 - (phi/2) (q - qprev)^2 by regressing
        # (r - p q) on (q^2 / 2, (q - qprev)^2 / 2) with negative coefficients.
        # I.e. r - p q = -(c/2) q^2 - (phi/2) (q - qprev)^2
        resid = r - p * q
        feats = np.column_stack([-0.5 * q**2, -0.5 * (q - qprev) ** 2])
        coef2, *_ = np.linalg.lstsq(feats, resid, rcond=None)
        self.c_hat = max(0.05, float(coef2[0]))
        self.phi_hat = max(0.0, float(coef2[1]))

    def act(self, state, t):
        q_prev = state[0]
        if t < self.warmup:
            return float(self.rng.uniform(self.q_min, self.q_max))
        try:
            lq = solve_oracle_lq(
                a=self.a_hat,
                b=self.b_hat,
                c=self.c_hat,
                phi=self.phi_hat,
                gamma=self.gamma,
                max_iter=200,
                tol=1e-8,
            )
            q = lq["K0"] + lq["Kq"] * q_prev
        except (ValueError, ZeroDivisionError):
            q = self.a_hat / (2 * (self.b_hat + 0.5 * self.c_hat))
        # Add exploration noise that decays over the episode.
        decay = max(0.1, 1.0 - t / 250.0)
        q = q + self.rng.normal(0, self.explore_std * decay)
        return float(np.clip(q, self.q_min, self.q_max))

    def observe(self, state, action, reward, next_state):
        q_prev = state[0]
        q_t = action
        p_t = next_state[1]
        self.data_qprev.append(q_prev)
        self.data_q.append(q_t)
        self.data_p.append(p_t)
        self.data_r.append(reward)
        if len(self.data_q) % 1 == 0:
            self._refit()

    def get_params(self):
        return dict(
            a_hat=self.a_hat, b_hat=self.b_hat, c_hat=self.c_hat, phi_hat=self.phi_hat
        )

    def greedy_action(self, q_ref, p_ref):
        try:
            lq = solve_oracle_lq(
                a=self.a_hat,
                b=self.b_hat,
                c=self.c_hat,
                phi=self.phi_hat,
                gamma=self.gamma,
                max_iter=200,
                tol=1e-8,
            )
            return lq["K0"] + lq["Kq"] * q_ref
        except (ValueError, ZeroDivisionError):
            return self.a_hat / (2 * (self.b_hat + 0.5 * self.c_hat))


# ---------------------------------------------------------------------------
# MB-LG-REINFORCE: simplified MBPO variant (linear-Gaussian ensemble dynamics
# + linear policy + branched rollouts + REINFORCE). Drops the SAC actor-critic
# and dropout-based disagreement weighting of Janner et al. 2019's MBPO and
# keeps the branched-rollout outer loop with a one-parameter-pair Gaussian
# policy. Class name MBPOPolicy retained for cache/test compatibility.
# ---------------------------------------------------------------------------


class MBPOPolicy(Paradigm):
    """Simplified MBPO variant: linear-Gaussian ensemble dynamics + REINFORCE.

    Maintains an ensemble of linear-Gaussian demand models and a learned
    reward model, both fit from the replay buffer (no true parameters). At
    each step samples n_rollouts trajectories of length rollout_horizon from
    buffer-uniform initial states under a random ensemble member and the
    current linear policy q = K0 + Kq * q_prev plus Gaussian exploration.
    Accumulates discounted rollout returns and updates (K0, Kq) by REINFORCE
    with a moving-average baseline. Acts under the same stochastic policy
    with decaying exploration noise.

    Distinct from Janner et al. 2019's MBPO in three ways: (i) policy is a
    two-parameter Gaussian rather than a SAC actor, (ii) dynamics are
    linear-Gaussian bootstrap-ensembled rather than dropout-disagreement
    neural networks, (iii) no entropy regularization on the policy. The
    branched-rollout outer loop is preserved. Labelled "MB-LG-REINFORCE"
    (model-based linear-Gaussian REINFORCE) in figures and tables.
    """

    name = "MB-LG-REINFORCE"

    def __init__(
        self,
        gamma,
        explore_std,
        warmup,
        ensemble_size=5,
        rollout_horizon=5,
        n_rollouts=10,
        policy_lr=0.005,
        q_min=0.0,
        q_max=4.0,
    ):
        self.gamma = gamma
        self.explore_std = explore_std
        self.warmup = warmup
        self.ensemble_size = ensemble_size
        self.rollout_horizon = rollout_horizon
        self.n_rollouts = n_rollouts
        self.policy_lr = policy_lr
        self.q_min, self.q_max = q_min, q_max
        self.K0 = 0.0
        self.Kq = 0.0
        self.ensemble = []
        self.c_hat = 1.0
        self.phi_hat = 0.2
        self.buffer = []
        self.rng = None
        self.baseline = 0.0
        self.baseline_alpha = 0.05

    def reset(self, regime_params, seed=0):
        self.rng = np.random.default_rng(seed + 13579)
        self.K0 = 0.0
        self.Kq = 0.0
        self.ensemble = []
        self.c_hat = 1.0
        self.phi_hat = 0.2
        self.buffer = []
        self.baseline = 0.0

    def _fit_ensemble(self):
        if len(self.buffer) < 4:
            return
        q = np.array([x[1] for x in self.buffer])
        qprev = np.array([x[0] for x in self.buffer])
        p = np.array([x[2] for x in self.buffer])
        r = np.array([x[3] for x in self.buffer])
        n = len(q)
        self.ensemble = []
        for _ in range(self.ensemble_size):
            idx = self.rng.integers(0, n, size=n)
            X = np.column_stack([np.ones(n), q[idx]])
            coef, *_ = np.linalg.lstsq(X, p[idx], rcond=None)
            a_h, neg_b = coef
            b_h = max(0.05, float(-neg_b))
            pred = a_h - b_h * q[idx]
            sigma = float(np.std(p[idx] - pred, ddof=1)) if n > 2 else 0.1
            sigma = max(0.01, sigma)
            self.ensemble.append({"a_hat": float(a_h), "b_hat": b_h, "sigma": sigma})
        resid = r - p * q
        feats = np.column_stack([-0.5 * q**2, -0.5 * (q - qprev) ** 2])
        coef2, *_ = np.linalg.lstsq(feats, resid, rcond=None)
        self.c_hat = max(0.05, float(coef2[0]))
        self.phi_hat = max(0.0, float(coef2[1]))

    def _rollout(self, q_prev_init):
        """One H-step branched rollout under (random ensemble member, policy).

        Returns: (grad_K0, grad_Kq, total_return) where the gradient
        components are the REINFORCE score-function summed over the rollout.
        """
        member = self.ensemble[self.rng.integers(0, len(self.ensemble))]
        q_prev = float(q_prev_init)
        gamma_t = 1.0
        total_return = 0.0
        grad_K0 = 0.0
        grad_Kq = 0.0
        for _ in range(self.rollout_horizon):
            mean = self.K0 + self.Kq * q_prev
            eps_a = self.rng.normal(0, self.explore_std)
            a_unclipped = mean + eps_a
            a = float(np.clip(a_unclipped, self.q_min, self.q_max))
            eps_p = self.rng.normal(0, member["sigma"])
            p = member["a_hat"] - member["b_hat"] * a + eps_p
            r = p * a - 0.5 * self.c_hat * a**2 - 0.5 * self.phi_hat * (a - q_prev) ** 2
            total_return += gamma_t * r
            # Score function ∇log π(a|s) for Gaussian policy with mean=K0+Kq*q_prev
            score = (a_unclipped - mean) / (self.explore_std**2)
            grad_K0 += score
            grad_Kq += score * q_prev
            q_prev = a
            gamma_t *= self.gamma
        return grad_K0, grad_Kq, total_return

    def _update_policy(self):
        if len(self.buffer) < 1 or len(self.ensemble) == 0:
            return
        returns = np.zeros(self.n_rollouts)
        grads_K0 = np.zeros(self.n_rollouts)
        grads_Kq = np.zeros(self.n_rollouts)
        for k in range(self.n_rollouts):
            init_idx = self.rng.integers(0, len(self.buffer))
            q_prev_init = self.buffer[init_idx][0]
            g0, gq, ret = self._rollout(q_prev_init)
            grads_K0[k] = g0
            grads_Kq[k] = gq
            returns[k] = ret
        baseline_old = self.baseline
        self.baseline = (
            1 - self.baseline_alpha
        ) * self.baseline + self.baseline_alpha * float(returns.mean())
        advantages = returns - baseline_old
        self.K0 += self.policy_lr * float(np.mean(grads_K0 * advantages))
        self.Kq += self.policy_lr * float(np.mean(grads_Kq * advantages))

    def act(self, state, t):
        q_prev = state[0]
        if t < self.warmup:
            return float(self.rng.uniform(self.q_min, self.q_max))
        mean = self.K0 + self.Kq * q_prev
        decay = max(0.1, 1.0 - t / 250.0)
        q = mean + self.rng.normal(0, self.explore_std * decay)
        return float(np.clip(q, self.q_min, self.q_max))

    def observe(self, state, action, reward, next_state):
        q_prev = state[0]
        q_t = action
        p_t = next_state[1]
        self.buffer.append((q_prev, q_t, p_t, reward))
        self._fit_ensemble()
        if len(self.ensemble) > 0:
            self._update_policy()

    def get_params(self):
        if not self.ensemble:
            return dict(a_hat=1.0, b_hat=1.0, c_hat=self.c_hat, phi_hat=self.phi_hat)
        a_mean = float(np.mean([m["a_hat"] for m in self.ensemble]))
        b_mean = float(np.mean([m["b_hat"] for m in self.ensemble]))
        return dict(a_hat=a_mean, b_hat=b_mean, c_hat=self.c_hat, phi_hat=self.phi_hat)

    def greedy_action(self, q_ref, p_ref):
        return float(self.K0 + self.Kq * q_ref)


# ---------------------------------------------------------------------------
# MB-LG-Pathwise: same ensemble model learning as MBPOPolicy, but plans by
# analytic (pathwise) gradients instead of REINFORCE score functions.
# The forward sensitivity recursion is exact for the deterministic skeleton
# (noise eps_p enters only as zero-mean additive noise to each reward step;
# dropping it from the gradient is unbiased and eliminates score-function
# variance). Gradient clipping is applied with max-norm 1.0 for stability.
# policy_lr = 0.05 and n_rollouts = 20 (vs 0.005 and 10 for REINFORCE); the
# larger step is safe because the pathwise gradient is low-variance. All other
# hypers match REINFORCE.
# ---------------------------------------------------------------------------


class MBPathwisePolicy(Paradigm):
    """Model-based learner with pathwise (analytic) policy gradients.

    Identical to MBPOPolicy in all respects except the gradient estimator:
    - Model learning: bootstrap linear-Gaussian ensemble fit of (a_hat, b_hat,
      sigma) plus OLS regression for (c_hat, phi_hat) from the replay buffer.
      Uses ONLY learned parameters. Does NOT read true regime params.
      Does NOT call solve_oracle_lq.
    - Policy class: linear q_t = K0 + Kq * q_{t-1}, identical to MBPOPolicy.
    - Planning: forward sensitivity recursion differentiates the expected
      discounted model-rollout return analytically. For a rollout starting at
      q_0 (sampled from buffer) with deterministic lag q_{t+1} = a_t:

        Expected reward:
          r_t = a_hat * a_t - b_hat * a_t^2 - (c_hat/2)*a_t^2
                - (phi_hat/2)*(a_t - q_t)^2
        where a_t = K0 + Kq * q_t  (mean action; noise eps_a only adds
        zero-mean terms to dr/da, so E[grad] equals the deterministic gradient).

        Forward sensitivity (theta in {K0, Kq}):
          dq_0/dtheta = 0  (buffer initial state is fixed)
          da_t/dK0 = 1    + Kq * dq_t/dK0
          da_t/dKq = q_t  + Kq * dq_t/dKq
          dq_{t+1}/dtheta = da_t/dtheta  (because q_{t+1} = a_t)
          dr_t/da_t = a_hat - 2*b_hat*a_t - c_hat*a_t - phi_hat*(a_t - q_t)
          dr_t/dq_t = phi_hat*(a_t - q_t)
          dJ/dtheta = sum_t gamma^t [ dr_t/da_t * da_t/dtheta
                                      + dr_t/dq_t * dq_t/dtheta ]

      Averaged over ensemble members and buffer-sampled initial states, then
      gradient-ascent: K0 += lr * dJ/dK0, Kq += lr * dJ/dKq.
      Gradient clipped to max L2-norm 1.0 before the update.

    Labeled 'MB-LG-Pathwise' in figures and tables.
    """

    name = "MB-LG-Pathwise"

    def __init__(
        self,
        gamma,
        explore_std,
        warmup,
        ensemble_size=5,
        rollout_horizon=5,
        n_rollouts=20,
        policy_lr=0.05,
        q_min=0.0,
        q_max=4.0,
    ):
        self.gamma = gamma
        self.explore_std = explore_std
        self.warmup = warmup
        self.ensemble_size = ensemble_size
        self.rollout_horizon = rollout_horizon
        self.n_rollouts = n_rollouts
        self.policy_lr = policy_lr
        self.q_min, self.q_max = q_min, q_max
        self.K0 = 0.0
        self.Kq = 0.0
        self.ensemble = []
        self.c_hat = 1.0
        self.phi_hat = 0.2
        self.buffer = []
        self.rng = None

    def reset(self, regime_params, seed=0):
        self.rng = np.random.default_rng(seed + 24680)
        self.K0 = 0.0
        self.Kq = 0.0
        self.ensemble = []
        self.c_hat = 1.0
        self.phi_hat = 0.2
        self.buffer = []

    def _fit_ensemble(self):
        # Identical to MBPOPolicy._fit_ensemble: bootstrap OLS on replay buffer.
        if len(self.buffer) < 4:
            return
        q = np.array([x[1] for x in self.buffer])
        qprev = np.array([x[0] for x in self.buffer])
        p = np.array([x[2] for x in self.buffer])
        r = np.array([x[3] for x in self.buffer])
        n = len(q)
        self.ensemble = []
        for _ in range(self.ensemble_size):
            idx = self.rng.integers(0, n, size=n)
            X = np.column_stack([np.ones(n), q[idx]])
            coef, *_ = np.linalg.lstsq(X, p[idx], rcond=None)
            a_h, neg_b = coef
            b_h = max(0.05, float(-neg_b))
            pred = a_h - b_h * q[idx]
            sigma = float(np.std(p[idx] - pred, ddof=1)) if n > 2 else 0.1
            sigma = max(0.01, sigma)
            self.ensemble.append({"a_hat": float(a_h), "b_hat": b_h, "sigma": sigma})
        resid = r - p * q
        feats = np.column_stack([-0.5 * q**2, -0.5 * (q - qprev) ** 2])
        coef2, *_ = np.linalg.lstsq(feats, resid, rcond=None)
        self.c_hat = max(0.05, float(coef2[0]))
        self.phi_hat = max(0.0, float(coef2[1]))

    def _pathwise_rollout(self, q_prev_init, member):
        """One H-step pathwise gradient rollout under a single ensemble member.

        Uses the deterministic skeleton (no eps_p noise in gradient).
        Returns (dJ_dK0, dJ_dKq, total_return).
        """
        a_hat = member["a_hat"]
        b_hat = member["b_hat"]
        c_hat = self.c_hat
        phi_hat = self.phi_hat

        q_t = float(q_prev_init)
        # Forward sensitivity: dq_t/dtheta, initialised at 0 (q_0 is fixed).
        dq_dK0 = 0.0
        dq_dKq = 0.0

        gamma_t = 1.0
        total_return = 0.0
        dJ_dK0 = 0.0
        dJ_dKq = 0.0

        for _ in range(self.rollout_horizon):
            # Mean action under current policy (deterministic skeleton).
            a_t = self.K0 + self.Kq * q_t

            # Sensitivity of a_t w.r.t. theta.
            da_dK0 = 1.0 + self.Kq * dq_dK0
            da_dKq = q_t + self.Kq * dq_dKq

            # Expected reward (E[eps_p] = 0 so noise drops out).
            # r_t = a_hat*a_t - b_hat*a_t^2 - (c_hat/2)*a_t^2
            #       - (phi_hat/2)*(a_t - q_t)^2
            r_t = (
                a_hat * a_t
                - b_hat * a_t**2
                - 0.5 * c_hat * a_t**2
                - 0.5 * phi_hat * (a_t - q_t) ** 2
            )
            total_return += gamma_t * r_t

            # Partial derivatives of r_t.
            dr_da = a_hat - 2.0 * b_hat * a_t - c_hat * a_t - phi_hat * (a_t - q_t)
            dr_dq = phi_hat * (a_t - q_t)

            dJ_dK0 += gamma_t * (dr_da * da_dK0 + dr_dq * dq_dK0)
            dJ_dKq += gamma_t * (dr_da * da_dKq + dr_dq * dq_dKq)

            # Advance state: q_{t+1} = a_t (deterministic lag).
            q_t = a_t
            dq_dK0 = da_dK0
            dq_dKq = da_dKq
            gamma_t *= self.gamma

        return dJ_dK0, dJ_dKq, total_return

    def _update_policy(self):
        if len(self.buffer) < 1 or len(self.ensemble) == 0:
            return
        grad_K0_total = 0.0
        grad_Kq_total = 0.0
        for k in range(self.n_rollouts):
            # Sample initial state from buffer (same as MBPOPolicy).
            init_idx = self.rng.integers(0, len(self.buffer))
            q_prev_init = self.buffer[init_idx][0]
            # Sample a random ensemble member.
            member = self.ensemble[self.rng.integers(0, len(self.ensemble))]
            g0, gq, _ = self._pathwise_rollout(q_prev_init, member)
            grad_K0_total += g0
            grad_Kq_total += gq
        grad_K0 = grad_K0_total / self.n_rollouts
        grad_Kq = grad_Kq_total / self.n_rollouts
        # Gradient clipping (L2 norm, max 1.0) for numerical stability.
        grad_norm = np.sqrt(grad_K0**2 + grad_Kq**2) + 1e-12
        if grad_norm > 1.0:
            grad_K0 /= grad_norm
            grad_Kq /= grad_norm
        self.K0 += self.policy_lr * grad_K0
        self.Kq += self.policy_lr * grad_Kq

    def act(self, state, t):
        # Identical to MBPOPolicy.act: mean action + decaying exploration noise.
        q_prev = state[0]
        if t < self.warmup:
            return float(self.rng.uniform(self.q_min, self.q_max))
        mean = self.K0 + self.Kq * q_prev
        decay = max(0.1, 1.0 - t / 250.0)
        q = mean + self.rng.normal(0, self.explore_std * decay)
        return float(np.clip(q, self.q_min, self.q_max))

    def observe(self, state, action, reward, next_state):
        q_prev = state[0]
        q_t = action
        p_t = next_state[1]
        self.buffer.append((q_prev, q_t, p_t, reward))
        self._fit_ensemble()
        if len(self.ensemble) > 0:
            self._update_policy()

    def get_params(self):
        if not self.ensemble:
            return dict(a_hat=1.0, b_hat=1.0, c_hat=self.c_hat, phi_hat=self.phi_hat)
        a_mean = float(np.mean([m["a_hat"] for m in self.ensemble]))
        b_mean = float(np.mean([m["b_hat"] for m in self.ensemble]))
        return dict(a_hat=a_mean, b_hat=b_mean, c_hat=self.c_hat, phi_hat=self.phi_hat)

    def greedy_action(self, q_ref, p_ref):
        return float(self.K0 + self.Kq * q_ref)


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------


def _regime_ref_state(rp):
    """Reference (q_ref, p_ref) for policy distance: static optimum on demand."""
    q_ref = rp["a"] / (2.0 * (rp["b"] + 0.5 * rp["c"]))
    p_ref = rp["a"] - rp["b"] * q_ref
    return q_ref, p_ref


def rollout(paradigm, regime_params, T, gamma, seed):
    """One episode against the environment.

    Returns a dict with:
      rewards:        (T,) array of per-step rewards
      params_history: length-T list of dicts with the paradigm's current
                      parameter estimates (empty for paradigms without an
                      explicit demand model)
      greedy_actions: (T,) array of the paradigm's noise-free action at the
                      regime's reference state, evaluated AFTER each step's
                      observe()
    """
    env = CobwebEnv(
        a=regime_params["a"],
        b=regime_params["b"],
        c=regime_params["c"],
        phi=regime_params["phi"],
        sigma=regime_params["sigma"],
        gamma=gamma,
        T=T,
        seed=seed,
    )
    paradigm.reset(regime_params, seed=seed)
    state = env.reset()
    q_ref, p_ref = _regime_ref_state(regime_params)
    rewards = np.zeros(T)
    params_history = []
    greedy_actions = np.full(T, np.nan)
    for t in range(T):
        action = paradigm.act(state, t)
        next_state, reward, done, _ = env.step(action)
        paradigm.observe(state, action, reward, next_state)
        rewards[t] = reward
        params_history.append(paradigm.get_params())
        ga = paradigm.greedy_action(q_ref, p_ref)
        greedy_actions[t] = ga if ga is not None else np.nan
        state = next_state
        if done:
            break
    return dict(
        rewards=rewards, params_history=params_history, greedy_actions=greedy_actions
    )


def cumulative_regret(oracle_rewards, paradigm_rewards):
    """Per-step regret = oracle - paradigm; cumulative sum."""
    diff = oracle_rewards - paradigm_rewards
    return np.cumsum(diff)


# ---------------------------------------------------------------------------
# Per-paradigm compute functions
# ---------------------------------------------------------------------------


def make_paradigm(name, config):
    """Factory."""
    if name == "Oracle":
        return OraclePolicy(gamma=config["GAMMA"])
    if name == "Naive":
        return NaivePolicy()
    if name == "RLS":
        return RLSPolicy(gamma=config["GAMMA"], init_var=config["INIT_VAR"])
    if name == "Q-Learning":
        return QLearningPolicy(
            gamma=config["GAMMA"],
            g_s=config["G_S"],
            g_a=config["G_A"],
            p_min=config["P_MIN"],
            p_max=config["P_MAX"],
            alpha=config["ALPHA"],
            eps_hi=config["EPS_HI"],
            eps_lo=config["EPS_LO"],
            q_min=config["Q_MIN"],
            q_max=config["Q_MAX"],
            T_episode=config["T_EPISODE"],
        )
    if name == "Arifovic GA":
        return ArifovicGAPolicy(
            n_pop=config["N_POP"],
            L_bits=config["L_BITS"],
            p_cross=config["P_CROSS"],
            p_mut=config["P_MUT"],
            gen_len=config["GEN_LEN"],
            q_min=config["Q_MIN"],
            q_max=config["Q_MAX"],
        )
    if name == "Model-Based LQ":
        return ParametricLQLearner(
            gamma=config["GAMMA"],
            explore_std=config["EXPLORE_STD"],
            warmup=config["WARMUP"],
            q_min=config["Q_MIN"],
            q_max=config["Q_MAX"],
        )
    if name == "MB-LG-REINFORCE":
        return MBPOPolicy(
            gamma=config["GAMMA"],
            explore_std=config["EXPLORE_STD"],
            warmup=config["WARMUP"],
            ensemble_size=config["ENSEMBLE_SIZE"],
            rollout_horizon=config["ROLLOUT_HORIZON"],
            n_rollouts=config["N_ROLLOUTS"],
            policy_lr=config["POLICY_LR"],
            q_min=config["Q_MIN"],
            q_max=config["Q_MAX"],
        )
    if name == "MB-LG-Pathwise":
        return MBPathwisePolicy(
            gamma=config["GAMMA"],
            explore_std=config["EXPLORE_STD"],
            warmup=config["WARMUP"],
            ensemble_size=config["ENSEMBLE_SIZE"],
            rollout_horizon=config["ROLLOUT_HORIZON"],
            n_rollouts=config["N_ROLLOUTS"],
            policy_lr=config["POLICY_LR"],
            q_min=config["Q_MIN"],
            q_max=config["Q_MAX"],
        )
    raise ValueError(name)


def compute_shared(config):
    """For each regime: solve oracle, run oracle rollout per seed, store rewards
    and the oracle's reference-state greedy action trajectory.
    """
    out = {}
    N = config["N_SEEDS"]
    T = config["T_EPISODE"]
    for regime_name, rp in config["REGIMES"].items():
        oracle_rewards_all = np.zeros((N, T))
        oracle_greedy_all = np.zeros((N, T))
        q_ref, p_ref = _regime_ref_state(rp)
        for s in range(N):
            oracle = OraclePolicy(gamma=config["GAMMA"])
            res = rollout(oracle, rp, T, config["GAMMA"], seed=s)
            oracle_rewards_all[s] = res["rewards"]
            oracle_greedy_all[s] = res["greedy_actions"]
        out[regime_name] = dict(
            params=rp,
            oracle_rewards=oracle_rewards_all,
            oracle_greedy_actions=oracle_greedy_all,
            q_ref=q_ref,
            p_ref=p_ref,
        )
    return out


def compute_paradigm(config, shared, paradigm_name):
    """Run paradigm across (regimes, seeds). Returns dict per regime with
    regret curves, parameter trajectories, and policy distance to oracle.
    """
    out = {}
    N = config["N_SEEDS"]
    T = config["T_EPISODE"]
    for regime_name, sh in shared.items():
        rp = sh["params"]
        oracle_rewards_all = sh["oracle_rewards"]
        oracle_greedy_all = sh["oracle_greedy_actions"]
        regret_curves = np.zeros((N, T))
        final_regret = np.zeros(N)
        greedy_curves = np.zeros((N, T))
        params_per_seed = []
        param_keys = set()
        for s in range(N):
            np.random.seed(s)
            paradigm = make_paradigm(paradigm_name, config)
            res = rollout(paradigm, rp, T, config["GAMMA"], seed=s)
            curve = cumulative_regret(oracle_rewards_all[s], res["rewards"])
            regret_curves[s] = curve
            final_regret[s] = curve[-1]
            greedy_curves[s] = res["greedy_actions"]
            params_per_seed.append(res["params_history"])
            for d in res["params_history"]:
                param_keys.update(d.keys())

        # Aggregate parameter trajectories per key.
        param_trajectories = {}
        for key in sorted(param_keys):
            traj = np.full((N, T), np.nan)
            for s in range(N):
                for t in range(T):
                    val = params_per_seed[s][t].get(key, np.nan)
                    if val is not None:
                        traj[s, t] = val
            with np.errstate(invalid="ignore"):
                param_trajectories[key] = dict(
                    mean=np.nanmean(traj, axis=0),
                    se=(np.nanstd(traj, axis=0, ddof=1) / np.sqrt(N)),
                    final_mean=float(np.nanmean(traj[:, -1])),
                    final_se=float(np.nanstd(traj[:, -1], ddof=1) / np.sqrt(N)),
                )

        # Policy distance: |paradigm_greedy - oracle_greedy| at reference state.
        with np.errstate(invalid="ignore"):
            pd = np.abs(greedy_curves - oracle_greedy_all)
            pd_mean = np.nanmean(pd, axis=0)
            pd_se = np.nanstd(pd, axis=0, ddof=1) / np.sqrt(N)

        out[regime_name] = dict(
            regret_curves=regret_curves,
            final_regret=final_regret,
            mean_curve=regret_curves.mean(axis=0),
            se_curve=regret_curves.std(axis=0, ddof=1) / np.sqrt(N),
            final_mean=final_regret.mean(),
            final_se=final_regret.std(ddof=1) / np.sqrt(N),
            param_trajectories=param_trajectories,
            policy_distance_mean=pd_mean,
            policy_distance_se=pd_se,
        )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PARADIGM_REGISTRY = {
    "Oracle": (compute_paradigm, ORACLE_CONFIG),
    "Naive": (compute_paradigm, NAIVE_CONFIG),
    "RLS": (compute_paradigm, RLS_CONFIG),
    "Q-Learning": (compute_paradigm, QL_CONFIG),
    "Arifovic GA": (compute_paradigm, GA_CONFIG),
    "Model-Based LQ": (compute_paradigm, MBLQ_CONFIG),
    "MB-LG-REINFORCE": (compute_paradigm, MB_LG_REINFORCE_CONFIG),
    "MB-LG-Pathwise": (compute_paradigm, MB_PATHWISE_CONFIG),
}


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        SHARED_CONFIG,
        compute_shared,
        SHARED_CONFIG,
        force=("shared" in force),
    )
    results = {}
    for name, (fn, cfg) in PARADIGM_REGISTRY.items():
        # Skip Oracle (it would have zero regret by construction); useful to
        # plot but no work to do. We compute it anyway for completeness.
        cache_key = name.replace(" ", "_")
        results[name] = compute_or_load(
            CACHE_DIR,
            SCRIPT_NAME,
            cache_key,
            cfg,
            fn,
            cfg,
            shared,
            name,
            force=(name in force or "shared" in force),
        )
    return dict(shared=shared, results=results)


# ---------------------------------------------------------------------------
# Outputs: figure + table + stdout
# ---------------------------------------------------------------------------

PARADIGM_ORDER = [
    "Oracle",
    "RLS",
    "Model-Based LQ",
    "MB-LG-Pathwise",
    "Arifovic GA",
    "Naive",
    "MB-LG-REINFORCE",
    "Q-Learning",
]
PARADIGM_COLORS = {
    "Oracle": COLORS["black"],
    "Naive": COLORS["gray"],
    "RLS": COLORS["red"],
    "Q-Learning": COLORS["blue"],
    "Arifovic GA": COLORS["green"],
    "Model-Based LQ": COLORS["purple"],
    "MB-LG-REINFORCE": COLORS["orange"],
    "MB-LG-Pathwise": COLORS["cyan"],
}
REGIME_ORDER = ["stable", "borderline", "unstable"]


def _plot_regret_curves(data):
    fig, axes = plt.subplots(1, 3, figsize=FIG_TRIPLE, sharey=False)
    T = SHARED_CONFIG["T_EPISODE"]
    t_axis = np.arange(1, T + 1)
    for ax, regime in zip(axes, REGIME_ORDER):
        for name in PARADIGM_ORDER:
            res = data["results"][name][regime]
            mean = res["mean_curve"]
            se = res["se_curve"]
            color = PARADIGM_COLORS[name]
            ax.plot(t_axis, mean, label=name, color=color, linewidth=1.6)
            ax.fill_between(t_axis, mean - se, mean + se, color=color, alpha=0.15)
        b_over_c = (
            data["shared"][regime]["params"]["b"]
            / data["shared"][regime]["params"]["c"]
        )
        ax.set_title(f"{regime} (b/c = {b_over_c:.1f})")
        ax.set_xlabel("environment step $t$")
        if regime == "stable":
            ax.set_ylabel("cumulative regret")
        ax.axhline(0, **BENCH_STYLE)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Cumulative regret across stability regimes (20 seeds, mean $\\pm$ SE)",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cobweb_paradigms.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


PARAM_RECOVERY_LEARNERS = ["RLS", "Model-Based LQ", "MB-LG-REINFORCE", "MB-LG-Pathwise"]


def _plot_param_recovery(data):
    """3 rows x 2 cols: a_hat and b_hat trajectories for the parametric
    learners (RLS, Model-Based LQ, MB-LG-REINFORCE, MB-LG-Pathwise) per regime."""
    T = SHARED_CONFIG["T_EPISODE"]
    t_axis = np.arange(1, T + 1)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for r_idx, regime in enumerate(REGIME_ORDER):
        rp = data["shared"][regime]["params"]
        ax_a = axes[r_idx, 0]
        ax_b = axes[r_idx, 1]
        for name in PARAM_RECOVERY_LEARNERS:
            traj = data["results"][name][regime]["param_trajectories"]
            if "a_hat" in traj:
                m, s = traj["a_hat"]["mean"], traj["a_hat"]["se"]
                ax_a.plot(
                    t_axis, m, label=name, color=PARADIGM_COLORS[name], linewidth=1.6
                )
                ax_a.fill_between(
                    t_axis, m - s, m + s, color=PARADIGM_COLORS[name], alpha=0.15
                )
            if "b_hat" in traj:
                m, s = traj["b_hat"]["mean"], traj["b_hat"]["se"]
                ax_b.plot(
                    t_axis, m, label=name, color=PARADIGM_COLORS[name], linewidth=1.6
                )
                ax_b.fill_between(
                    t_axis, m - s, m + s, color=PARADIGM_COLORS[name], alpha=0.15
                )
        ax_a.axhline(rp["a"], **BENCH_STYLE)
        ax_b.axhline(rp["b"], **BENCH_STYLE)
        ax_a.set_ylabel(f"{regime}\n$\\hat a_t$")
        ax_b.set_ylabel("$\\hat b_t$")
        if r_idx == 0:
            ax_a.legend(loc="upper right", fontsize=8)
            ax_a.set_title(f"$\\hat a_t$ (true a = {rp['a']:.1f})")
            ax_b.set_title("$\\hat b_t$ (true b varies)")
    axes[-1, 0].set_xlabel("environment step $t$")
    axes[-1, 1].set_xlabel("environment step $t$")
    fig.suptitle(
        "Parameter recovery for RLS, Model-Based LQ, MB-LG-REINFORCE, "
        "and MB-LG-Pathwise (20 seeds, mean $\\pm$ SE; "
        "dashed lines mark the true values)",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cobweb_paradigms_param_recovery.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


def _plot_policy_distance(data):
    """1 row x 3 cols: |q_paradigm(q_ref) - q_oracle(q_ref)| vs t, per regime."""
    T = SHARED_CONFIG["T_EPISODE"]
    t_axis = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=FIG_TRIPLE, sharey=False)
    learner_names = [n for n in PARADIGM_ORDER if n != "Oracle"]
    for ax, regime in zip(axes, REGIME_ORDER):
        for name in learner_names:
            res = data["results"][name][regime]
            mean = res["policy_distance_mean"]
            se = res["policy_distance_se"]
            color = PARADIGM_COLORS[name]
            ax.plot(t_axis, mean, label=name, color=color, linewidth=1.6)
            ax.fill_between(
                t_axis, np.maximum(mean - se, 1e-4), mean + se, color=color, alpha=0.15
            )
        b_over_c = (
            data["shared"][regime]["params"]["b"]
            / data["shared"][regime]["params"]["c"]
        )
        ax.set_title(f"{regime} (b/c = {b_over_c:.1f})")
        ax.set_xlabel("environment step $t$")
        ax.set_yscale("log")
        if regime == "stable":
            ax.set_ylabel(
                "$|q_{\\mathrm{paradigm}}(q_{\\mathrm{ref}}) - q_{\\mathrm{oracle}}(q_{\\mathrm{ref}})|$"
            )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Policy distance to oracle at the reference state "
        "(log scale, 20 seeds, mean $\\pm$ SE)",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cobweb_paradigms_policy_distance.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


def _write_table_regret(data):
    tbl_path = os.path.join(OUTPUT_DIR, "cobweb_paradigms_results.tex")
    with open(tbl_path, "w") as f:
        f.write("% Cumulative regret at T=500, mean ± SE across 20 seeds.\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Paradigm & Stable ($b/c=0.5$) & Borderline ($b/c=1.0$) & Unstable ($b/c=2.0$) \\\\\n"
        )
        f.write("\\midrule\n")
        for name in PARADIGM_ORDER:
            row = [name]
            for regime in REGIME_ORDER:
                r = data["results"][name][regime]
                row.append(f"{r['final_mean']:.2f} $\\pm$ {r['final_se']:.2f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Table saved: {tbl_path}")


def _write_table_recovery(data):
    """Final |Delta_a|, |Delta_b|, [|Delta_c|, |Delta_phi|] for parametric learners."""
    tbl_path = os.path.join(OUTPUT_DIR, "cobweb_paradigms_final_recovery.tex")
    with open(tbl_path, "w") as f:
        f.write(
            "% Final parameter recovery error |hat - true|, mean +/- SE over 20 seeds.\n"
        )
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Paradigm & Regime & $|\\hat a - a|$ & $|\\hat b - b|$ & $|\\hat c - c|$ & $|\\hat\\phi - \\phi|$ \\\\\n"
        )
        f.write("\\midrule\n")
        for name in PARAM_RECOVERY_LEARNERS:
            for regime in REGIME_ORDER:
                rp = data["shared"][regime]["params"]
                traj = data["results"][name][regime]["param_trajectories"]
                row = [name, regime]
                for key, true in [
                    ("a_hat", rp["a"]),
                    ("b_hat", rp["b"]),
                    ("c_hat", rp["c"]),
                    ("phi_hat", rp["phi"]),
                ]:
                    if key in traj:
                        err = abs(traj[key]["final_mean"] - true)
                        se = traj[key]["final_se"]
                        row.append(f"{err:.3f} $\\pm$ {se:.3f}")
                    else:
                        row.append("---")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Table saved: {tbl_path}")


def _print_summary(data):
    print("\n=== Cumulative regret at T=500 (mean ± SE, n=20 seeds) ===\n")
    header = f"{'Paradigm':<17}" + "".join(f"{r:>22}" for r in REGIME_ORDER)
    print(header)
    print("-" * len(header))
    for name in PARADIGM_ORDER:
        row = f"{name:<17}"
        for regime in REGIME_ORDER:
            r = data["results"][name][regime]
            row += f"  {r['final_mean']:>10.2f} ± {r['final_se']:>5.2f}    "
        print(row)

    print("\n=== Final |hat - true| parameter recovery (n=20 seeds) ===\n")
    for name in PARAM_RECOVERY_LEARNERS:
        print(f"\n  {name}:")
        for regime in REGIME_ORDER:
            rp = data["shared"][regime]["params"]
            traj = data["results"][name][regime]["param_trajectories"]
            line = f"    {regime:<12}"
            for key, true in [
                ("a_hat", rp["a"]),
                ("b_hat", rp["b"]),
                ("c_hat", rp["c"]),
                ("phi_hat", rp["phi"]),
            ]:
                if key in traj:
                    err = abs(traj[key]["final_mean"] - true)
                    line += f"  |{key[0:1]}|={err:.3f}"
            print(line)

    print(
        "\n=== Policy distance to oracle at reference state, final step (mean ± SE) ===\n"
    )
    learner_names = [n for n in PARADIGM_ORDER if n != "Oracle"]
    header = f"{'Paradigm':<17}" + "".join(f"{r:>18}" for r in REGIME_ORDER)
    print(header)
    print("-" * len(header))
    for name in learner_names:
        row = f"{name:<17}"
        for regime in REGIME_ORDER:
            r = data["results"][name][regime]
            row += f"  {r['policy_distance_mean'][-1]:>8.3f} ± {r['policy_distance_se'][-1]:>5.3f}  "
        print(row)


def generate_outputs(data):
    apply_style()
    _plot_regret_curves(data)
    _plot_param_recovery(data)
    _plot_policy_distance(data)
    _write_table_regret(data)
    _write_table_recovery(data)
    _print_summary(data)


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print(f"=== {SCRIPT_NAME} ===")
    print(f"Regimes: {list(REGIMES.keys())}")
    print(f"Paradigms: {PARADIGM_ORDER}")
    print(
        f"N_SEEDS = {SHARED_CONFIG['N_SEEDS']}, T_EPISODE = {SHARED_CONFIG['T_EPISODE']}\n"
    )

    if args.plots_only:
        data = compute_data(force=set())
        generate_outputs(data)
        return
    data = compute_data(force=force)
    if args.data_only:
        print("Data-only run complete.")
        return
    generate_outputs(data)


if __name__ == "__main__":
    main()
