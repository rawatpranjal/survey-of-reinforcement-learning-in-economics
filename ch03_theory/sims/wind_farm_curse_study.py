"""
Wind Farm Storage Control: Curse of Dimensionality Study
Chapter 3, Theory. Compares tabular DP (exponential in state dimension) against
three RL methods (factored Q-learning, DQN, linear actor-critic on polynomial
features) on a wind-farm battery storage MDP as the state dimension grows 3 -> 6.
Environment adapted from the wind-farm storage experiment in Lu et al. (2025),
"Overcoming the Curse of Dimensionality in RL Through Approximate Factorization".

Cached components: one per (method, dimension), 16 total (DP_d3..DP_d6,
factored_d3..d6, dqn_d3..d6, linear_ac_d3..d6). Force one with
--algo dqn_d5, or a whole method with --algo dqn.

NOTE: a cold-cache full compute takes ~2 hours, which exceeds the 3600 s
per-script timeout in scripts/run_all_sims.py. Run the first compute directly
(python3 ch03_theory/sims/wind_farm_curse_study.py); per-component caching means
interrupted or runner-truncated runs resume where they left off.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, BENCH_STYLE, FIG_SINGLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

import argparse
import time
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTDIR, "cache")
SCRIPT_NAME = "wind_farm_curse_study"

# =============================================================================
# Configuration
# =============================================================================
HORIZON = 24  # hours
GAMMA = 0.95

W_MIN, W_MAX = 0, 100  # wind power (kW)
P_MIN, P_MAX = 0, 1  # price ($/kWh)
C_MIN, C_MAX = 0, 50  # battery SoC (kWh)
A_MIN, A_MAX = -20, 20  # charge/discharge (kW)
EXTRA_MIN, EXTRA_MAX = 0, 1  # auxiliary dims (scaling study)

N_ACTIONS = 11  # action discretization shared by all methods

DIMS = [3, 4, 5, 6]

ENV_PARAMS = {
    "HORIZON": HORIZON,
    "GAMMA": GAMMA,
    "W": [W_MIN, W_MAX],
    "P": [P_MIN, P_MAX],
    "C": [C_MIN, C_MAX],
    "A": [A_MIN, A_MAX],
    "EXTRA": [EXTRA_MIN, EXTRA_MAX],
    "N_ACTIONS": N_ACTIONS,
}
EVAL_PARAMS = {"EVAL_SEED": 99, "EVAL_EPISODES": 50}

RL_COMMON = {**ENV_PARAMS, **EVAL_PARAMS, "N_SEEDS": 10, "N_EPISODES": 3000}

DP_CONFIG = {
    **ENV_PARAMS,
    **EVAL_PARAMS,
    "N_BINS": 7,
    "N_MC": 20,
    "TIMEOUT_MIN": 10,
    "DP_SEED": 42,
}
FACTORED_CONFIG = {
    **RL_COMMON,
    "N_BINS": 30,
    "ALPHA": 0.2,
    "EPSILON": 0.2,
    "REWARD_DIMS": 3,
}
DQN_CONFIG = {
    **RL_COMMON,
    "LR": 5e-4,
    "HIDDEN": 128,
    "BUFFER": 10000,
    "BATCH": 64,
    "TARGET_UPDATE": 100,
    "EPS_START": 0.3,
    "EPS_END": 0.01,
    "EPS_DECAY": 0.9995,
}
LINEAR_AC_CONFIG = {**RL_COMMON, "LR_ACTOR": 0.02, "LR_CRITIC": 0.08}

# Display names and rank-stable plot styles
METHOD_LABELS = {
    "DP": "Tabular DP",
    "factored": "Factored RL",
    "dqn": "DQN",
    "linear_ac": "Linear AC",
}
METHOD_COLORS = {
    "DP": COLORS["black"],
    "factored": COLORS["blue"],
    "dqn": ALGO_COLORS["DQN"],
    "linear_ac": ALGO_COLORS["Actor-Critic"],
}
METHOD_MARKERS = {"DP": "o", "factored": "s", "dqn": "^", "linear_ac": "D"}
RL_METHODS = ["factored", "dqn", "linear_ac"]


# =============================================================================
# Environment
# =============================================================================
class ExtendedWindFarmEnv:
    """Wind farm battery storage MDP with d >= 3 state dimensions.

    Base state (w, p, c) = wind power, spot price, battery state-of-charge.
    Auxiliary dimensions (d > 3) are AR(1) processes weakly coupled to wind.
    Reward depends only on the base state, never on the auxiliary dims.
    """

    def __init__(self, n_dims=3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = None
        self.n_dims = max(3, n_dims)
        self.n_extra = self.n_dims - 3

    def reset(self):
        self.t = 0
        w = 50.0 + self.rng.normal(0, 10)
        p = 0.5 + self.rng.normal(0, 0.1)
        c = 25.0
        base_state = [np.clip(w, W_MIN, W_MAX), np.clip(p, P_MIN, P_MAX), c]
        extra_state = [0.5 for _ in range(self.n_extra)]
        self.state = np.array(base_state + extra_state)
        return self.state.copy()

    def step(self, action):
        w, p, c = self.state[:3]
        a = np.clip(action, A_MIN, A_MAX)

        # Demand with diurnal pattern
        demand = self.rng.poisson(50 + 10 * np.sin(2 * np.pi * self.t / 24))

        # Reward: revenue - storage cost - shortage penalty (base state only)
        supply = w + a
        revenue = p * min(supply, demand)
        storage_cost = 0.01 * c
        shortage_penalty = 5 * max(0, demand - supply)
        reward = revenue - storage_cost - shortage_penalty

        # AR(1) transitions for base dims
        eps_w = self.rng.normal(30, 5)
        eps_p = self.rng.normal(0.4, 0.1)
        w_next = np.clip(0.7 * w + eps_w, W_MIN, W_MAX)
        p_next = np.clip(0.6 * p + 0.05 * (w / 100) + eps_p, P_MIN, P_MAX)
        c_next = np.clip(c + 0.9 * a, C_MIN, C_MAX)
        next_state = [w_next, p_next, c_next]

        # Auxiliary dims: AR(1) with weak coupling to wind
        for i in range(self.n_extra):
            x_i = self.state[3 + i]
            coupling = 0.01 * (w / W_MAX - 0.5)
            eps_i = self.rng.normal(0, 0.05)
            next_state.append(
                np.clip(0.8 * x_i + coupling + 0.1 + eps_i, EXTRA_MIN, EXTRA_MAX)
            )

        self.state = np.array(next_state)
        self.t += 1
        done = self.t >= HORIZON
        return self.state.copy(), reward, done, {}


# =============================================================================
# Tabular DP (backward induction with wall-clock timeout)
# =============================================================================
class TabularDP:
    """Backward induction over a discretized state grid.

    State space: n_bins^d states. At n_bins=7: 343 (d=3), 2,401 (d=4),
    16,807 (d=5), 117,649 (d=6). Expectations under the Bellman operator are
    Monte Carlo estimates with n_mc_samples draws per state-action pair.
    solve() aborts after timeout_minutes of wall time and reports completion.
    """

    def __init__(self, n_dims=3, n_bins=7, n_action_bins=N_ACTIONS, n_mc_samples=20):
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.n_action_bins = n_action_bins
        self.n_mc_samples = n_mc_samples
        self.n_states = n_bins**n_dims
        self.state_grids = self._build_state_grids()
        self.action_grid = np.linspace(A_MIN, A_MAX, n_action_bins)
        self.V = None
        self.policy = None
        self.elapsed_time = 0
        self.completed = False

    def _build_state_grids(self):
        grids = [
            np.linspace(W_MIN, W_MAX, self.n_bins),
            np.linspace(P_MIN, P_MAX, self.n_bins),
            np.linspace(C_MIN, C_MAX, self.n_bins),
        ]
        for _ in range(self.n_dims - 3):
            grids.append(np.linspace(EXTRA_MIN, EXTRA_MAX, self.n_bins))
        return grids

    def _state_to_idx(self, state_tuple):
        idx = 0
        mult = 1
        for i in reversed(range(self.n_dims)):
            idx += state_tuple[i] * mult
            mult *= self.n_bins
        return idx

    def _idx_to_state(self, idx):
        state_tuple = []
        for _ in range(self.n_dims):
            state_tuple.append(idx % self.n_bins)
            idx //= self.n_bins
        return tuple(reversed(state_tuple))

    def _get_state_values(self, state_tuple):
        return np.array([self.state_grids[i][b] for i, b in enumerate(state_tuple)])

    def _discretize_state(self, state):
        indices = []
        for i in range(self.n_dims):
            grid = self.state_grids[i]
            idx = np.clip(np.searchsorted(grid, state[i]) - 1, 0, self.n_bins - 1)
            indices.append(idx)
        return tuple(indices)

    def _sample_next_state(self, state, action, rng):
        w, p, c = state[:3]
        a = np.clip(action, A_MIN, A_MAX)
        eps_w = rng.normal(30, 5)
        eps_p = rng.normal(0.4, 0.1)
        w_next = np.clip(0.7 * w + eps_w, W_MIN, W_MAX)
        p_next = np.clip(0.6 * p + 0.05 * (w / 100) + eps_p, P_MIN, P_MAX)
        c_next = np.clip(c + 0.9 * a, C_MIN, C_MAX)
        next_state = [w_next, p_next, c_next]
        for i in range(self.n_dims - 3):
            x_i = state[3 + i]
            coupling = 0.01 * (w / W_MAX - 0.5)
            eps_i = rng.normal(0, 0.05)
            next_state.append(
                np.clip(0.8 * x_i + coupling + 0.1 + eps_i, EXTRA_MIN, EXTRA_MAX)
            )
        return np.array(next_state)

    def _compute_reward(self, state, action, t, rng):
        w, p, c = state[:3]
        a = np.clip(action, A_MIN, A_MAX)
        demand = rng.poisson(50 + 10 * np.sin(2 * np.pi * t / 24))
        supply = w + a
        revenue = p * min(supply, demand)
        storage_cost = 0.01 * c
        shortage_penalty = 5 * max(0, demand - supply)
        return revenue - storage_cost - shortage_penalty

    def solve(self, timeout_minutes=10, dp_seed=42, verbose=True):
        """Backward induction. Returns (completed, elapsed_seconds)."""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        if verbose:
            print(
                f"  DP: n_dims={self.n_dims}, n_bins={self.n_bins}, "
                f"n_states={self.n_states:,}, timeout={timeout_minutes}min"
            )

        self.V = [{} for _ in range(HORIZON + 1)]
        self.policy = [{} for _ in range(HORIZON)]
        rng = np.random.default_rng(dp_seed)

        for t in range(HORIZON - 1, -1, -1):
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                self.elapsed_time = elapsed
                self.completed = False
                if verbose:
                    print(f"  DP TIMEOUT after {elapsed:.1f}s at t={t}")
                return self.completed, self.elapsed_time
            if verbose:
                print(f"    DP d={self.n_dims} t={t} elapsed={elapsed:.1f}s")

            for state_idx in range(self.n_states):
                if state_idx > 0 and state_idx % 1000 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        self.elapsed_time = elapsed
                        self.completed = False
                        if verbose:
                            print(
                                f"  DP TIMEOUT after {elapsed:.1f}s "
                                f"at t={t}, state={state_idx}"
                            )
                        return self.completed, self.elapsed_time

                state = self._get_state_values(self._idx_to_state(state_idx))
                best_q = -np.inf
                best_a_idx = 0
                for a_idx, action in enumerate(self.action_grid):
                    q_samples = []
                    for _ in range(self.n_mc_samples):
                        reward = self._compute_reward(state, action, t, rng)
                        next_state = self._sample_next_state(state, action, rng)
                        next_idx = self._state_to_idx(
                            self._discretize_state(next_state)
                        )
                        v_next = (
                            0 if t == HORIZON - 1 else self.V[t + 1].get(next_idx, 0)
                        )
                        q_samples.append(reward + GAMMA * v_next)
                    q_value = np.mean(q_samples)
                    if q_value > best_q:
                        best_q = q_value
                        best_a_idx = a_idx
                self.V[t][state_idx] = best_q
                self.policy[t][state_idx] = best_a_idx

        self.elapsed_time = time.time() - start_time
        self.completed = True
        if verbose:
            print(f"  DP COMPLETE in {self.elapsed_time:.1f}s")
        return self.completed, self.elapsed_time

    def get_action(self, state, t):
        state_idx = self._state_to_idx(self._discretize_state(state))
        t = min(t, HORIZON - 1)
        action_idx = self.policy[t].get(state_idx, self.n_action_bins // 2)
        return self.action_grid[action_idx]

    def evaluate(self, n_episodes=50, seed=99):
        if not self.completed:
            return None
        env = ExtendedWindFarmEnv(n_dims=self.n_dims, seed=seed)
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            for t in range(HORIZON):
                state, reward, done, _ = env.step(self.get_action(state, t))
                total_return += reward
                if done:
                    break
        return total_return / n_episodes


# =============================================================================
# Factored Q-learning
# =============================================================================
class FactoredQL:
    """Factored Q-learning: one Q-table per state dimension, summed for Q(s,a).

    The reward-relevant dimensions (the first reward_dims) are GIVEN to the
    agent, mirroring the known-structure premise of factored MDPs; the
    auxiliary dimensions receive zero TD-error weight and their tables stay at
    zero. The structure is assumed, not learned.
    """

    def __init__(self, n_dims=3, n_bins=30, alpha=0.2, epsilon=0.2, reward_dims=3):
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.reward_dims = min(reward_dims, n_dims)
        self.n_actions = N_ACTIONS
        self.action_grid = np.linspace(A_MIN, A_MAX, self.n_actions)
        self.Q_tables = [np.zeros((n_bins, self.n_actions)) for _ in range(n_dims)]
        self.grids = [
            np.linspace(W_MIN, W_MAX, n_bins),
            np.linspace(P_MIN, P_MAX, n_bins),
            np.linspace(C_MIN, C_MAX, n_bins),
        ]
        for _ in range(n_dims - 3):
            self.grids.append(np.linspace(EXTRA_MIN, EXTRA_MAX, n_bins))

    def discretize(self, state):
        indices = []
        for i in range(self.n_dims):
            idx = np.clip(
                np.searchsorted(self.grids[i], state[i]) - 1, 0, self.n_bins - 1
            )
            indices.append(idx)
        return indices

    def get_Q(self, state, action_idx):
        indices = self.discretize(state)
        return sum(self.Q_tables[i][indices[i], action_idx] for i in range(self.n_dims))

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = [self.get_Q(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

    def update(self, state, action_idx, reward, next_state, done):
        indices = self.discretize(state)
        if done:
            target = reward
        else:
            next_q = [self.get_Q(next_state, a) for a in range(self.n_actions)]
            target = reward + GAMMA * max(next_q)
        td_error = target - self.get_Q(state, action_idx)

        # TD error goes entirely to the given reward-relevant dims
        if self.n_dims <= self.reward_dims:
            weights = [1.0 / self.n_dims] * self.n_dims
        else:
            weights = [
                1.0 / self.reward_dims if i < self.reward_dims else 0.0
                for i in range(self.n_dims)
            ]
        for i in range(self.n_dims):
            self.Q_tables[i][indices[i], action_idx] += (
                self.alpha * td_error * weights[i]
            )

    def train(self, env, n_episodes=3000):
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_idx = self.get_action(state)
                next_state, reward, done, _ = env.step(self.action_grid[action_idx])
                self.update(state, action_idx, reward, next_state, done)
                state = next_state
                if done:
                    break

    def evaluate(self, env, n_episodes=50):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_idx = self.get_action(state, greedy=True)
                state, reward, done, _ = env.step(self.action_grid[action_idx])
                total_return += reward
                if done:
                    break
        return total_return / n_episodes


# =============================================================================
# Deep Q-Network
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Standard DQN with experience replay and a target network."""

    def __init__(
        self,
        state_dim=3,
        n_actions=N_ACTIONS,
        lr=5e-4,
        hidden=128,
        buffer_size=10000,
        batch_size=64,
        target_update=100,
        epsilon_start=0.3,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.action_grid = np.linspace(A_MIN, A_MAX, n_actions)
        self.q_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor([t[4] for t in batch])

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, env, n_episodes=3000):
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_idx = self.get_action(state)
                next_state, reward, done, _ = env.step(self.action_grid[action_idx])
                self.store(state, action_idx, reward, next_state, float(done))
                self.train_step()
                state = next_state
                if done:
                    break

    def evaluate(self, env, n_episodes=50):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_idx = self.get_action(state, greedy=True)
                state, reward, done, _ = env.step(self.action_grid[action_idx])
                total_return += reward
                if done:
                    break
        return total_return / n_episodes


# =============================================================================
# Linear actor-critic on polynomial features
# =============================================================================
class LinearAC:
    """Linear actor-critic on polynomial features (intercept, linear, squared,
    pairwise cross terms). Illustrates the linear function-approximation
    pathway; it is not the BiLin-UCB algorithm of Du et al. (2021).
    """

    def __init__(self, n_dims=3, lr_actor=0.02, lr_critic=0.08, n_actions=N_ACTIONS):
        self.n_dims = n_dims
        self.n_actions = n_actions
        self.action_grid = np.linspace(A_MIN, A_MAX, n_actions)
        # 1 + d + d + d(d-1)/2 features; d=3: 10, d=6: 28
        self.feature_dim = 1 + 2 * n_dims + (n_dims * (n_dims - 1)) // 2
        self.theta_v = np.zeros(self.feature_dim)
        self.theta_mu = np.zeros(self.feature_dim)
        self.log_std = 0.0
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.bounds = [(W_MIN, W_MAX), (P_MIN, P_MAX), (C_MIN, C_MAX)]
        for _ in range(n_dims - 3):
            self.bounds.append((EXTRA_MIN, EXTRA_MAX))

    def features(self, state):
        normalized = []
        for i in range(self.n_dims):
            lo, hi = self.bounds[i]
            normalized.append((state[i] - lo) / (hi - lo + 1e-8))
        x = np.array(normalized)
        features = [1.0]
        features.extend(x)
        features.extend(x**2)
        for i in range(self.n_dims):
            for j in range(i + 1, self.n_dims):
                features.append(x[i] * x[j])
        return np.array(features)

    def get_action(self, state, greedy=False):
        phi = self.features(state)
        mu = np.dot(self.theta_mu, phi)
        if greedy:
            action = np.clip(mu * (A_MAX - A_MIN) / 2, A_MIN, A_MAX)
            return np.argmin(np.abs(self.action_grid - action))
        std = np.exp(self.log_std)
        action_raw = mu + std * np.random.randn()
        action = np.clip(action_raw * (A_MAX - A_MIN) / 2, A_MIN, A_MAX)
        action_idx = np.argmin(np.abs(self.action_grid - action))
        return action_idx, action_raw, mu, std

    def update(self, state, action_info, reward, next_state, done):
        action_idx, action_raw, mu, std = action_info
        phi = self.features(state)
        phi_next = self.features(next_state)
        v = np.dot(self.theta_v, phi)
        v_next = 0 if done else np.dot(self.theta_v, phi_next)
        td_error = reward + GAMMA * v_next - v
        self.theta_v += self.lr_critic * td_error * phi
        score = (action_raw - mu) / (std**2 + 1e-8) * phi
        self.theta_mu += self.lr_actor * td_error * score
        score_std = ((action_raw - mu) ** 2 - std**2) / (std**2 + 1e-8)
        self.log_std += self.lr_actor * td_error * score_std * 0.1
        self.log_std = np.clip(self.log_std, -2, 1)

    def train(self, env, n_episodes=3000):
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_info = self.get_action(state)
                next_state, reward, done, _ = env.step(self.action_grid[action_info[0]])
                self.update(state, action_info, reward, next_state, done)
                state = next_state
                if done:
                    break

    def evaluate(self, env, n_episodes=50):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            while True:
                action_idx = self.get_action(state, greedy=True)
                state, reward, done, _ = env.step(self.action_grid[action_idx])
                total_return += reward
                if done:
                    break
        return total_return / n_episodes


# =============================================================================
# Component compute functions
# =============================================================================


def compute_dp(d):
    """Solve tabular DP at dimension d with wall-clock timeout, then evaluate."""
    dp = TabularDP(n_dims=d, n_bins=DP_CONFIG["N_BINS"], n_mc_samples=DP_CONFIG["N_MC"])
    completed, elapsed = dp.solve(
        timeout_minutes=DP_CONFIG["TIMEOUT_MIN"], dp_seed=DP_CONFIG["DP_SEED"]
    )
    ret = (
        dp.evaluate(n_episodes=DP_CONFIG["EVAL_EPISODES"], seed=DP_CONFIG["EVAL_SEED"])
        if completed
        else None
    )
    if completed:
        print(f"  DP d={d}: return={ret:.1f}, time={elapsed:.1f}s")
    return {
        "completed": completed,
        "time": elapsed,
        "return": ret,
        "n_states": dp.n_states,
    }


def _make_agent(method, d):
    if method == "factored":
        return FactoredQL(
            n_dims=d,
            n_bins=FACTORED_CONFIG["N_BINS"],
            alpha=FACTORED_CONFIG["ALPHA"],
            epsilon=FACTORED_CONFIG["EPSILON"],
            reward_dims=FACTORED_CONFIG["REWARD_DIMS"],
        )
    if method == "dqn":
        return DQNAgent(
            state_dim=d,
            lr=DQN_CONFIG["LR"],
            hidden=DQN_CONFIG["HIDDEN"],
            buffer_size=DQN_CONFIG["BUFFER"],
            batch_size=DQN_CONFIG["BATCH"],
            target_update=DQN_CONFIG["TARGET_UPDATE"],
            epsilon_start=DQN_CONFIG["EPS_START"],
            epsilon_end=DQN_CONFIG["EPS_END"],
            epsilon_decay=DQN_CONFIG["EPS_DECAY"],
        )
    if method == "linear_ac":
        return LinearAC(
            n_dims=d,
            lr_actor=LINEAR_AC_CONFIG["LR_ACTOR"],
            lr_critic=LINEAR_AC_CONFIG["LR_CRITIC"],
        )
    raise ValueError(method)


def compute_rl(method, d):
    """Train one RL method at dimension d across N_SEEDS seeds.

    Each (method, dim, seed) unit re-seeds numpy and torch so components are
    reproducible independently of execution order. Timing covers training
    only; evaluation runs after the clock stops, greedy, on a common
    EVAL_SEED environment (paired evaluation shocks across methods).
    """
    returns, times = [], []
    for seed in range(RL_COMMON["N_SEEDS"]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = ExtendedWindFarmEnv(n_dims=d, seed=seed)
        agent = _make_agent(method, d)
        t_start = time.time()
        agent.train(env, n_episodes=RL_COMMON["N_EPISODES"])
        t_elapsed = time.time() - t_start
        eval_env = ExtendedWindFarmEnv(n_dims=d, seed=EVAL_PARAMS["EVAL_SEED"])
        ret = agent.evaluate(eval_env, n_episodes=EVAL_PARAMS["EVAL_EPISODES"])
        returns.append(ret)
        times.append(t_elapsed)
        print(
            f"  {method} d={d} seed={seed}: return={ret:.1f}, "
            f"train_time={t_elapsed:.1f}s"
        )
    return {"returns": returns, "times": times}


COMPONENT_CONFIGS = {
    "DP": DP_CONFIG,
    "factored": FACTORED_CONFIG,
    "dqn": DQN_CONFIG,
    "linear_ac": LINEAR_AC_CONFIG,
}

# Cheapest method first so plumbing bugs surface in minutes
COMPUTE_ORDER = ["linear_ac", "factored", "DP", "dqn"]


def compute_data(force=None):
    force = force or set()
    data = {m: {} for m in COMPONENT_CONFIGS}
    for method in COMPUTE_ORDER:
        for d in DIMS:
            component = f"{method}_d{d}"
            cfg = {**COMPONENT_CONFIGS[method], "DIM": d}
            forced = component in force or method in force
            if method == "DP":
                data[method][d] = compute_or_load(
                    CACHE_DIR, SCRIPT_NAME, component, cfg, compute_dp, d, force=forced
                )
            else:
                data[method][d] = compute_or_load(
                    CACHE_DIR,
                    SCRIPT_NAME,
                    component,
                    cfg,
                    compute_rl,
                    method,
                    d,
                    force=forced,
                )
    return data


# =============================================================================
# Outputs
# =============================================================================


def _rl_stats(data, method, d):
    returns = np.array(data[method][d]["returns"])
    mean = returns.mean()
    se = returns.std(ddof=1) / np.sqrt(len(returns))
    return mean, se


def _format_se(se):
    """Format an SE with enough digits that it never displays as zero."""
    assert se > 0, "zero SE cell: identical returns across seeds"
    digits = 1
    while round(se, digits) == 0 and digits < 4:
        digits += 1
    return f"{se:.{digits}f}"


def _dp_extrapolation(data):
    """Exponential fit through the completed DP dims, extrapolated to all DIMS.

    Returns (growth_rate, {d: predicted_seconds}).
    """
    completed = [(d, data["DP"][d]["time"]) for d in DIMS if data["DP"][d]["completed"]]
    if len(completed) < 2:
        return None, {}
    (d0, t0), (d1, t1) = completed[0], completed[-1]
    growth_rate = (np.log(t1) - np.log(t0)) / (d1 - d0)
    pred = {d: t0 * np.exp(growth_rate * (d - d0)) for d in DIMS}
    return growth_rate, pred


def _rank_order(data):
    """Methods ranked by mean return at d=3 (the dim where all complete)."""
    entries = [("DP", data["DP"][3]["return"])]
    for m in RL_METHODS:
        entries.append((m, _rl_stats(data, m, 3)[0]))
    return [m for m, _ in sorted(entries, key=lambda e: -e[1])]


def plot_computation_times(data, save_path):
    """Log-scale computation time vs dimension for all methods."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    dp_completed = [d for d in DIMS if data["DP"][d]["completed"]]
    dp_timeout = [d for d in DIMS if not data["DP"][d]["completed"]]

    _, pred = _dp_extrapolation(data)
    if pred:
        ref_dims = np.linspace(min(DIMS), max(DIMS), 50)
        growth, _ = _dp_extrapolation(data)
        d0 = dp_completed[0]
        t0 = data["DP"][d0]["time"]
        ax.semilogy(
            ref_dims,
            t0 * np.exp(growth * (ref_dims - d0)),
            linestyle=":",
            color=METHOD_COLORS["DP"],
            alpha=0.5,
            label="DP extrapolation",
        )

    order = _rank_order(data)
    for method in order:
        color, marker = METHOD_COLORS[method], METHOD_MARKERS[method]
        label = METHOD_LABELS[method]
        if method == "DP":
            if dp_completed:
                ax.semilogy(
                    dp_completed,
                    [data["DP"][d]["time"] for d in dp_completed],
                    color=color,
                    marker=marker,
                    markersize=9,
                    label=label,
                    zorder=5,
                )
            if dp_timeout:
                ax.semilogy(
                    dp_timeout,
                    [data["DP"][d]["time"] for d in dp_timeout],
                    linestyle="none",
                    color=color,
                    marker="x",
                    markersize=11,
                    markeredgewidth=2.5,
                    label="DP (timeout)",
                    zorder=5,
                )
        else:
            times = [np.mean(data[method][d]["times"]) for d in DIMS]
            ax.semilogy(
                DIMS, times, color=color, marker=marker, markersize=7, label=label
            )

    ax.axhline(
        y=DP_CONFIG["TIMEOUT_MIN"] * 60,
        **BENCH_STYLE,
        label=f"{DP_CONFIG['TIMEOUT_MIN']} min timeout",
    )
    ax.set_xlabel("State dimension $d$")
    ax.set_ylabel("Computation time (seconds, log scale)")
    ax.set_xticks(DIMS)
    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_results_table(data, save_path):
    """LaTeX tabular: average return by method and dimension, rank-ordered."""
    order = _rank_order(data)
    lines = [
        r"\begin{tabular}{l" + "c" * len(DIMS) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$d={d}$" for d in DIMS) + r" \\",
        r"\midrule",
    ]
    for method in order:
        cells = []
        for d in DIMS:
            if method == "DP":
                r = data["DP"][d]
                cells.append(f"{r['return']:.0f}" if r["completed"] else "TIMEOUT")
            else:
                mean, se = _rl_stats(data, method, d)
                cells.append(f"{mean:.0f} $\\pm$ {_format_se(se)}")
        lines.append(METHOD_LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {save_path}")


def generate_outputs(data):
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Horizon: {HORIZON}, Discount: {GAMMA}, Dims: {DIMS}")
    print(
        f"RL: {RL_COMMON['N_EPISODES']} episodes, {RL_COMMON['N_SEEDS']} seeds; "
        f"eval: seed {EVAL_PARAMS['EVAL_SEED']}, "
        f"{EVAL_PARAMS['EVAL_EPISODES']} episodes, greedy"
    )
    print(
        f"DP: n_bins={DP_CONFIG['N_BINS']}, n_mc={DP_CONFIG['N_MC']}, "
        f"timeout={DP_CONFIG['TIMEOUT_MIN']}min, seed={DP_CONFIG['DP_SEED']}"
    )

    print()
    print("DP results:")
    print(f"{'dim':<5} {'states':<10} {'time_s':<10} {'return':<10}")
    for d in DIMS:
        r = data["DP"][d]
        ret = f"{r['return']:.1f}" if r["completed"] else "TIMEOUT"
        print(f"{d:<5} {r['n_states']:<10,} {r['time']:<10.1f} {ret:<10}")

    growth, pred = _dp_extrapolation(data)
    if pred:
        print()
        print(
            f"DP exponential fit: growth rate {growth:.3f} per dim "
            f"(x{np.exp(growth):.1f} per added dimension)"
        )
        for d in DIMS:
            status = (
                "completed"
                if data["DP"][d]["completed"]
                else "TIMEOUT"
                if d in [dd for dd in DIMS if not data["DP"][dd]["completed"]]
                else ""
            )
            hours = f" = {pred[d] / 3600:.1f} hours" if pred[d] >= 3600 else ""
            print(
                f"  extrapolated DP time d={d}: {pred[d]:.0f}s "
                f"({pred[d] / 60:.1f} min{hours}) [{status}]"
            )

    print()
    print(
        f"RL results (mean +/- SE over {RL_COMMON['N_SEEDS']} seeds; train time mean):"
    )
    header = f"{'method':<12}"
    for d in DIMS:
        header += f"{'d=' + str(d):<20}"
    print(header)
    for method in RL_METHODS:
        row = f"{method:<12}"
        for d in DIMS:
            mean, se = _rl_stats(data, method, d)
            row += f"{mean:.1f} +/- {_format_se(se):<10}"
        print(row)
    print()
    print("RL train times (mean seconds per seed):")
    for method in RL_METHODS:
        row = f"{method:<12}"
        for d in DIMS:
            row += f"{np.mean(data[method][d]['times']):<10.1f}"
        print(row)

    print()
    print("Per-seed returns:")
    for method in RL_METHODS:
        for d in DIMS:
            rets = ", ".join(f"{r:.1f}" for r in data[method][d]["returns"])
            print(f"  {method} d={d}: [{rets}]")

    print()
    print(
        f"Rank order (by mean return at d=3): "
        f"{[METHOD_LABELS[m] for m in _rank_order(data)]}"
    )

    fig_path = os.path.join(OUTDIR, f"{SCRIPT_NAME}_times.png")
    tab_path = os.path.join(OUTDIR, f"{SCRIPT_NAME}_results.tex")
    plot_computation_times(data, fig_path)
    generate_results_table(data, tab_path)

    print()
    print("Output files:")
    print(f"  - {fig_path}")
    print(f"  - {tab_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Wind farm storage control: curse of dimensionality study"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("WIND FARM STORAGE CONTROL: CURSE OF DIMENSIONALITY STUDY")
    print("=" * 70)

    if args.plots_only:
        data = compute_data()
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)


if __name__ == "__main__":
    main()
