"""
Wind Farm Storage Control: Curse of Dimensionality Study
Chapter 3 - Theory
Demonstrates how modern RL methods break the curse through structural exploitation.
Includes Tabular DP baseline with timeout to show exponential blowup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
HORIZON = 24  # hours
GAMMA = 0.95

# State bounds
W_MIN, W_MAX = 0, 100  # wind power (kW)
P_MIN, P_MAX = 0, 1     # price ($/kWh)
C_MIN, C_MAX = 0, 50    # battery SoC (kWh)

# Action bounds
A_MIN, A_MAX = -20, 20  # charge/discharge (kW)

# Extra dimension bounds (for scaling study)
EXTRA_MIN, EXTRA_MAX = 0, 1

# DP timeout
DP_TIMEOUT_MINUTES = 10  # 10 minutes per dimension

# =============================================================================
# Base Environment (d=3)
# =============================================================================
class WindFarmEnv:
    """Wind farm storage control MDP (d=3)."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = None
        self.n_dims = 3

    def reset(self):
        self.t = 0
        w = 50.0 + self.rng.normal(0, 10)
        p = 0.5 + self.rng.normal(0, 0.1)
        c = 25.0
        self.state = np.array([
            np.clip(w, W_MIN, W_MAX),
            np.clip(p, P_MIN, P_MAX),
            c
        ])
        return self.state.copy()

    def step(self, action):
        w, p, c = self.state[:3]
        a = np.clip(action, A_MIN, A_MAX)

        # Demand (hourly pattern)
        demand = self.rng.poisson(50 + 10 * np.sin(2 * np.pi * self.t / 24))

        # Reward: revenue - storage cost - shortage penalty
        supply = w + a
        revenue = p * min(supply, demand)
        storage_cost = 0.01 * c
        shortage_penalty = 5 * max(0, demand - supply)
        reward = revenue - storage_cost - shortage_penalty

        # Transitions (AR(1) dynamics)
        eps_w = self.rng.normal(30, 5)
        eps_p = self.rng.normal(0.4, 0.1)

        w_next = np.clip(0.7 * w + eps_w, W_MIN, W_MAX)
        p_next = np.clip(0.6 * p + 0.05 * (w / 100) + eps_p, P_MIN, P_MAX)
        c_next = np.clip(c + 0.9 * a, C_MIN, C_MAX)

        self.state = np.array([w_next, p_next, c_next])
        self.t += 1
        done = (self.t >= HORIZON)

        return self.state.copy(), reward, done, {}


# =============================================================================
# Extended Environment (d >= 3) for Scaling Study
# =============================================================================
class ExtendedWindFarmEnv:
    """Wind farm with additional state dimensions for scaling study.

    Extra dimensions are AR(1) processes with weak coupling to base state.
    Reward depends only on first 3 dims (w, p, c).
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

        base_state = [
            np.clip(w, W_MIN, W_MAX),
            np.clip(p, P_MIN, P_MAX),
            c
        ]

        # Extra dimensions: start at midpoint
        extra_state = [0.5 for _ in range(self.n_extra)]

        self.state = np.array(base_state + extra_state)
        return self.state.copy()

    def step(self, action):
        w, p, c = self.state[:3]
        a = np.clip(action, A_MIN, A_MAX)

        # Demand (hourly pattern)
        demand = self.rng.poisson(50 + 10 * np.sin(2 * np.pi * self.t / 24))

        # Reward: revenue - storage cost - shortage penalty (ONLY depends on w, p, c)
        supply = w + a
        revenue = p * min(supply, demand)
        storage_cost = 0.01 * c
        shortage_penalty = 5 * max(0, demand - supply)
        reward = revenue - storage_cost - shortage_penalty

        # Transitions for base dims (AR(1) dynamics)
        eps_w = self.rng.normal(30, 5)
        eps_p = self.rng.normal(0.4, 0.1)

        w_next = np.clip(0.7 * w + eps_w, W_MIN, W_MAX)
        p_next = np.clip(0.6 * p + 0.05 * (w / 100) + eps_p, P_MIN, P_MAX)
        c_next = np.clip(c + 0.9 * a, C_MIN, C_MAX)

        next_state = [w_next, p_next, c_next]

        # Extra dimensions: AR(1) with weak coupling to wind
        for i in range(self.n_extra):
            x_i = self.state[3 + i]
            # Weak coupling: 0.01 * normalized wind
            coupling = 0.01 * (w / W_MAX - 0.5)
            eps_i = self.rng.normal(0, 0.05)
            x_next = np.clip(0.8 * x_i + coupling + 0.1 + eps_i, EXTRA_MIN, EXTRA_MAX)
            next_state.append(x_next)

        self.state = np.array(next_state)
        self.t += 1
        done = (self.t >= HORIZON)

        return self.state.copy(), reward, done, {}


# =============================================================================
# Tabular DP (Backward Induction with Timeout)
# =============================================================================
class TabularDP:
    """Backward induction with discretized state space.

    State space: n_bins^d states.
    For d=3, n_bins=15: 3,375 states
    For d=4, n_bins=15: 50,625 states
    For d=5, n_bins=15: 759,375 states (likely timeout)
    For d=6, n_bins=15: 11,390,625 states (will timeout)
    """

    def __init__(self, n_dims=3, n_bins=15, n_action_bins=11, n_mc_samples=50):
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.n_action_bins = n_action_bins
        self.n_mc_samples = n_mc_samples

        # State space size
        self.n_states = n_bins ** n_dims

        # Build grids
        self.state_grids = self._build_state_grids()
        self.action_grid = np.linspace(A_MIN, A_MAX, n_action_bins)

        # Value function: V[t, state_idx]
        self.V = None
        self.policy = None

        # Timing
        self.elapsed_time = 0
        self.completed = False

    def _build_state_grids(self):
        """Build discretization grids for each dimension."""
        grids = [
            np.linspace(W_MIN, W_MAX, self.n_bins),   # wind
            np.linspace(P_MIN, P_MAX, self.n_bins),   # price
            np.linspace(C_MIN, C_MAX, self.n_bins),   # SoC
        ]
        # Extra dimensions
        for _ in range(self.n_dims - 3):
            grids.append(np.linspace(EXTRA_MIN, EXTRA_MAX, self.n_bins))
        return grids

    def _state_to_idx(self, state_tuple):
        """Convert tuple of bin indices to flat index."""
        idx = 0
        mult = 1
        for i in reversed(range(self.n_dims)):
            idx += state_tuple[i] * mult
            mult *= self.n_bins
        return idx

    def _idx_to_state(self, idx):
        """Convert flat index to tuple of bin indices."""
        state_tuple = []
        for _ in range(self.n_dims):
            state_tuple.append(idx % self.n_bins)
            idx //= self.n_bins
        return tuple(reversed(state_tuple))

    def _get_state_values(self, state_tuple):
        """Get continuous state values from bin indices."""
        values = []
        for i, bin_idx in enumerate(state_tuple):
            values.append(self.state_grids[i][bin_idx])
        return np.array(values)

    def _discretize_state(self, state):
        """Convert continuous state to bin indices."""
        indices = []
        for i in range(self.n_dims):
            grid = self.state_grids[i]
            idx = np.clip(np.searchsorted(grid, state[i]) - 1, 0, self.n_bins - 1)
            indices.append(idx)
        return tuple(indices)

    def _sample_next_state(self, state, action, rng):
        """Sample next state given current state and action."""
        w, p, c = state[:3]
        a = np.clip(action, A_MIN, A_MAX)

        # Transitions (AR(1) dynamics)
        eps_w = rng.normal(30, 5)
        eps_p = rng.normal(0.4, 0.1)

        w_next = np.clip(0.7 * w + eps_w, W_MIN, W_MAX)
        p_next = np.clip(0.6 * p + 0.05 * (w / 100) + eps_p, P_MIN, P_MAX)
        c_next = np.clip(c + 0.9 * a, C_MIN, C_MAX)

        next_state = [w_next, p_next, c_next]

        # Extra dimensions
        for i in range(self.n_dims - 3):
            x_i = state[3 + i]
            coupling = 0.01 * (w / W_MAX - 0.5)
            eps_i = rng.normal(0, 0.05)
            x_next = np.clip(0.8 * x_i + coupling + 0.1 + eps_i, EXTRA_MIN, EXTRA_MAX)
            next_state.append(x_next)

        return np.array(next_state)

    def _compute_reward(self, state, action, t, rng):
        """Compute reward for state-action pair."""
        w, p, c = state[:3]
        a = np.clip(action, A_MIN, A_MAX)

        # Demand (hourly pattern)
        demand = rng.poisson(50 + 10 * np.sin(2 * np.pi * t / 24))

        # Reward: revenue - storage cost - shortage penalty
        supply = w + a
        revenue = p * min(supply, demand)
        storage_cost = 0.01 * c
        shortage_penalty = 5 * max(0, demand - supply)
        reward = revenue - storage_cost - shortage_penalty

        return reward

    def solve(self, timeout_minutes=30, verbose=True):
        """Run backward induction with timeout.

        Returns:
            completed: bool, whether DP finished before timeout
            elapsed_time: float, seconds elapsed
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        if verbose:
            print(f"  DP: n_dims={self.n_dims}, n_bins={self.n_bins}, "
                  f"n_states={self.n_states:,}, timeout={timeout_minutes}min")

        # Initialize value function and policy
        # V[t][state_idx] = value at time t, state s
        # policy[t][state_idx] = action index at time t, state s
        self.V = [{} for _ in range(HORIZON + 1)]
        self.policy = [{} for _ in range(HORIZON)]

        # Terminal values are 0
        # Initialize only states we visit (lazy evaluation)

        rng = np.random.default_rng(42)

        # Backward induction: t = H-1, H-2, ..., 0
        states_processed = 0
        time_steps = range(HORIZON - 1, -1, -1)
        if verbose:
            time_steps = tqdm(time_steps, desc=f"    DP d={self.n_dims}", leave=True)

        for t in time_steps:
            # Check timeout at start of each time step
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                self.elapsed_time = elapsed
                self.completed = False
                if verbose:
                    print(f"\n  DP TIMEOUT after {elapsed:.1f}s at t={t}")
                return self.completed, self.elapsed_time

            # Iterate over all states
            for state_idx in range(self.n_states):
                # Check timeout periodically (every 1000 states)
                if state_idx > 0 and state_idx % 1000 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        self.elapsed_time = elapsed
                        self.completed = False
                        if verbose:
                            print(f"\n  DP TIMEOUT after {elapsed:.1f}s at t={t}, state={state_idx}")
                        return self.completed, self.elapsed_time

                state_tuple = self._idx_to_state(state_idx)
                state = self._get_state_values(state_tuple)

                # Compute Q(s, a) for each action
                best_q = -np.inf
                best_a_idx = 0

                for a_idx, action in enumerate(self.action_grid):
                    # Monte Carlo estimate of E[r + gamma * V(s')]
                    q_samples = []
                    for _ in range(self.n_mc_samples):
                        reward = self._compute_reward(state, action, t, rng)
                        next_state = self._sample_next_state(state, action, rng)
                        next_tuple = self._discretize_state(next_state)
                        next_idx = self._state_to_idx(next_tuple)

                        # Get V(s') from next time step (0 if terminal)
                        if t == HORIZON - 1:
                            v_next = 0
                        else:
                            v_next = self.V[t + 1].get(next_idx, 0)

                        q_samples.append(reward + GAMMA * v_next)

                    q_value = np.mean(q_samples)
                    if q_value > best_q:
                        best_q = q_value
                        best_a_idx = a_idx

                self.V[t][state_idx] = best_q
                self.policy[t][state_idx] = best_a_idx
                states_processed += 1

        self.elapsed_time = time.time() - start_time
        self.completed = True
        if verbose:
            print(f"  DP COMPLETE in {self.elapsed_time:.1f}s")

        return self.completed, self.elapsed_time

    def get_action(self, state, t):
        """Get action from computed policy."""
        if self.policy is None:
            raise ValueError("Must call solve() first")

        state_tuple = self._discretize_state(state)
        state_idx = self._state_to_idx(state_tuple)

        # Clamp t to valid range
        t = min(t, HORIZON - 1)

        action_idx = self.policy[t].get(state_idx, self.n_action_bins // 2)
        return self.action_grid[action_idx]

    def evaluate(self, n_episodes=100, seed=99):
        """Evaluate DP policy in environment."""
        if not self.completed:
            return None

        env = ExtendedWindFarmEnv(n_dims=self.n_dims, seed=seed)
        total_return = 0

        for _ in range(n_episodes):
            state = env.reset()
            ep_return = 0
            for t in range(HORIZON):
                action = self.get_action(state, t)
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                state = next_state
                if done:
                    break
            total_return += ep_return

        return total_return / n_episodes


# =============================================================================
# Factored Q-Learning (Lu et al. 2025 style)
# =============================================================================
class FactoredQL:
    """Factored Q-learning exploiting weak coupling structure."""

    def __init__(self, n_dims=3, n_bins=20, alpha=0.1, epsilon=0.1, reward_dims=3):
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.reward_dims = min(reward_dims, n_dims)  # dims that affect reward

        self.n_actions = 11
        self.action_grid = np.linspace(A_MIN, A_MAX, self.n_actions)

        # Separate Q-tables for each dimension
        self.Q_tables = [np.zeros((n_bins, self.n_actions)) for _ in range(n_dims)]

        # Grids
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
            grid = self.grids[i]
            idx = np.clip(np.searchsorted(grid, state[i]) - 1, 0, self.n_bins - 1)
            indices.append(idx)
        return indices

    def get_Q(self, state, action_idx):
        indices = self.discretize(state)
        q_sum = sum(self.Q_tables[i][indices[i], action_idx] for i in range(self.n_dims))
        return q_sum

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
            next_q_values = [self.get_Q(next_state, a) for a in range(self.n_actions)]
            target = reward + GAMMA * max(next_q_values)

        current_q = self.get_Q(state, action_idx)
        td_error = target - current_q

        # Weight TD error by reward relevance: reward dims get most of TD error
        if self.n_dims <= self.reward_dims:
            weights = [1.0 / self.n_dims] * self.n_dims
        else:
            # Reward dims share 100% of TD error, irrelevant dims get zero
            # This prevents irrelevant dimensions from diluting learning signal
            reward_weight = 1.0 / self.reward_dims
            other_weight = 0.0
            weights = [reward_weight if i < self.reward_dims else other_weight for i in range(self.n_dims)]

        # Distribute TD error across factors
        for i in range(self.n_dims):
            self.Q_tables[i][indices[i], action_idx] += self.alpha * td_error * weights[i]

    def train(self, env, n_episodes=3000, eval_interval=300):
        eval_returns = []

        for ep in range(n_episodes):
            state = env.reset()

            while True:
                action_idx = self.get_action(state)
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)
                self.update(state, action_idx, reward, next_state, done)
                state = next_state
                if done:
                    break

            if (ep + 1) % eval_interval == 0:
                eval_ret = self.evaluate(ExtendedWindFarmEnv(n_dims=self.n_dims, seed=99), n_episodes=20)
                eval_returns.append((ep + 1, eval_ret))

        return eval_returns

    def evaluate(self, env, n_episodes=100):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            ep_return = 0
            while True:
                action_idx = self.get_action(state, greedy=True)
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                state = next_state
                if done:
                    break
            total_return += ep_return
        return total_return / n_episodes

    def param_count(self):
        return self.n_dims * self.n_bins * self.n_actions


# =============================================================================
# Deep Q-Network (Liu et al. 2022 style)
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Deep Q-Network with experience replay and target network."""

    def __init__(self, state_dim=3, n_actions=11, lr=1e-3, buffer_size=10000,
                 batch_size=64, target_update=100, epsilon_start=0.3, epsilon_end=0.01,
                 epsilon_decay=0.9995):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.action_grid = np.linspace(A_MIN, A_MAX, n_actions)

        self.q_net = QNetwork(state_dim, n_actions)
        self.target_net = QNetwork(state_dim, n_actions)
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
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
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

        return loss.item()

    def train(self, env, n_episodes=3000, eval_interval=300):
        eval_returns = []

        for ep in range(n_episodes):
            state = env.reset()

            while True:
                action_idx = self.get_action(state)
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)

                self.store(state, action_idx, reward, next_state, float(done))
                self.train_step()

                state = next_state
                if done:
                    break

            if (ep + 1) % eval_interval == 0:
                eval_ret = self.evaluate(ExtendedWindFarmEnv(n_dims=self.state_dim, seed=99), n_episodes=20)
                eval_returns.append((ep + 1, eval_ret))

        return eval_returns

    def evaluate(self, env, n_episodes=100):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            ep_return = 0
            while True:
                action_idx = self.get_action(state, greedy=True)
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                state = next_state
                if done:
                    break
            total_return += ep_return
        return total_return / n_episodes

    def param_count(self):
        return sum(p.numel() for p in self.q_net.parameters())


# =============================================================================
# Bilinear Actor-Critic (Du et al. 2021 style)
# =============================================================================
class BilinearAC:
    """Actor-Critic with linear function approximation."""

    def __init__(self, n_dims=3, lr_actor=0.01, lr_critic=0.05, n_actions=11):
        self.n_dims = n_dims
        self.n_actions = n_actions
        self.action_grid = np.linspace(A_MIN, A_MAX, n_actions)

        # Feature dimension: 1 + d + d + d*(d-1)/2 (intercept, linear, squared, cross)
        # For d=3: 1 + 3 + 3 + 3 = 10
        # For d=6: 1 + 6 + 6 + 15 = 28
        self.feature_dim = 1 + n_dims + n_dims + (n_dims * (n_dims - 1)) // 2

        self.theta_v = np.zeros(self.feature_dim)
        self.theta_mu = np.zeros(self.feature_dim)
        self.log_std = 0.0

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Normalization bounds
        self.bounds = [
            (W_MIN, W_MAX),
            (P_MIN, P_MAX),
            (C_MIN, C_MAX),
        ]
        for _ in range(n_dims - 3):
            self.bounds.append((EXTRA_MIN, EXTRA_MAX))

    def features(self, state):
        # Normalize state
        normalized = []
        for i in range(self.n_dims):
            lo, hi = self.bounds[i]
            normalized.append((state[i] - lo) / (hi - lo + 1e-8))
        x = np.array(normalized)

        # Build features: [1, x1, x2, ..., x1^2, x2^2, ..., x1*x2, x1*x3, ...]
        features = [1.0]
        features.extend(x)  # linear
        features.extend(x ** 2)  # squared
        for i in range(self.n_dims):
            for j in range(i + 1, self.n_dims):
                features.append(x[i] * x[j])  # cross terms
        return np.array(features)

    def get_value(self, state):
        phi = self.features(state)
        return np.dot(self.theta_v, phi)

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

        score_std = ((action_raw - mu)**2 - std**2) / (std**2 + 1e-8)
        self.log_std += self.lr_actor * td_error * score_std * 0.1
        self.log_std = np.clip(self.log_std, -2, 1)

    def train(self, env, n_episodes=3000, eval_interval=300):
        eval_returns = []

        for ep in range(n_episodes):
            state = env.reset()

            while True:
                action_info = self.get_action(state)
                action_idx = action_info[0]
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)

                self.update(state, action_info, reward, next_state, done)

                state = next_state
                if done:
                    break

            if (ep + 1) % eval_interval == 0:
                eval_ret = self.evaluate(ExtendedWindFarmEnv(n_dims=self.n_dims, seed=99), n_episodes=20)
                eval_returns.append((ep + 1, eval_ret))

        return eval_returns

    def evaluate(self, env, n_episodes=100):
        total_return = 0
        for _ in range(n_episodes):
            state = env.reset()
            ep_return = 0
            while True:
                action_idx = self.get_action(state, greedy=True)
                action = self.action_grid[action_idx]
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                state = next_state
                if done:
                    break
            total_return += ep_return
        return total_return / n_episodes

    def param_count(self):
        return len(self.theta_v) + len(self.theta_mu) + 1


# =============================================================================
# Plotting and Output
# =============================================================================

def plot_dp_scaling(dp_results, save_path='curse_scaling.png'):
    """Plot DP computation time vs dimension (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    dims = sorted(dp_results.keys())
    times = []
    completed = []

    for d in dims:
        result = dp_results[d]
        times.append(result['time'])
        completed.append(result['completed'])

    # Plot completed vs timed out differently
    completed_dims = [d for d, c in zip(dims, completed) if c]
    completed_times = [t for t, c in zip(times, completed) if c]
    timeout_dims = [d for d, c in zip(dims, completed) if not c]
    timeout_times = [t for t, c in zip(times, completed) if not c]

    if completed_dims:
        ax.semilogy(completed_dims, completed_times, 'bo-', markersize=10, linewidth=2, label='DP completed')
    if timeout_dims:
        ax.semilogy(timeout_dims, timeout_times, 'rx', markersize=12, markeredgewidth=3, label='DP timeout')

    # Add exponential reference line
    if len(completed_dims) >= 2:
        d0, t0 = completed_dims[0], completed_times[0]
        d1, t1 = completed_dims[-1], completed_times[-1]
        growth_rate = (np.log(t1) - np.log(t0)) / (d1 - d0) if d1 != d0 else 1
        ref_dims = np.linspace(min(dims), max(dims), 50)
        ref_times = t0 * np.exp(growth_rate * (ref_dims - d0))
        ax.semilogy(ref_dims, ref_times, 'k--', alpha=0.5, label=f'Exponential fit')

    ax.axhline(y=DP_TIMEOUT_MINUTES * 60, color='r', linestyle=':', alpha=0.7, label=f'{DP_TIMEOUT_MINUTES} min timeout')

    ax.set_xlabel('State Dimension (d)', fontsize=12)
    ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax.set_title('Curse of Dimensionality: DP Computation Time', fontsize=14)
    ax.set_xticks(dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_rl_scaling(rl_results, save_path='rl_scaling.png'):
    """Plot RL returns vs dimension for each method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'factored': 'blue', 'dqn': 'green', 'bilinear': 'red'}
    labels = {'factored': 'Factored RL', 'dqn': 'DQN', 'bilinear': 'Bilinear AC'}

    dims = sorted(rl_results['factored'].keys())

    for method in ['factored', 'dqn', 'bilinear']:
        means = []
        stds = []
        for d in dims:
            data = rl_results[method][d]
            means.append(data['mean'])
            stds.append(data['std'])

        means = np.array(means)
        stds = np.array(stds)
        n_seeds = max(1, len(rl_results[method][dims[0]]['returns']))
        se = stds / np.sqrt(n_seeds) if n_seeds > 1 else stds * 0  # No error bars with 1 seed

        ax.plot(dims, means, color=colors[method], marker='o', linewidth=2,
                markersize=8, label=labels[method])
        ax.fill_between(dims, means - 1.96*se, means + 1.96*se,
                        color=colors[method], alpha=0.2)

    ax.set_xlabel('State Dimension (d)', fontsize=12)
    ax.set_ylabel('Average Return ($/day)', fontsize=12)
    ax.set_title('RL Methods Scale Gracefully with Dimension', fontsize=14)
    ax.set_xticks(dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_sample_efficiency(results, save_path='sample_efficiency.png'):
    """Plot learning curves with confidence bands for d=3."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'factored': 'blue', 'dqn': 'green', 'bilinear': 'red'}
    labels = {'factored': 'Factored RL', 'dqn': 'Deep Q-Network', 'bilinear': 'Bilinear AC'}

    for method, runs in results.items():
        if not runs:
            continue
        episodes = [r[0] for r in runs[0]]

        returns_matrix = np.array([[r[1] for r in run] for run in runs])
        mean_returns = returns_matrix.mean(axis=0)
        std_returns = returns_matrix.std(axis=0)

        ax.plot(episodes, mean_returns, color=colors[method], linewidth=2, label=labels[method])
        ax.fill_between(episodes,
                        mean_returns - 1.96 * std_returns / np.sqrt(len(runs)),
                        mean_returns + 1.96 * std_returns / np.sqrt(len(runs)),
                        color=colors[method], alpha=0.2)

    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Return ($/day)', fontsize=12)
    ax.set_title('Sample Efficiency: Learning Curves (d=3)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_combined_scaling(dp_results, rl_results, save_path='scaling_combined.png'):
    """Create two-panel figure: DP time vs dimension (left), RL returns vs dimension (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dims = sorted(rl_results['factored'].keys())

    # Left panel: DP computation time (log scale)
    times = []
    completed = []
    for d in dims:
        result = dp_results[d]
        times.append(result['time'])
        completed.append(result['completed'])

    completed_dims = [d for d, c in zip(dims, completed) if c]
    completed_times = [t for t, c in zip(times, completed) if c]
    timeout_dims = [d for d, c in zip(dims, completed) if not c]
    timeout_times = [t for t, c in zip(times, completed) if not c]

    if completed_dims:
        ax1.semilogy(completed_dims, completed_times, 'bo-', markersize=10, linewidth=2, label='DP completed')
    if timeout_dims:
        ax1.semilogy(timeout_dims, timeout_times, 'rx', markersize=12, markeredgewidth=3, label='DP timeout')

    # Exponential reference line from completed points
    if len(completed_dims) >= 2:
        d0, t0 = completed_dims[0], completed_times[0]
        d1, t1 = completed_dims[-1], completed_times[-1]
        growth_rate = (np.log(t1) - np.log(t0)) / (d1 - d0) if d1 != d0 else 1
        ref_dims = np.linspace(min(dims), max(dims), 50)
        ref_times = t0 * np.exp(growth_rate * (ref_dims - d0))
        ax1.semilogy(ref_dims, ref_times, 'k--', alpha=0.5, label='Exponential fit')

    ax1.axhline(y=DP_TIMEOUT_MINUTES * 60, color='r', linestyle=':', alpha=0.7, label=f'{DP_TIMEOUT_MINUTES} min timeout')
    ax1.set_xlabel('State Dimension (d)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('(a) DP Computation Time', fontsize=13)
    ax1.set_xticks(dims)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right panel: RL returns vs dimension
    colors = {'factored': 'blue', 'dqn': 'green', 'bilinear': 'red'}
    labels = {'factored': 'Factored RL', 'dqn': 'DQN', 'bilinear': 'Bilinear AC'}

    for method in ['factored', 'dqn', 'bilinear']:
        means = []
        stds = []
        for d in dims:
            data = rl_results[method][d]
            means.append(data['mean'])
            stds.append(data['std'])

        means = np.array(means)
        stds = np.array(stds)
        n_seeds = max(1, len(rl_results[method][dims[0]]['returns']))
        se = stds / np.sqrt(n_seeds) if n_seeds > 1 else stds * 0

        ax2.plot(dims, means, color=colors[method], marker='o', linewidth=2,
                markersize=8, label=labels[method])
        if n_seeds > 1:
            ax2.fill_between(dims, means - 1.96*se, means + 1.96*se,
                            color=colors[method], alpha=0.2)

    # Add DP baseline where available
    dp_dims = [d for d in dims if dp_results[d]['completed']]
    dp_returns = [dp_results[d]['return'] for d in dp_dims]
    if dp_dims:
        ax2.plot(dp_dims, dp_returns, 'ks--', markersize=10, linewidth=2, label='Tabular DP')

    ax2.set_xlabel('State Dimension (d)', fontsize=12)
    ax2.set_ylabel('Average Return ($/day)', fontsize=12)
    ax2.set_title('(b) Policy Performance', fontsize=13)
    ax2.set_xticks(dims)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_performance(dp_results, rl_results, save_path='performance.png'):
    """Single figure showing policy returns vs dimension for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = sorted(rl_results['factored'].keys())

    # DP returns (where available)
    dp_dims = [d for d in dims if dp_results[d]['completed']]
    dp_returns = [dp_results[d]['return'] for d in dp_dims]
    if dp_dims:
        ax.plot(dp_dims, dp_returns, 'ko-', markersize=12, linewidth=2.5,
                label='Tabular DP', zorder=5)

    # RL returns
    colors = {'factored': '#1f77b4', 'dqn': '#2ca02c', 'bilinear': '#d62728'}
    markers = {'factored': 's', 'dqn': '^', 'bilinear': 'D'}
    labels = {'factored': 'Factored RL', 'dqn': 'DQN', 'bilinear': 'Bilinear AC'}

    for method in ['factored', 'dqn', 'bilinear']:
        means = [rl_results[method][d]['mean'] for d in dims]
        ax.plot(dims, means, color=colors[method], marker=markers[method],
                markersize=9, linewidth=2, label=labels[method])

    ax.set_xlabel('State Dimension (d)', fontsize=13)
    ax.set_ylabel('Average Return ($/day)', fontsize=13)
    ax.set_title('Policy Performance: DP vs RL Methods', fontsize=14)
    ax.set_xticks(dims)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Set y-axis to start near minimum value for better visibility
    all_returns = dp_returns + [rl_results[m][d]['mean'] for m in ['factored', 'dqn', 'bilinear'] for d in dims]
    ax.set_ylim(bottom=min(all_returns) - 20, top=max(all_returns) + 20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_computation_times(dp_results, rl_results, save_path='computation_times.png'):
    """Single figure showing computation time vs dimension for all methods (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = sorted(rl_results['factored'].keys())

    # DP times
    dp_completed_dims = [d for d in dims if dp_results[d]['completed']]
    dp_completed_times = [dp_results[d]['time'] for d in dp_completed_dims]
    dp_timeout_dims = [d for d in dims if not dp_results[d]['completed']]
    dp_timeout_times = [dp_results[d]['time'] for d in dp_timeout_dims]

    if dp_completed_dims:
        ax.semilogy(dp_completed_dims, dp_completed_times, 'ko-', markersize=12,
                    linewidth=2.5, label='Tabular DP', zorder=5)
    if dp_timeout_dims:
        ax.semilogy(dp_timeout_dims, dp_timeout_times, 'kx', markersize=14,
                    markeredgewidth=3, label='DP (timeout)', zorder=5)

    # Exponential extrapolation for DP
    if len(dp_completed_dims) >= 2:
        d0, t0 = dp_completed_dims[0], dp_completed_times[0]
        d1, t1 = dp_completed_dims[-1], dp_completed_times[-1]
        growth_rate = (np.log(t1) - np.log(t0)) / (d1 - d0)
        ref_dims = np.linspace(min(dims), max(dims), 50)
        ref_times = t0 * np.exp(growth_rate * (ref_dims - d0))
        ax.semilogy(ref_dims, ref_times, 'k:', alpha=0.4, linewidth=1.5, label='DP extrapolation')

    # RL times
    colors = {'factored': '#1f77b4', 'dqn': '#2ca02c', 'bilinear': '#d62728'}
    markers = {'factored': 's', 'dqn': '^', 'bilinear': 'D'}
    labels = {'factored': 'Factored RL', 'dqn': 'DQN', 'bilinear': 'Bilinear AC'}

    for method in ['factored', 'dqn', 'bilinear']:
        times = [rl_results[method][d]['time'] for d in dims]
        ax.semilogy(dims, times, color=colors[method], marker=markers[method],
                    markersize=9, linewidth=2, label=labels[method])

    # Timeout line
    ax.axhline(y=DP_TIMEOUT_MINUTES * 60, color='red', linestyle='--',
               alpha=0.6, linewidth=1.5, label=f'{DP_TIMEOUT_MINUTES} min timeout')

    ax.set_xlabel('State Dimension (d)', fontsize=13)
    ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=13)
    ax.set_title('Computation Time: DP vs RL Methods', fontsize=14)
    ax.set_xticks(dims)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=1)  # Start from 1 second

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_results_table(dp_results, rl_results, save_path='results.tex'):
    """Generate LaTeX results table with DP and RL methods across dimensions."""

    dims = sorted(rl_results['factored'].keys())

    # Header
    table = r"""\begin{tabular}{l""" + "c" * len(dims) + r"""}
\toprule
Method & """ + " & ".join([f"$d={d}$" for d in dims]) + r""" \\
\midrule
"""

    # DP row
    dp_row = "Tabular DP"
    for d in dims:
        if d in dp_results:
            result = dp_results[d]
            if result['completed']:
                time_str = f"({result['time']:.0f}s)"
                dp_row += f" & {result['return']:.0f} {time_str}"
            else:
                dp_row += f" & TIMEOUT"
        else:
            dp_row += " & --"
    dp_row += r" \\" + "\n"
    table += dp_row

    # RL rows
    for method, label in [('factored', 'Factored RL'), ('dqn', 'DQN'), ('bilinear', 'Bilinear AC')]:
        row = label
        for d in dims:
            data = rl_results[method][d]
            row += f" & {data['mean']:.0f} $\\pm$ {data['std']:.0f}"
        row += r" \\" + "\n"
        table += row

    table += r"""\bottomrule
\end{tabular}"""

    with open(save_path, 'w') as f:
        f.write(table)
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("WIND FARM STORAGE CONTROL: CURSE OF DIMENSIONALITY STUDY")
    print("="*70)
    print(f"Horizon: {HORIZON}, Discount: {GAMMA}")
    print(f"Base state: wind in [{W_MIN},{W_MAX}], price in [{P_MIN},{P_MAX}], SoC in [{C_MIN},{C_MAX}]")
    print(f"Action: charge/discharge in [{A_MIN},{A_MAX}]")
    print(f"DP timeout: {DP_TIMEOUT_MINUTES} minutes")
    print()

    # Experiment parameters
    dims_to_test = [3, 4, 5, 6]
    n_episodes = 3000
    n_seeds = 1  # Single seed for quick iteration
    eval_interval = 300
    dp_n_bins = 7  # Reduced for tractability (7^3 = 343 states at d=3, 7^4 = 2401 at d=4)
    dp_n_mc = 20  # Reduced MC samples

    print(f"Dimensions to test: {dims_to_test}")
    print(f"RL: {n_episodes} episodes, {n_seeds} seeds, eval every {eval_interval}")
    print(f"DP: n_bins={dp_n_bins}, n_mc_samples={dp_n_mc}")
    print()

    # ==============
    # EXPERIMENT 1: DP Scaling Study
    # ==============
    print("="*70)
    print("EXPERIMENT 1: DP SCALING STUDY")
    print("="*70)

    dp_results = {}

    for d in dims_to_test:
        n_states = dp_n_bins ** d
        print(f"\nDimension d={d}: {n_states:,} states")

        dp = TabularDP(n_dims=d, n_bins=dp_n_bins, n_mc_samples=dp_n_mc)
        completed, elapsed = dp.solve(timeout_minutes=DP_TIMEOUT_MINUTES, verbose=True)

        if completed:
            ret = dp.evaluate(n_episodes=50, seed=99)
            dp_results[d] = {
                'completed': True,
                'time': elapsed,
                'return': ret,
                'n_states': n_states
            }
            print(f"  Return: {ret:.1f}")
        else:
            dp_results[d] = {
                'completed': False,
                'time': elapsed,
                'return': None,
                'n_states': n_states
            }

    print("\n" + "-"*50)
    print("DP Results Summary:")
    print(f"{'Dim':<6} {'States':<15} {'Time (s)':<12} {'Return':<12}")
    print("-"*50)
    for d in dims_to_test:
        r = dp_results[d]
        states_str = f"{r['n_states']:,}"
        time_str = f"{r['time']:.1f}"
        ret_str = f"{r['return']:.1f}" if r['completed'] else "TIMEOUT"
        print(f"{d:<6} {states_str:<15} {time_str:<12} {ret_str:<12}")

    # ==============
    # EXPERIMENT 2: RL Scaling Study
    # ==============
    print()
    print("="*70)
    print("EXPERIMENT 2: RL SCALING STUDY")
    print("="*70)

    rl_results = {'factored': {}, 'dqn': {}, 'bilinear': {}}
    learning_curves_d3 = {'factored': [], 'dqn': [], 'bilinear': []}

    for d in dims_to_test:
        print(f"\n--- Dimension d={d} ---")

        for method in ['factored', 'dqn', 'bilinear']:
            rl_results[method][d] = {'returns': [], 'times': []}

        for seed in tqdm(range(n_seeds), desc=f"d={d} seeds"):
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Factored Q-learning
            env = ExtendedWindFarmEnv(n_dims=d, seed=seed)
            agent = FactoredQL(n_dims=d, n_bins=30, alpha=0.2, epsilon=0.2, reward_dims=3)
            t_start = time.time()
            eval_rets = agent.train(env, n_episodes=n_episodes, eval_interval=eval_interval)
            t_elapsed = time.time() - t_start
            final_return = agent.evaluate(ExtendedWindFarmEnv(n_dims=d, seed=99), n_episodes=50)
            rl_results['factored'][d]['returns'].append(final_return)
            rl_results['factored'][d]['times'].append(t_elapsed)
            if d == 3:
                learning_curves_d3['factored'].append(eval_rets)

            # DQN
            env = ExtendedWindFarmEnv(n_dims=d, seed=seed)
            agent = DQNAgent(state_dim=d, n_actions=11, lr=5e-4, epsilon_start=0.3)
            t_start = time.time()
            eval_rets = agent.train(env, n_episodes=n_episodes, eval_interval=eval_interval)
            t_elapsed = time.time() - t_start
            final_return = agent.evaluate(ExtendedWindFarmEnv(n_dims=d, seed=99), n_episodes=50)
            rl_results['dqn'][d]['returns'].append(final_return)
            rl_results['dqn'][d]['times'].append(t_elapsed)
            if d == 3:
                learning_curves_d3['dqn'].append(eval_rets)

            # Bilinear AC
            env = ExtendedWindFarmEnv(n_dims=d, seed=seed)
            agent = BilinearAC(n_dims=d, lr_actor=0.02, lr_critic=0.08)
            t_start = time.time()
            eval_rets = agent.train(env, n_episodes=n_episodes, eval_interval=eval_interval)
            t_elapsed = time.time() - t_start
            final_return = agent.evaluate(ExtendedWindFarmEnv(n_dims=d, seed=99), n_episodes=50)
            rl_results['bilinear'][d]['returns'].append(final_return)
            rl_results['bilinear'][d]['times'].append(t_elapsed)
            if d == 3:
                learning_curves_d3['bilinear'].append(eval_rets)

        # Compute stats
        for method in ['factored', 'dqn', 'bilinear']:
            returns = rl_results[method][d]['returns']
            times = rl_results[method][d]['times']
            rl_results[method][d]['mean'] = np.mean(returns)
            rl_results[method][d]['std'] = np.std(returns)
            rl_results[method][d]['time'] = np.mean(times)

    # Print RL results
    print("\n" + "-"*70)
    print("RL Results Summary (Final Returns):")
    print(f"{'Method':<15}", end="")
    for d in dims_to_test:
        print(f"{'d=' + str(d):<18}", end="")
    print()
    print("-"*70)
    for method in ['factored', 'dqn', 'bilinear']:
        print(f"{method:<15}", end="")
        for d in dims_to_test:
            data = rl_results[method][d]
            print(f"{data['mean']:.1f} +/- {data['std']:.1f}     ", end="")
        print()

    # Print RL training times
    print("\n" + "-"*70)
    print("RL Training Times (seconds):")
    print(f"{'Method':<15}", end="")
    for d in dims_to_test:
        print(f"{'d=' + str(d):<12}", end="")
    print()
    print("-"*70)
    for method in ['factored', 'dqn', 'bilinear']:
        print(f"{method:<15}", end="")
        for d in dims_to_test:
            data = rl_results[method][d]
            print(f"{data['time']:.1f}s       ", end="")
        print()

    # ==============
    # Generate outputs
    # ==============
    print()
    print("="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # Primary outputs: computation time and performance figures
    plot_computation_times(dp_results, rl_results)
    plot_performance(dp_results, rl_results)

    # Two-panel figure (legacy)
    plot_combined_scaling(dp_results, rl_results)

    # Other legacy outputs
    plot_dp_scaling(dp_results)
    plot_rl_scaling(rl_results)
    plot_sample_efficiency(learning_curves_d3)
    generate_results_table(dp_results, rl_results)

    print()
    print("Output files:")
    print("  - computation_times.png (Single figure: all methods computation time)")
    print("  - performance.png (Single figure: all methods policy returns)")
    print("  - scaling_combined.png (Two-panel: DP time + RL returns)")
    print("  - curse_scaling.png (DP computation time vs dimension)")
    print("  - rl_scaling.png (RL returns vs dimension)")
    print("  - sample_efficiency.png (Learning curves at d=3)")
    print("  - results.tex (Combined results table)")


if __name__ == '__main__':
    main()
