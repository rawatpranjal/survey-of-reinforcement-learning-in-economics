# Shared Abstractions for Economic Benchmark Environments
# Chapter 3 -- Economic Benchmarks
# Provides base class, neural network, replay buffer, and common algorithms
# used across all benchmark scripts.

import time
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Base class for economic benchmark environments
# ---------------------------------------------------------------------------

class EconBenchmark(ABC):
    """Base class for economic benchmark environments.

    Subclasses must implement the core MDP interface (reset, step) and
    the DP interface (enumerate_states, transition_distribution, expected_reward).
    The NN interface (state_to_features) enables DQN training.
    """

    # Metadata (override in subclasses)
    complexity_param: int = 1
    dp_feasible: bool = True

    @abstractmethod
    def reset(self):
        """Reset environment and return initial state (tuple)."""
        ...

    @abstractmethod
    def step(self, action):
        """Take action, return (next_state, reward, done)."""
        ...

    @property
    @abstractmethod
    def num_states(self):
        """Total number of states."""
        ...

    @property
    @abstractmethod
    def num_actions(self):
        """Total number of actions."""
        ...

    @abstractmethod
    def state_to_index(self, state):
        """Map state tuple to integer index."""
        ...

    @abstractmethod
    def index_to_state(self, idx):
        """Map integer index to state tuple."""
        ...

    @abstractmethod
    def state_to_features(self, state):
        """Convert state tuple to normalized float array for neural network input."""
        ...

    @abstractmethod
    def transition_distribution(self, state, action):
        """Return list of (next_state, probability) pairs for DP."""
        ...

    @abstractmethod
    def expected_reward(self, state, action):
        """Return expected immediate reward E[r | s, a] for DP."""
        ...

    @property
    def feature_dim(self):
        """Dimension of feature vector returned by state_to_features."""
        dummy = self.reset()
        return len(self.state_to_features(dummy))


# ---------------------------------------------------------------------------
# Neural network and replay buffer (shared across all benchmarks)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Two hidden-layer MLP for Q-value approximation (128-64 ReLU)."""

    def __init__(self, state_dim, num_actions, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity replay buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_feat, action_idx, reward, next_state_feat, done):
        self.buffer.append((state_feat, action_idx, reward, next_state_feat, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Algorithm 1: Value Iteration (generic, works with any EconBenchmark)
# ---------------------------------------------------------------------------

def run_value_iteration(env, gamma=0.95, tol=1e-6, max_iter=500):
    """Tabular value iteration for any EconBenchmark.
    Returns (V_array, policy_array, wall_time)."""
    n_states = env.num_states
    n_actions = env.num_actions
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    t0 = time.time()
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        for si in range(n_states):
            s = env.index_to_state(si)
            best_val = -np.inf
            best_a = 0
            for a in range(n_actions):
                r = env.expected_reward(s, a)
                trans = env.transition_distribution(s, a)
                val = r
                for ns, prob in trans:
                    nsi = env.state_to_index(ns)
                    val += gamma * prob * V[nsi]
                if val > best_val:
                    best_val = val
                    best_a = a
            V_new[si] = best_val
            policy[si] = best_a
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break
    wall_time = time.time() - t0
    return V, policy, wall_time


# ---------------------------------------------------------------------------
# Algorithm 2: DQN training (generic, works with any EconBenchmark)
# ---------------------------------------------------------------------------

def run_dqn(env, gamma=0.95, num_episodes=5000, episode_horizon=50,
            seed=42, replay_size=10_000, batch_size=128, lr=1e-3,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_frac=0.6,
            target_update_freq=50, hidden1=128, hidden2=64):
    """Train DQN on an EconBenchmark environment.
    Returns (q_net, wall_time)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.feature_dim
    n_actions = env.num_actions

    q_net = QNetwork(state_dim, n_actions, hidden1, hidden2)
    target_net = QNetwork(state_dim, n_actions, hidden1, hidden2)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_size)

    eps = epsilon_start
    eps_step = (epsilon_start - epsilon_end) / max(1, epsilon_decay_frac * num_episodes)

    t0 = time.time()
    for ep in range(num_episodes):
        s = env.reset()
        s_feat = env.state_to_features(s)
        for t in range(episode_horizon):
            # Epsilon-greedy
            if random.random() < eps:
                a = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0))
                    a = q_vals.argmax(dim=1).item()

            ns, r, done = env.step(a)
            ns_feat = env.state_to_features(ns)
            buffer.push(s_feat, a, r, ns_feat, float(done))
            s = ns
            s_feat = ns_feat

            # Train
            if len(buffer) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
                q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_b).max(dim=1)[0]
                    target = r_b + gamma * next_q * (1.0 - d_b)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        eps = max(epsilon_end, eps - eps_step)
        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    wall_time = time.time() - t0
    return q_net, wall_time


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_dp_policy(env, policy, gamma=0.95, n_episodes=200, horizon=50):
    """Evaluate a tabular DP policy via simulation. Returns mean episode reward."""
    rewards = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r = 0.0
        for t in range(horizon):
            si = env.state_to_index(s)
            a = policy[si]
            ns, r, done = env.step(a)
            ep_r += r
            s = ns
            if done:
                break
        rewards.append(ep_r)
    return np.mean(rewards)


def evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=50):
    """Evaluate a trained DQN policy via simulation. Returns mean episode reward."""
    q_net.eval()
    rewards = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r = 0.0
        for t in range(horizon):
            s_feat = env.state_to_features(s)
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0))
                a = q_vals.argmax(dim=1).item()
            ns, r, done = env.step(a)
            ep_r += r
            s = ns
            if done:
                break
        rewards.append(ep_r)
    return np.mean(rewards)


def evaluate_heuristic(env, heuristic_fn, n_episodes=200, horizon=50):
    """Evaluate a heuristic policy function. heuristic_fn(state) -> action.
    Returns mean episode reward."""
    rewards = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r = 0.0
        for t in range(horizon):
            a = heuristic_fn(s)
            ns, r, done = env.step(a)
            ep_r += r
            s = ns
            if done:
                break
        rewards.append(ep_r)
    return np.mean(rewards)
