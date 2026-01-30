# Shared Abstractions for Economic Benchmark Environments
# Chapter 3 -- Economic Benchmarks
# Provides base class, neural network, replay buffer, and common algorithms
# used across all benchmark scripts.

import sys
import time
import random
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Set, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metrics dataclasses for tracking training/convergence statistics
# ---------------------------------------------------------------------------

@dataclass
class DQNMetrics:
    """Metrics tracked during DQN training."""
    episode_rewards: List[float] = field(default_factory=list)
    eval_checkpoints: List[float] = field(default_factory=list)
    checkpoint_episodes: List[int] = field(default_factory=list)
    total_transitions: int = 0
    total_gradient_updates: int = 0
    states_visited: Set = field(default_factory=set)
    wall_time: float = 0.0


@dataclass
class VIMetrics:
    """Metrics tracked during Value Iteration."""
    iterations: int = 0
    final_residual: float = 0.0
    wall_time: float = 0.0


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

def run_value_iteration(env, gamma=0.95, tol=1e-6, max_iter=500, show_progress=True):
    """Tabular value iteration for any EconBenchmark.
    Returns (V_array, policy_array, VIMetrics).

    Args:
        show_progress: If True, display tqdm progress bar with residual
    """
    n_states = env.num_states
    n_actions = env.num_actions
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    t0 = time.time()
    iterations = 0
    final_residual = 0.0

    pbar = tqdm(range(max_iter), desc=f"VI ({n_states} states)", disable=not show_progress,
                ncols=100, leave=False)

    for iteration in pbar:
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
        iterations = iteration + 1
        final_residual = diff

        if show_progress:
            pbar.set_postfix({'residual': f'{diff:.2e}'})

        if diff < tol:
            break

    pbar.close()
    wall_time = time.time() - t0

    metrics = VIMetrics(
        iterations=iterations,
        final_residual=final_residual,
        wall_time=wall_time
    )
    return V, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 2: DQN training (generic, works with any EconBenchmark)
# ---------------------------------------------------------------------------

def run_dqn(env, gamma=0.95, num_episodes=5000, episode_horizon=50,
            seed=42, replay_size=10_000, batch_size=128, lr=1e-3,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_frac=0.6,
            target_update_freq=50, hidden1=128, hidden2=64,
            eval_freq=100, eval_episodes=20, show_progress=True, desc=None):
    """Train DQN on an EconBenchmark environment.
    Returns (q_net, DQNMetrics).

    Args:
        show_progress: If True, display tqdm progress bar with eval scores
        desc: Description for progress bar (e.g., "seed=42")
    """
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

    # Metrics tracking
    episode_rewards = []
    eval_checkpoints = []
    checkpoint_episodes = []
    total_transitions = 0
    total_gradient_updates = 0
    states_visited = set()
    last_eval = None

    t0 = time.time()

    # Progress bar
    pbar_desc = desc if desc else f"DQN(seed={seed})"
    pbar = tqdm(range(num_episodes), desc=pbar_desc, disable=not show_progress,
                ncols=100, leave=False)

    for ep in pbar:
        s = env.reset()
        s_feat = env.state_to_features(s)
        ep_reward = 0.0

        for t in range(episode_horizon):
            # Track state visitation
            states_visited.add(env.state_to_index(s))

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

            ep_reward += r
            total_transitions += 1

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
                total_gradient_updates += 1

            if done:
                break

        episode_rewards.append(ep_reward)
        eps = max(epsilon_end, eps - eps_step)

        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic evaluation checkpoint
        if (ep + 1) % eval_freq == 0:
            eval_reward = evaluate_dqn_policy(env, q_net, n_episodes=eval_episodes,
                                               horizon=episode_horizon)
            eval_checkpoints.append(eval_reward)
            checkpoint_episodes.append(ep + 1)
            last_eval = eval_reward

            # Update progress bar with eval score
            if show_progress:
                pbar.set_postfix({'eval': f'{eval_reward:.2f}', 'eps': f'{eps:.2f}'})

    pbar.close()
    wall_time = time.time() - t0

    metrics = DQNMetrics(
        episode_rewards=episode_rewards,
        eval_checkpoints=eval_checkpoints,
        checkpoint_episodes=checkpoint_episodes,
        total_transitions=total_transitions,
        total_gradient_updates=total_gradient_updates,
        states_visited=states_visited,
        wall_time=wall_time
    )
    return q_net, metrics


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


# ---------------------------------------------------------------------------
# Utility functions for metrics computation
# ---------------------------------------------------------------------------

def compute_policy_entropy(q_net, env, n_samples=1000, temperature=1.0):
    """Compute average entropy of action distribution across sampled states.

    Uses softmax over Q-values to define action probabilities.
    Returns entropy in [0, log(num_actions)].
    """
    q_net.eval()
    entropies = []
    for _ in range(n_samples):
        s = env.reset()
        s_feat = env.state_to_features(s)
        with torch.no_grad():
            q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0)).squeeze(0)
            probs = torch.softmax(q_vals / temperature, dim=0).numpy()
            probs = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
    return np.mean(entropies)


def compute_dp_policy_entropy(policy, env, n_samples=1000):
    """Compute entropy of DP policy (deterministic, so entropy is 0).

    For comparison purposes, we return 0 for deterministic policies.
    If policy has stochasticity, this should be extended.
    """
    return 0.0


def compute_policy_agreement(q_net, dp_policy, env, n_samples=1000):
    """Fraction of states where DQN greedy action matches DP policy."""
    q_net.eval()
    agreements = 0
    for _ in range(n_samples):
        s = env.reset()
        si = env.state_to_index(s)
        dp_action = dp_policy[si]

        s_feat = env.state_to_features(s)
        with torch.no_grad():
            q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0))
            dqn_action = q_vals.argmax(dim=1).item()

        if dqn_action == dp_action:
            agreements += 1

    return agreements / n_samples


def episodes_to_threshold(rewards, threshold_frac=0.95):
    """First episode where reward exceeds threshold_frac * max reward.

    Returns episode number (1-indexed) or None if never reached.
    """
    if len(rewards) == 0:
        return None
    max_reward = max(rewards)
    threshold = threshold_frac * max_reward
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i + 1
    return None


def state_coverage(states_visited, num_states):
    """Fraction of state space visited during training."""
    return len(states_visited) / num_states


# ---------------------------------------------------------------------------
# Q-value error computation
# ---------------------------------------------------------------------------

def compute_q_error(q_net, V_dp, env, n_samples=1000, gamma=0.95):
    """Compute MSE between DQN Q-values and DP V-values.

    For each sampled state, computes the DQN value as max_a Q(s,a)
    and compares to the DP value V(s). Returns RMSE.

    Args:
        q_net: Trained Q-network
        V_dp: Value function array from DP (indexed by state)
        env: Environment implementing EconBenchmark
        n_samples: Number of states to sample
        gamma: Discount factor (unused but kept for API consistency)

    Returns:
        RMSE between DQN max-Q and DP V-values
    """
    q_net.eval()
    errors = []

    for _ in range(n_samples):
        s = env.reset()
        si = env.state_to_index(s)
        v_dp = V_dp[si]

        s_feat = env.state_to_features(s)
        with torch.no_grad():
            q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0))
            v_dqn = q_vals.max(dim=1)[0].item()

        errors.append((v_dqn - v_dp) ** 2)

    return np.sqrt(np.mean(errors))


# ---------------------------------------------------------------------------
# Decomposed evaluation
# ---------------------------------------------------------------------------

@dataclass
class DecomposedResult:
    """Results from decomposed evaluation."""
    total_reward: float
    component_rewards: Dict[str, float]
    per_episode_totals: List[float]
    per_episode_components: List[Dict[str, float]]


def evaluate_with_decomposition(env, policy_fn, n_episodes, horizon, decompose_fn):
    """Evaluate policy returning both total and component rewards.

    Args:
        env: Environment implementing EconBenchmark
        policy_fn: Callable taking state and returning action
        n_episodes: Number of episodes to run
        horizon: Maximum steps per episode
        decompose_fn: Callable taking (state, action, reward, next_state, env)
                      and returning Dict[str, float] of component values

    Returns:
        DecomposedResult with aggregated statistics
    """
    per_episode_totals = []
    per_episode_components = []

    for _ in range(n_episodes):
        s = env.reset()
        ep_total = 0.0
        ep_components = {}

        for t in range(horizon):
            a = policy_fn(s)
            ns, r, done = env.step(a)

            ep_total += r

            # Get component breakdown
            components = decompose_fn(s, a, r, ns, env)
            for key, val in components.items():
                ep_components[key] = ep_components.get(key, 0.0) + val

            s = ns
            if done:
                break

        per_episode_totals.append(ep_total)
        per_episode_components.append(ep_components)

    # Aggregate
    total_reward = np.mean(per_episode_totals)
    component_rewards = {}
    if per_episode_components:
        all_keys = per_episode_components[0].keys()
        for key in all_keys:
            vals = [ep[key] for ep in per_episode_components]
            component_rewards[key] = np.mean(vals)

    return DecomposedResult(
        total_reward=total_reward,
        component_rewards=component_rewards,
        per_episode_totals=per_episode_totals,
        per_episode_components=per_episode_components
    )


# ---------------------------------------------------------------------------
# Stdout capture context manager
# ---------------------------------------------------------------------------

class TeeOutput:
    """Tee stdout to both console and file."""

    def __init__(self, filepath, original_stdout):
        self.file = open(filepath, 'w')
        self.stdout = original_stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


@contextmanager
def capture_stdout(filepath):
    """Context manager to tee stdout to both console and file.

    Usage:
        with capture_stdout('output.txt'):
            print("This goes to both console and file")
    """
    original_stdout = sys.stdout
    tee = TeeOutput(filepath, original_stdout)
    sys.stdout = tee
    try:
        yield tee
    finally:
        sys.stdout = original_stdout
        tee.close()


# ---------------------------------------------------------------------------
# Standardized LaTeX table generators
# ---------------------------------------------------------------------------

def make_scaling_table(results: List[Dict[str, Any]], complexity_name: str,
                       domain_name: str) -> str:
    """Generate LaTeX table with scaling results.

    Args:
        results: List of dicts, each with keys:
            - complexity: int (complexity parameter value)
            - states: int (state space size)
            - dp_reward: float or None
            - dp_time: float or None
            - dqn_reward_mean: float
            - dqn_reward_std: float
            - dqn_time_mean: float
            - q_error: float or None
            - agreement: float or None (0-1)
        complexity_name: Name of complexity parameter (e.g., "K", "C", "N")
        domain_name: Name of the domain for caption

    Returns:
        LaTeX table as string
    """
    lines = []
    lines.append(r'\begin{tabular}{rrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(f'{complexity_name} & States & DP Reward & DP Time & '
                 r'DQN Reward & DQN Time & Q-Error & Agreement \\')
    lines.append(r'\midrule')

    for row in results:
        comp = row['complexity']
        states = row['states']

        if row.get('dp_reward') is not None:
            dp_r = f"${row['dp_reward']:.2f}$"
            dp_t = f"${row['dp_time']:.2f}$"
        else:
            dp_r = "---"
            dp_t = "---"

        dqn_r = f"${row['dqn_reward_mean']:.2f} \\pm {row['dqn_reward_std']:.2f}$"
        dqn_t = f"${row['dqn_time_mean']:.1f}$"

        if row.get('q_error') is not None:
            q_err = f"${row['q_error']:.3f}$"
        else:
            q_err = "---"

        if row.get('agreement') is not None:
            agr = f"{row['agreement']:.1%}"
        else:
            agr = "---"

        lines.append(f"{comp} & {states:,} & {dp_r} & {dp_t} & "
                     f"{dqn_r} & {dqn_t} & {q_err} & {agr} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    return '\n'.join(lines)


def make_decomposition_table(decomposition: Dict[str, float], domain_name: str) -> str:
    """Generate LaTeX table showing reward/cost decomposition.

    Args:
        decomposition: Dict mapping component name to mean value
        domain_name: Name of domain for caption

    Returns:
        LaTeX table as string
    """
    lines = []
    lines.append(r'\begin{tabular}{lr}')
    lines.append(r'\toprule')
    lines.append(r'Component & Mean Value \\')
    lines.append(r'\midrule')

    for component, value in decomposition.items():
        # Format component name nicely
        nice_name = component.replace('_', ' ').title()
        lines.append(f"{nice_name} & ${value:.3f}$ \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# DP feasibility threshold
# ---------------------------------------------------------------------------

DP_FEASIBLE_THRESHOLD = 65536  # Skip DP above this many states
