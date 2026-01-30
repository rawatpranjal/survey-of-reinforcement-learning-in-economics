# Gridworld Algorithm Implementations
# Chapter 3 -- Comprehensive Gridworld Algorithm Comparison
# Provides unified implementations of classical RL algorithms for tabular gridworld.

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np


# ---------------------------------------------------------------------------
# Metrics dataclass for algorithm results
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmMetrics:
    """Metrics tracked during algorithm training/execution."""
    # Core metrics
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    # Checkpointed evaluations
    eval_returns: List[float] = field(default_factory=list)
    eval_steps: List[float] = field(default_factory=list)
    checkpoint_episodes: List[int] = field(default_factory=list)

    # Value function tracking
    value_errors: List[float] = field(default_factory=list)
    policy_agreements: List[float] = field(default_factory=list)

    # Cumulative regret
    cumulative_regret: List[float] = field(default_factory=list)

    # State coverage
    states_visited: Set = field(default_factory=set)

    # Timing
    wall_time: float = 0.0

    # Planning-specific
    iterations: int = 0
    final_residual: float = 0.0


# ---------------------------------------------------------------------------
# Gridworld Environment (standalone, no external dependencies)
# ---------------------------------------------------------------------------

class GridworldEnv:
    """NxN deterministic gridworld for algorithm comparison.

    States: (row, col) tuples with row, col in [0, N-1]
    Actions: 0=Left, 1=Right, 2=Up, 3=Down, 4=Stay
    Terminal: (N-1, N-1) with reward TERMINAL_REWARD
    All other transitions yield STEP_PENALTY
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['Left', 'Right', 'Up', 'Down', 'Stay']

    def __init__(self, N: int, gamma: float = 0.95,
                 step_penalty: float = -0.1, terminal_reward: float = 10.0):
        self.N = N
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.terminal = (N - 1, N - 1)
        self.initial_state = (0, 0)
        self.state = self.initial_state

        # Precompute state/action space
        self.states = [(r, c) for r in range(N) for c in range(N)]
        self.num_states = N * N
        self.num_actions = len(self.ACTIONS)

    def reset(self) -> Tuple[int, int]:
        """Reset to initial state."""
        self.state = self.initial_state
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take action, return (next_state, reward, done)."""
        if self.state == self.terminal:
            return self.state, 0.0, True

        dr, dc = self.ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.N - 1, r + dr))
        nc = max(0, min(self.N - 1, c + dc))
        self.state = (nr, nc)

        if self.state == self.terminal:
            return self.state, self.terminal_reward, True
        return self.state, self.step_penalty, False

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state without modifying environment."""
        if state == self.terminal:
            return state
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        return (nr, nc)

    def get_reward(self, state: Tuple[int, int], action: int,
                   next_state: Tuple[int, int]) -> float:
        """Get reward for transition."""
        if state == self.terminal:
            return 0.0
        if next_state == self.terminal:
            return self.terminal_reward
        return self.step_penalty

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal."""
        return state == self.terminal

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state tuple to flat index."""
        return state[0] * self.N + state[1]

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to state tuple."""
        return (idx // self.N, idx % self.N)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def q_to_policy(Q: Dict, env: GridworldEnv) -> Dict:
    """Extract greedy policy from Q-values."""
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
        else:
            q_vals = [Q.get((s, a), 0.0) for a in range(env.num_actions)]
            policy[s] = int(np.argmax(q_vals))
    return policy


def v_to_array(V: Dict, env: GridworldEnv) -> np.ndarray:
    """Convert V dict to numpy array indexed by state index."""
    V_arr = np.zeros(env.num_states)
    for s in env.states:
        V_arr[env.state_to_index(s)] = V.get(s, 0.0)
    return V_arr


def policy_to_array(policy: Dict, env: GridworldEnv) -> np.ndarray:
    """Convert policy dict to numpy array indexed by state index."""
    pi_arr = np.zeros(env.num_states, dtype=int)
    for s in env.states:
        pi_arr[env.state_to_index(s)] = policy.get(s, 0)
    return pi_arr


def compute_value_error(V_learned: Dict, V_optimal: np.ndarray,
                        env: GridworldEnv) -> float:
    """Compute RMSE between learned and optimal value functions."""
    errors = []
    for s in env.states:
        v_l = V_learned.get(s, 0.0)
        v_o = V_optimal[env.state_to_index(s)]
        errors.append((v_l - v_o) ** 2)
    return np.sqrt(np.mean(errors))


def compute_policy_agreement(policy_learned: Dict, policy_optimal: np.ndarray,
                             env: GridworldEnv) -> float:
    """Compute fraction of states where policies agree."""
    agreements = 0
    for s in env.states:
        if policy_learned.get(s, 0) == policy_optimal[env.state_to_index(s)]:
            agreements += 1
    return agreements / env.num_states


def evaluate_policy(env: GridworldEnv, policy: Dict, n_episodes: int = 100,
                    horizon: int = 100) -> Tuple[float, float]:
    """Evaluate policy via simulation. Returns (mean_return, mean_steps)."""
    returns = []
    steps = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_return = 0.0
        for t in range(horizon):
            a = policy.get(s, 0)
            ns, r, done = env.step(a)
            ep_return += r
            if done:
                steps.append(t + 1)
                break
            s = ns
        else:
            steps.append(horizon)
        returns.append(ep_return)
    return np.mean(returns), np.mean(steps)


def epsilon_greedy_action(Q: Dict, state: Tuple, num_actions: int,
                          epsilon: float) -> int:
    """Select action using epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    q_vals = [Q.get((state, a), 0.0) for a in range(num_actions)]
    return int(np.argmax(q_vals))


def softmax_action(Q: Dict, state: Tuple, num_actions: int,
                   temperature: float = 1.0) -> int:
    """Select action using softmax policy."""
    q_vals = np.array([Q.get((state, a), 0.0) for a in range(num_actions)])
    q_vals = q_vals - np.max(q_vals)  # For numerical stability
    probs = np.exp(q_vals / temperature)
    probs = probs / np.sum(probs)
    return np.random.choice(num_actions, p=probs)


def softmax_probs(Q: Dict, state: Tuple, num_actions: int,
                  temperature: float = 1.0) -> np.ndarray:
    """Get softmax action probabilities."""
    q_vals = np.array([Q.get((state, a), 0.0) for a in range(num_actions)])
    q_vals = q_vals - np.max(q_vals)
    probs = np.exp(q_vals / temperature)
    return probs / np.sum(probs)


# ---------------------------------------------------------------------------
# Algorithm 1: Value Iteration (Model-Based Planning)
# ---------------------------------------------------------------------------

def run_value_iteration(env: GridworldEnv, tol: float = 1e-8,
                        max_iter: int = 1000) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Tabular Value Iteration.

    Returns:
        V: Dict[state -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    t0 = time.time()
    gamma = env.gamma

    # Initialize value function
    V = {s: 0.0 for s in env.states}

    iterations = 0
    final_residual = 0.0

    for iteration in range(max_iter):
        V_new = {}
        max_diff = 0.0

        for s in env.states:
            if env.is_terminal(s):
                V_new[s] = 0.0
                continue

            # Bellman optimality update
            best_val = float('-inf')
            for a in range(env.num_actions):
                ns = env.get_next_state(s, a)
                r = env.get_reward(s, a, ns)
                val = r + gamma * V[ns]
                if val > best_val:
                    best_val = val
            V_new[s] = best_val
            max_diff = max(max_diff, abs(V_new[s] - V[s]))

        V = V_new
        iterations = iteration + 1
        final_residual = max_diff

        if max_diff < tol:
            break

    # Extract greedy policy
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
            continue
        best_a = 0
        best_val = float('-inf')
        for a in range(env.num_actions):
            ns = env.get_next_state(s, a)
            r = env.get_reward(s, a, ns)
            val = r + gamma * V[ns]
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a

    wall_time = time.time() - t0

    metrics = AlgorithmMetrics(
        iterations=iterations,
        final_residual=final_residual,
        wall_time=wall_time
    )

    return V, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 2: Policy Iteration (Model-Based Planning)
# ---------------------------------------------------------------------------

def run_policy_iteration(env: GridworldEnv, tol: float = 1e-8,
                         max_iter: int = 100,
                         max_eval_iter: int = 1000) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Tabular Policy Iteration.

    Returns:
        V: Dict[state -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    t0 = time.time()
    gamma = env.gamma

    # Initialize random policy
    policy = {s: np.random.randint(env.num_actions) for s in env.states}
    V = {s: 0.0 for s in env.states}

    iterations = 0

    for iteration in range(max_iter):
        # Policy Evaluation
        for _ in range(max_eval_iter):
            V_new = {}
            max_diff = 0.0
            for s in env.states:
                if env.is_terminal(s):
                    V_new[s] = 0.0
                    continue
                a = policy[s]
                ns = env.get_next_state(s, a)
                r = env.get_reward(s, a, ns)
                V_new[s] = r + gamma * V[ns]
                max_diff = max(max_diff, abs(V_new[s] - V[s]))
            V = V_new
            if max_diff < tol:
                break

        # Policy Improvement
        policy_stable = True
        for s in env.states:
            if env.is_terminal(s):
                continue
            old_action = policy[s]
            best_a = 0
            best_val = float('-inf')
            for a in range(env.num_actions):
                ns = env.get_next_state(s, a)
                r = env.get_reward(s, a, ns)
                val = r + gamma * V[ns]
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s] = best_a
            if old_action != best_a:
                policy_stable = False

        iterations = iteration + 1

        if policy_stable:
            break

    wall_time = time.time() - t0

    metrics = AlgorithmMetrics(
        iterations=iterations,
        final_residual=0.0,
        wall_time=wall_time
    )

    return V, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 3: Q-Learning (Off-Policy TD Control)
# ---------------------------------------------------------------------------

def run_q_learning(env: GridworldEnv, num_episodes: int = 5000,
                   horizon: int = 100, alpha: float = 0.1,
                   alpha_decay: float = 0.999, epsilon_start: float = 1.0,
                   epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                   seed: int = 42, V_optimal: np.ndarray = None,
                   policy_optimal: np.ndarray = None, optimal_return: float = None,
                   eval_freq: int = 100, eval_episodes: int = 100
                   ) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Tabular Q-Learning with epsilon-greedy exploration.

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table
    Q = {}
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0

    epsilon = epsilon_start
    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Epsilon-greedy action selection
            a = epsilon_greedy_action(Q, s, env.num_actions, epsilon)

            # Take action
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            # Q-learning update (off-policy: use max over next state)
            if done:
                td_target = r
            else:
                max_q_next = max(Q.get((ns, a_), 0.0) for a_ in range(env.num_actions))
                td_target = r + gamma * max_q_next

            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha_t * td_error

            if done:
                break
            s = ns

        # Record episode metrics
        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Decay epsilon and alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            # Value error and policy agreement
            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 4: SARSA (On-Policy TD Control)
# ---------------------------------------------------------------------------

def run_sarsa(env: GridworldEnv, num_episodes: int = 5000,
              horizon: int = 100, alpha: float = 0.1,
              alpha_decay: float = 0.999, epsilon_start: float = 1.0,
              epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
              seed: int = 42, V_optimal: np.ndarray = None,
              policy_optimal: np.ndarray = None, optimal_return: float = None,
              eval_freq: int = 100, eval_episodes: int = 100
              ) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Tabular SARSA (on-policy TD control).

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table
    Q = {}
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0

    epsilon = epsilon_start
    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        s = env.reset()
        a = epsilon_greedy_action(Q, s, env.num_actions, epsilon)
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Take action
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            # SARSA update (on-policy: use actual next action)
            if done:
                td_target = r
                td_error = td_target - Q[(s, a)]
                Q[(s, a)] += alpha_t * td_error
                break
            else:
                # Select next action (on-policy)
                na = epsilon_greedy_action(Q, ns, env.num_actions, epsilon)
                td_target = r + gamma * Q[(ns, na)]
                td_error = td_target - Q[(s, a)]
                Q[(s, a)] += alpha_t * td_error
                s = ns
                a = na

        # Record episode metrics
        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Decay epsilon and alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 5: Expected SARSA
# ---------------------------------------------------------------------------

def run_expected_sarsa(env: GridworldEnv, num_episodes: int = 5000,
                       horizon: int = 100, alpha: float = 0.1,
                       alpha_decay: float = 0.999, epsilon_start: float = 1.0,
                       epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                       seed: int = 42, V_optimal: np.ndarray = None,
                       policy_optimal: np.ndarray = None, optimal_return: float = None,
                       eval_freq: int = 100, eval_episodes: int = 100
                       ) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Expected SARSA: uses expected Q-value under current policy.

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table
    Q = {}
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0

    epsilon = epsilon_start
    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Epsilon-greedy action selection
            a = epsilon_greedy_action(Q, s, env.num_actions, epsilon)

            # Take action
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            # Expected SARSA update
            if done:
                td_target = r
            else:
                # Compute expected value under epsilon-greedy policy
                q_vals = np.array([Q.get((ns, a_), 0.0) for a_ in range(env.num_actions)])
                greedy_action = np.argmax(q_vals)
                expected_q = 0.0
                for a_ in range(env.num_actions):
                    if a_ == greedy_action:
                        prob = 1.0 - epsilon + epsilon / env.num_actions
                    else:
                        prob = epsilon / env.num_actions
                    expected_q += prob * q_vals[a_]
                td_target = r + gamma * expected_q

            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha_t * td_error

            if done:
                break
            s = ns

        # Record episode metrics
        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Decay epsilon and alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 6: Monte Carlo Control (First-Visit, Epsilon-Greedy)
# ---------------------------------------------------------------------------

def run_mc_control(env: GridworldEnv, num_episodes: int = 5000,
                   horizon: int = 100, epsilon_start: float = 1.0,
                   epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                   seed: int = 42, V_optimal: np.ndarray = None,
                   policy_optimal: np.ndarray = None, optimal_return: float = None,
                   eval_freq: int = 100, eval_episodes: int = 100
                   ) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """First-visit Monte Carlo Control with epsilon-greedy policy.

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table and visit counts
    Q = {}
    N = {}  # Visit counts for incremental mean
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0
            N[(s, a)] = 0

    epsilon = epsilon_start

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        # Generate episode
        episode = []
        s = env.reset()
        ep_return = 0.0

        for t in range(horizon):
            metrics.states_visited.add(s)
            a = epsilon_greedy_action(Q, s, env.num_actions, epsilon)
            ns, r, done = env.step(a)
            episode.append((s, a, r))
            ep_return += r
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(len(episode))

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # First-visit MC update
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                N[(s, a)] += 1
                # Incremental mean update
                Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 7: SARSA(λ) with Eligibility Traces
# ---------------------------------------------------------------------------

def run_sarsa_lambda(env: GridworldEnv, num_episodes: int = 5000,
                     horizon: int = 100, alpha: float = 0.1,
                     alpha_decay: float = 0.999, lambda_: float = 0.9,
                     epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                     epsilon_decay: float = 0.995, seed: int = 42,
                     V_optimal: np.ndarray = None, policy_optimal: np.ndarray = None,
                     optimal_return: float = None, eval_freq: int = 100,
                     eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """SARSA(λ) with replacing eligibility traces.

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table
    Q = {}
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0

    epsilon = epsilon_start
    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        # Initialize eligibility traces
        E = {}

        s = env.reset()
        a = epsilon_greedy_action(Q, s, env.num_actions, epsilon)
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Take action
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            # Compute TD error
            if done:
                delta = r - Q[(s, a)]
            else:
                na = epsilon_greedy_action(Q, ns, env.num_actions, epsilon)
                delta = r + gamma * Q[(ns, na)] - Q[(s, a)]

            # Update eligibility trace (replacing traces)
            E[(s, a)] = 1.0

            # Update all Q-values
            keys_to_update = list(E.keys())
            for key in keys_to_update:
                Q[key] += alpha_t * delta * E[key]
                E[key] *= gamma * lambda_
                # Remove negligible traces
                if E[key] < 1e-6:
                    del E[key]

            if done:
                break

            s = ns
            a = na

        # Record episode metrics
        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Decay epsilon and alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 8: Watkins's Q(λ)
# ---------------------------------------------------------------------------

def run_q_lambda(env: GridworldEnv, num_episodes: int = 5000,
                 horizon: int = 100, alpha: float = 0.1,
                 alpha_decay: float = 0.999, lambda_: float = 0.9,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, seed: int = 42,
                 V_optimal: np.ndarray = None, policy_optimal: np.ndarray = None,
                 optimal_return: float = None, eval_freq: int = 100,
                 eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Watkins's Q(λ): eligibility traces reset on non-greedy actions.

    Returns:
        Q: Dict[(state, action) -> value]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize Q-table
    Q = {}
    for s in env.states:
        for a in range(env.num_actions):
            Q[(s, a)] = 0.0

    epsilon = epsilon_start
    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        # Initialize eligibility traces
        E = {}

        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Epsilon-greedy action selection
            q_vals = [Q.get((s, a_), 0.0) for a_ in range(env.num_actions)]
            greedy_a = int(np.argmax(q_vals))

            if np.random.random() < epsilon:
                a = np.random.randint(env.num_actions)
            else:
                a = greedy_a

            is_greedy = (a == greedy_a)

            # Take action
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            # Compute TD error (Q-learning style: max over next state)
            if done:
                delta = r - Q[(s, a)]
            else:
                max_q_next = max(Q.get((ns, a_), 0.0) for a_ in range(env.num_actions))
                delta = r + gamma * max_q_next - Q[(s, a)]

            # Update eligibility trace (replacing traces)
            E[(s, a)] = 1.0

            # Update all Q-values
            keys_to_update = list(E.keys())
            for key in keys_to_update:
                Q[key] += alpha_t * delta * E[key]

                # Watkins's Q(λ): if action was greedy, decay trace; else reset all
                if is_greedy:
                    E[key] *= gamma * lambda_
                else:
                    E[key] = 0.0

                # Remove negligible traces
                if E[key] < 1e-6:
                    del E[key]

            if done:
                break
            s = ns

        # Record episode metrics
        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Decay epsilon and alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            policy = q_to_policy(Q, env)
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                V_learned = {s: max(Q.get((s, a), 0.0) for a in range(env.num_actions))
                             for s in env.states}
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    policy = q_to_policy(Q, env)
    metrics.wall_time = time.time() - t0

    return Q, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 9: REINFORCE (Policy Gradient)
# ---------------------------------------------------------------------------

def run_reinforce(env: GridworldEnv, num_episodes: int = 5000,
                  horizon: int = 100, alpha: float = 0.01,
                  alpha_decay: float = 0.999, temperature: float = 1.0,
                  seed: int = 42, V_optimal: np.ndarray = None,
                  policy_optimal: np.ndarray = None, optimal_return: float = None,
                  eval_freq: int = 100, eval_episodes: int = 100,
                  baseline: bool = True) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """REINFORCE policy gradient with softmax policy.

    Uses tabular softmax policy parameterization: π(a|s) ∝ exp(θ(s,a)/τ)

    Returns:
        theta: Dict[(state, action) -> preference]
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize policy parameters (preferences)
    theta = {}
    for s in env.states:
        for a in range(env.num_actions):
            theta[(s, a)] = 0.0

    # Baseline: average return per state
    V_baseline = {s: 0.0 for s in env.states}
    V_count = {s: 0 for s in env.states}

    alpha_t = alpha

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        # Generate episode using current policy
        episode = []
        s = env.reset()
        ep_return = 0.0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Sample action from softmax policy
            a = softmax_action(theta, s, env.num_actions, temperature)
            ns, r, done = env.step(a)
            episode.append((s, a, r))
            ep_return += r
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(len(episode))

        # Cumulative regret
        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Compute returns and update policy
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r

            # Update baseline
            V_count[s] += 1
            V_baseline[s] += (G - V_baseline[s]) / V_count[s]

            # Advantage
            if baseline:
                advantage = G - V_baseline[s]
            else:
                advantage = G

            # Policy gradient update
            probs = softmax_probs(theta, s, env.num_actions, temperature)
            for a_ in range(env.num_actions):
                if a_ == a:
                    grad = 1.0 - probs[a_]
                else:
                    grad = -probs[a_]
                theta[(s, a_)] += alpha_t * advantage * grad / temperature

        # Decay alpha
        alpha_t = alpha_t * alpha_decay

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            # Extract greedy policy
            policy = {}
            for s in env.states:
                probs = softmax_probs(theta, s, env.num_actions, temperature)
                policy[s] = int(np.argmax(probs))

            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                # Use baseline as V estimate
                v_err = compute_value_error(V_baseline, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    # Final policy
    policy = {}
    for s in env.states:
        probs = softmax_probs(theta, s, env.num_actions, temperature)
        policy[s] = int(np.argmax(probs))

    metrics.wall_time = time.time() - t0

    return theta, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 10: DQN (for comparison with tabular methods)
# ---------------------------------------------------------------------------

def run_dqn_tabular_comparison(env: GridworldEnv, num_episodes: int = 5000,
                               horizon: int = 100, seed: int = 42,
                               V_optimal: np.ndarray = None,
                               policy_optimal: np.ndarray = None,
                               optimal_return: float = None,
                               eval_freq: int = 100, eval_episodes: int = 100,
                               # DQN hyperparameters
                               replay_size: int = 10000, batch_size: int = 64,
                               lr: float = 1e-3, epsilon_start: float = 1.0,
                               epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                               target_update_freq: int = 50, hidden_dim: int = 64
                               ) -> Tuple[object, Dict, AlgorithmMetrics]:
    """DQN for gridworld (using PyTorch).

    Returns:
        q_net: Trained Q-network
        policy: Dict[state -> action]
        metrics: AlgorithmMetrics
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    gamma = env.gamma

    # Simple Q-network
    class QNet(nn.Module):
        def __init__(self, state_dim, num_actions, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            )

        def forward(self, x):
            return self.net(x)

    state_dim = 2  # (row, col) normalized
    q_net = QNet(state_dim, env.num_actions, hidden_dim)
    target_net = QNet(state_dim, env.num_actions, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=replay_size)

    def state_to_tensor(s):
        return torch.FloatTensor([s[0] / (env.N - 1), s[1] / (env.N - 1)])

    epsilon = epsilon_start
    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(env.num_actions)
            else:
                with torch.no_grad():
                    q_vals = q_net(state_to_tensor(s).unsqueeze(0))
                    a = q_vals.argmax(dim=1).item()

            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1

            replay_buffer.append((s, a, r, ns, done))

            # Train
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                s_batch = torch.stack([state_to_tensor(x[0]) for x in batch])
                a_batch = torch.LongTensor([x[1] for x in batch])
                r_batch = torch.FloatTensor([x[2] for x in batch])
                ns_batch = torch.stack([state_to_tensor(x[3]) for x in batch])
                d_batch = torch.FloatTensor([float(x[4]) for x in batch])

                q_values = q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_batch).max(dim=1)[0]
                    target = r_batch + gamma * next_q * (1.0 - d_batch)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            # Extract greedy policy
            policy = {}
            q_net.eval()
            for s in env.states:
                with torch.no_grad():
                    q_vals = q_net(state_to_tensor(s).unsqueeze(0))
                    policy[s] = q_vals.argmax(dim=1).item()
            q_net.train()

            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                # Estimate V from Q-network
                V_learned = {}
                q_net.eval()
                for s in env.states:
                    with torch.no_grad():
                        q_vals = q_net(state_to_tensor(s).unsqueeze(0))
                        V_learned[s] = q_vals.max().item()
                q_net.train()
                v_err = compute_value_error(V_learned, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    # Final policy
    policy = {}
    q_net.eval()
    for s in env.states:
        with torch.no_grad():
            q_vals = q_net(state_to_tensor(s).unsqueeze(0))
            policy[s] = q_vals.argmax(dim=1).item()

    metrics.wall_time = time.time() - t0

    return q_net, policy, metrics


# ---------------------------------------------------------------------------
# Bertsekas Framework: Planning Methods with Approximation
# ---------------------------------------------------------------------------

def compute_lookahead_value(env: GridworldEnv, state: Tuple[int, int],
                            V_base: Dict, gamma: float, depth: int) -> float:
    """Compute depth-step lookahead value from state using V_base as terminal.

    Args:
        env: GridworldEnv instance
        state: Current state tuple
        V_base: Base value function approximation (terminal values for lookahead)
        gamma: Discount factor
        depth: Lookahead depth (0 = use V_base directly)

    Returns:
        Best value achievable with depth-step lookahead
    """
    if env.is_terminal(state):
        return 0.0
    if depth == 0:
        return V_base.get(state, 0.0)

    # One-step lookahead: best action value with (depth-1) recursion
    best_val = float('-inf')
    for a in range(env.num_actions):
        ns = env.get_next_state(state, a)
        r = env.get_reward(state, a, ns)
        if env.is_terminal(ns):
            val = r
        else:
            val = r + gamma * compute_lookahead_value(env, ns, V_base, gamma, depth - 1)
        if val > best_val:
            best_val = val

    return best_val


def best_lookahead_action(env: GridworldEnv, state: Tuple[int, int],
                          V_base: Dict, gamma: float, depth: int) -> int:
    """Find best action using depth-step lookahead tree search.

    Args:
        env: GridworldEnv instance
        state: Current state tuple
        V_base: Base value function approximation
        gamma: Discount factor
        depth: Lookahead depth (1 = one-step rollout)

    Returns:
        Best action index
    """
    if depth == 0:
        # Greedy w.r.t. V_base (no lookahead)
        best_val = float('-inf')
        best_a = 0
        for a in range(env.num_actions):
            ns = env.get_next_state(state, a)
            r = env.get_reward(state, a, ns)
            val = r + gamma * V_base.get(ns, 0.0)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    best_val = float('-inf')
    best_a = 0

    for a in range(env.num_actions):
        ns = env.get_next_state(state, a)
        r = env.get_reward(state, a, ns)
        if env.is_terminal(ns):
            val = r
        elif depth == 1:
            val = r + gamma * V_base.get(ns, 0.0)
        else:
            # Recursive lookahead
            future_val = compute_lookahead_value(env, ns, V_base, gamma, depth - 1)
            val = r + gamma * future_val

        if val > best_val:
            best_val = val
            best_a = a

    return best_a


# ---------------------------------------------------------------------------
# Algorithm 11: Rollout (One-Step Lookahead with Base Policy)
# ---------------------------------------------------------------------------

def run_rollout(env: GridworldEnv, V_base: Dict = None, num_episodes: int = 5000,
                horizon: int = 100, seed: int = 42,
                V_optimal: np.ndarray = None, policy_optimal: np.ndarray = None,
                optimal_return: float = None, eval_freq: int = 100,
                eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Rollout policy: one-step lookahead using V_base as terminal value.

    The rollout policy at each state selects:
        a* = argmax_a [r(s,a) + gamma * V_base(s')]

    If V_base is the optimal value function, this recovers the optimal policy.
    If V_base is approximate, rollout provides policy improvement.

    Args:
        env: GridworldEnv instance
        V_base: Base value function approximation. If None, uses heuristic.
        num_episodes: Number of episodes to simulate
        horizon: Maximum steps per episode
        seed: Random seed
        V_optimal, policy_optimal, optimal_return: For metrics computation
        eval_freq, eval_episodes: Evaluation frequency and sample size

    Returns:
        V_base: The base value function used
        policy: Rollout policy (greedy w.r.t. one-step lookahead)
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize V_base if not provided (Manhattan distance heuristic)
    if V_base is None:
        V_base = {}
        for s in env.states:
            if env.is_terminal(s):
                V_base[s] = 0.0
            else:
                # Heuristic: discounted steps to goal assuming direct path
                dist = abs(env.terminal[0] - s[0]) + abs(env.terminal[1] - s[1])
                # Expected return: step_penalty * steps + terminal_reward * gamma^steps
                V_base[s] = env.step_penalty * dist + env.terminal_reward * (gamma ** dist)

    # Extract rollout policy (one-step lookahead)
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
        else:
            policy[s] = best_lookahead_action(env, s, V_base, gamma, depth=1)

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    # Simulate episodes with rollout policy
    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)
            a = policy[s]
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                v_err = compute_value_error(V_base, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    metrics.wall_time = time.time() - t0

    return V_base, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 12: Multi-Step Lookahead
# ---------------------------------------------------------------------------

def run_lookahead(env: GridworldEnv, V_base: Dict = None, depth: int = 2,
                  num_episodes: int = 5000, horizon: int = 100, seed: int = 42,
                  V_optimal: np.ndarray = None, policy_optimal: np.ndarray = None,
                  optimal_return: float = None, eval_freq: int = 100,
                  eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Multi-step lookahead policy: depth-step tree search + V_base terminal.

    At each state, performs depth-step tree search:
        a* = argmax_a [r(s,a) + gamma * V^(depth-1)(s')]
    where V^(k) is the k-step lookahead value.

    Note: Complexity is O(|A|^depth) per action selection.

    Args:
        env: GridworldEnv instance
        V_base: Base value function approximation
        depth: Lookahead depth (1=rollout, 2+=tree search)
        num_episodes: Number of episodes to simulate
        horizon: Maximum steps per episode
        seed: Random seed
        V_optimal, policy_optimal, optimal_return: For metrics
        eval_freq, eval_episodes: Evaluation frequency

    Returns:
        V_base: The base value function used
        policy: Lookahead policy
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize V_base if not provided
    if V_base is None:
        V_base = {}
        for s in env.states:
            if env.is_terminal(s):
                V_base[s] = 0.0
            else:
                dist = abs(env.terminal[0] - s[0]) + abs(env.terminal[1] - s[1])
                V_base[s] = env.step_penalty * dist + env.terminal_reward * (gamma ** dist)

    # Extract lookahead policy
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
        else:
            policy[s] = best_lookahead_action(env, s, V_base, gamma, depth=depth)

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    # Simulate episodes
    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)
            a = policy[s]
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        if (ep + 1) % eval_freq == 0:
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                v_err = compute_value_error(V_base, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    metrics.wall_time = time.time() - t0

    return V_base, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 13: MPC (Model Predictive Control / Receding Horizon)
# ---------------------------------------------------------------------------

def mpc_solve_horizon(env: GridworldEnv, start_state: Tuple[int, int],
                      H: int, gamma: float, V_terminal: Dict = None) -> int:
    """Solve H-step optimal control problem and return first action.

    Uses backward induction over H-step horizon with optional terminal value heuristic.

    Args:
        env: GridworldEnv instance
        start_state: Current state
        H: Planning horizon
        gamma: Discount factor
        V_terminal: Terminal value function for states at horizon boundary.
                   If None, uses Manhattan distance heuristic.

    Returns:
        Best first action
    """
    if env.is_terminal(start_state):
        return 0

    # Initialize terminal value heuristic if not provided
    if V_terminal is None:
        V_terminal = {}
        for s in env.states:
            if env.is_terminal(s):
                V_terminal[s] = 0.0
            else:
                dist = abs(env.terminal[0] - s[0]) + abs(env.terminal[1] - s[1])
                V_terminal[s] = env.step_penalty * dist + env.terminal_reward * (gamma ** dist)

    # For H=1, this is just one-step lookahead with terminal values
    if H == 1:
        best_val = float('-inf')
        best_a = 0
        for a in range(env.num_actions):
            ns = env.get_next_state(start_state, a)
            r = env.get_reward(start_state, a, ns)
            if env.is_terminal(ns):
                val = r
            else:
                val = r + gamma * V_terminal.get(ns, 0.0)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    # For H > 1, use recursive tree search with memoization
    # V_cache[h][s] = best value-to-go from state s with h steps remaining
    V_cache = [{} for _ in range(H + 1)]

    def get_value(state, h):
        """Get value at state with h steps remaining."""
        if env.is_terminal(state):
            return 0.0
        if h == 0:
            # At horizon boundary, use terminal value heuristic
            return V_terminal.get(state, 0.0)
        if state in V_cache[h]:
            return V_cache[h][state]

        # Compute value via Bellman optimality
        best_val = float('-inf')
        for a in range(env.num_actions):
            ns = env.get_next_state(state, a)
            r = env.get_reward(state, a, ns)
            val = r + gamma * get_value(ns, h - 1)
            if val > best_val:
                best_val = val
        V_cache[h][state] = best_val
        return best_val

    # Find best first action
    best_val = float('-inf')
    best_a = 0
    for a in range(env.num_actions):
        ns = env.get_next_state(start_state, a)
        r = env.get_reward(start_state, a, ns)
        val = r + gamma * get_value(ns, H - 1)
        if val > best_val:
            best_val = val
            best_a = a

    return best_a


def run_mpc(env: GridworldEnv, horizon_H: int = 5, num_episodes: int = 5000,
            episode_horizon: int = 100, seed: int = 42,
            V_optimal: np.ndarray = None, policy_optimal: np.ndarray = None,
            optimal_return: float = None, eval_freq: int = 100,
            eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Model Predictive Control: plan H steps, execute 1, replan.

    At each state:
    1. Solve H-step optimal control problem via dynamic programming
    2. Execute first action
    3. Observe next state, repeat

    Args:
        env: GridworldEnv instance
        horizon_H: MPC planning horizon
        num_episodes: Number of episodes to simulate
        episode_horizon: Maximum steps per episode
        seed: Random seed
        V_optimal, policy_optimal, optimal_return: For metrics
        eval_freq, eval_episodes: Evaluation frequency

    Returns:
        mpc_policy: Dict mapping states to MPC actions (cached)
        policy: Same as mpc_policy
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Cache MPC decisions for each state (since env is deterministic)
    mpc_policy = {}
    for s in env.states:
        if env.is_terminal(s):
            mpc_policy[s] = 0
        else:
            mpc_policy[s] = mpc_solve_horizon(env, s, horizon_H, gamma)

    policy = mpc_policy  # Alias for interface consistency

    metrics = AlgorithmMetrics()
    cumulative_regret = 0.0

    # Simulate episodes
    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(episode_horizon):
            metrics.states_visited.add(s)
            a = mpc_policy[s]
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        if (ep + 1) % eval_freq == 0:
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=episode_horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                # MPC doesn't maintain explicit V, use simulated values
                V_mpc = {}
                for s in env.states:
                    if env.is_terminal(s):
                        V_mpc[s] = 0.0
                    else:
                        # Estimate V via one rollout
                        test_s = s
                        val = 0.0
                        disc = 1.0
                        for _ in range(episode_horizon):
                            if env.is_terminal(test_s):
                                break
                            a = mpc_policy[test_s]
                            test_ns = env.get_next_state(test_s, a)
                            r = env.get_reward(test_s, a, test_ns)
                            val += disc * r
                            disc *= gamma
                            test_s = test_ns
                        V_mpc[s] = val
                v_err = compute_value_error(V_mpc, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    metrics.wall_time = time.time() - t0

    return mpc_policy, policy, metrics


# ---------------------------------------------------------------------------
# Algorithm 14: Truncated Rollout (VI Warmup + One-Step Lookahead)
# ---------------------------------------------------------------------------

def run_truncated_rollout(env: GridworldEnv, m_warmup: int = 5,
                          V_init: Dict = None, num_episodes: int = 5000,
                          horizon: int = 100, seed: int = 42,
                          V_optimal: np.ndarray = None,
                          policy_optimal: np.ndarray = None,
                          optimal_return: float = None, eval_freq: int = 100,
                          eval_episodes: int = 100) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Truncated Rollout: m VI steps on initial approximation, then one-step lookahead.

    This demonstrates Bertsekas's insight that VI iterations "pull" the approximation
    into the region of stability before applying the rollout policy improvement.

    Algorithm:
    1. Initialize V_base (heuristic or provided)
    2. Apply m iterations of VI to V_base
    3. Use one-step lookahead with improved V_base as terminal values

    Args:
        env: GridworldEnv instance
        m_warmup: Number of VI warmup iterations
        V_init: Initial value function (if None, uses heuristic)
        num_episodes: Number of episodes to simulate
        horizon: Maximum steps per episode
        seed: Random seed
        V_optimal, policy_optimal, optimal_return: For metrics
        eval_freq, eval_episodes: Evaluation frequency

    Returns:
        V_base: Warmed-up value function
        policy: Truncated rollout policy
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()
    gamma = env.gamma

    # Initialize V_base
    if V_init is None:
        V_base = {}
        for s in env.states:
            if env.is_terminal(s):
                V_base[s] = 0.0
            else:
                dist = abs(env.terminal[0] - s[0]) + abs(env.terminal[1] - s[1])
                V_base[s] = env.step_penalty * dist + env.terminal_reward * (gamma ** dist)
    else:
        V_base = V_init.copy()

    # Apply m VI warmup iterations
    for _ in range(m_warmup):
        V_new = {}
        for s in env.states:
            if env.is_terminal(s):
                V_new[s] = 0.0
                continue
            best_val = float('-inf')
            for a in range(env.num_actions):
                ns = env.get_next_state(s, a)
                r = env.get_reward(s, a, ns)
                val = r + gamma * V_base[ns]
                if val > best_val:
                    best_val = val
            V_new[s] = best_val
        V_base = V_new

    # Extract policy via one-step lookahead with warmed-up V_base
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
        else:
            policy[s] = best_lookahead_action(env, s, V_base, gamma, depth=1)

    metrics = AlgorithmMetrics()
    metrics.iterations = m_warmup  # Record warmup iterations
    cumulative_regret = 0.0

    # Simulate episodes
    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for t in range(horizon):
            metrics.states_visited.add(s)
            a = policy[s]
            ns, r, done = env.step(a)
            ep_return += r
            ep_steps = t + 1
            if done:
                break
            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_steps)

        if optimal_return is not None:
            cumulative_regret += (optimal_return - ep_return)
            metrics.cumulative_regret.append(cumulative_regret)

        if (ep + 1) % eval_freq == 0:
            eval_return, eval_steps = evaluate_policy(env, policy,
                                                       n_episodes=eval_episodes,
                                                       horizon=horizon)
            metrics.eval_returns.append(eval_return)
            metrics.eval_steps.append(eval_steps)
            metrics.checkpoint_episodes.append(ep + 1)

            if V_optimal is not None:
                v_err = compute_value_error(V_base, V_optimal, env)
                metrics.value_errors.append(v_err)

            if policy_optimal is not None:
                agr = compute_policy_agreement(policy, policy_optimal, env)
                metrics.policy_agreements.append(agr)

    metrics.wall_time = time.time() - t0

    return V_base, policy, metrics
