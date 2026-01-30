# Gridworld Algorithm Implementations
# Chapter 3 -- Comprehensive RL Algorithm Comparison Study
# Provides unified interface for classical RL algorithms on gridworld.

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# ---------------------------------------------------------------------------
# Metrics dataclass for tracking algorithm performance
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmMetrics:
    """Unified metrics tracked across all algorithms."""
    # Learning curve data (per checkpoint)
    checkpoint_episodes: List[int] = field(default_factory=list)
    checkpoint_returns: List[float] = field(default_factory=list)
    checkpoint_value_errors: List[float] = field(default_factory=list)
    checkpoint_policy_agreements: List[float] = field(default_factory=list)

    # Per-episode data
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    # Summary statistics
    final_return: float = 0.0
    final_steps: float = 0.0
    final_value_error: float = 0.0
    final_policy_agreement: float = 0.0
    states_visited: int = 0
    wall_time: float = 0.0

    # Algorithm-specific
    iterations: int = 0  # For VI/PI


# ---------------------------------------------------------------------------
# Gridworld Environment (standalone, no EconBenchmark dependency)
# ---------------------------------------------------------------------------

class GridworldEnv:
    """NxN deterministic gridworld for algorithm comparison.

    State: (row, col) tuple
    Actions: 0=Left, 1=Right, 2=Up, 3=Down, 4=Stay
    Terminal: (N-1, N-1) with reward TERMINAL_REWARD
    All other steps: STEP_PENALTY
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['Left', 'Right', 'Up', 'Down', 'Stay']

    def __init__(self, N: int, step_penalty: float = -0.1,
                 terminal_reward: float = 10.0):
        self.N = N
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.terminal = (N - 1, N - 1)
        self.initial = (0, 0)
        self.state = None
        self.reset()

    @property
    def num_states(self) -> int:
        return self.N * self.N

    @property
    def num_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> Tuple[int, int]:
        self.state = self.initial
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
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

    def state_to_index(self, state: Tuple[int, int]) -> int:
        return state[0] * self.N + state[1]

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        return (idx // self.N, idx % self.N)

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Return next state without modifying environment."""
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        return (nr, nc)

    def get_reward(self, state: Tuple[int, int], action: int) -> float:
        """Return reward for (state, action) pair."""
        if state == self.terminal:
            return 0.0
        next_state = self.get_next_state(state, action)
        if next_state == self.terminal:
            return self.terminal_reward
        return self.step_penalty

    def get_all_states(self) -> List[Tuple[int, int]]:
        """Return list of all states."""
        return [self.index_to_state(i) for i in range(self.num_states)]

    def state_to_features(self, state: Tuple[int, int]) -> np.ndarray:
        """Normalized feature vector for neural network input."""
        return np.array([state[0] / max(self.N - 1, 1),
                         state[1] / max(self.N - 1, 1)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_policy(env: GridworldEnv, policy: Dict, n_episodes: int = 100,
                   horizon: int = 100) -> Tuple[float, float]:
    """Evaluate a tabular policy. Returns (mean_return, mean_steps)."""
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


def compute_value_error(Q: Dict, V_optimal: np.ndarray,
                       env: GridworldEnv) -> float:
    """Compute RMSE between learned V (max Q) and optimal V."""
    errors = []
    for s in env.get_all_states():
        si = env.state_to_index(s)
        v_opt = V_optimal[si]
        if s in Q and Q[s]:
            v_learned = max(Q[s].values())
        else:
            v_learned = 0.0
        errors.append((v_learned - v_opt) ** 2)
    return np.sqrt(np.mean(errors))


def compute_policy_agreement(policy: Dict, optimal_policy: np.ndarray,
                            env: GridworldEnv) -> float:
    """Fraction of states where learned policy matches optimal."""
    agreements = 0
    for s in env.get_all_states():
        si = env.state_to_index(s)
        opt_a = optimal_policy[si]
        learned_a = policy.get(s, 0)
        if learned_a == opt_a:
            agreements += 1
    return agreements / env.num_states


def extract_policy_from_Q(Q: Dict, env: GridworldEnv) -> Dict:
    """Extract greedy policy from Q-table."""
    policy = {}
    for s in env.get_all_states():
        if s in Q and Q[s]:
            policy[s] = max(Q[s], key=Q[s].get)
        else:
            policy[s] = 0
    return policy


# ---------------------------------------------------------------------------
# Algorithm 1: Value Iteration (DP baseline)
# ---------------------------------------------------------------------------

def run_value_iteration(env: GridworldEnv, gamma: float = 0.95,
                       tol: float = 1e-8, max_iter: int = 1000,
                       **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Value Iteration for gridworld.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    t0 = time.time()
    n_states = env.num_states
    n_actions = env.num_actions

    V = np.zeros(n_states)
    policy_arr = np.zeros(n_states, dtype=int)

    iterations = 0
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        for si in range(n_states):
            s = env.index_to_state(si)
            if s == env.terminal:
                V_new[si] = 0.0
                continue

            best_val = -np.inf
            best_a = 0
            for a in range(n_actions):
                r = env.get_reward(s, a)
                ns = env.get_next_state(s, a)
                nsi = env.state_to_index(ns)
                val = r + gamma * V[nsi]
                if val > best_val:
                    best_val = val
                    best_a = a
            V_new[si] = best_val
            policy_arr[si] = best_a

        diff = np.max(np.abs(V_new - V))
        V = V_new
        iterations = iteration + 1

        if diff < tol:
            break

    # Convert to dict format
    policy = {}
    Q = {}
    for si in range(n_states):
        s = env.index_to_state(si)
        policy[s] = policy_arr[si]
        Q[s] = {}
        for a in range(n_actions):
            r = env.get_reward(s, a)
            ns = env.get_next_state(s, a)
            nsi = env.state_to_index(ns)
            Q[s][a] = r + gamma * V[nsi]

    wall_time = time.time() - t0

    # Evaluate
    mean_return, mean_steps = evaluate_policy(env, policy, n_episodes=100, horizon=100)

    metrics = AlgorithmMetrics(
        final_return=mean_return,
        final_steps=mean_steps,
        final_value_error=0.0,
        final_policy_agreement=1.0,
        states_visited=n_states,
        wall_time=wall_time,
        iterations=iterations
    )

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 2: Policy Iteration (DP baseline)
# ---------------------------------------------------------------------------

def run_policy_iteration(env: GridworldEnv, gamma: float = 0.95,
                        tol: float = 1e-8, max_iter: int = 100,
                        **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Policy Iteration for gridworld.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    t0 = time.time()
    n_states = env.num_states
    n_actions = env.num_actions

    # Initialize with arbitrary policy
    policy_arr = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    iterations = 0
    for iteration in range(max_iter):
        # Policy Evaluation
        for _ in range(1000):  # Inner loop for evaluation
            V_new = np.zeros(n_states)
            for si in range(n_states):
                s = env.index_to_state(si)
                if s == env.terminal:
                    V_new[si] = 0.0
                    continue
                a = policy_arr[si]
                r = env.get_reward(s, a)
                ns = env.get_next_state(s, a)
                nsi = env.state_to_index(ns)
                V_new[si] = r + gamma * V[nsi]

            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        # Policy Improvement
        policy_stable = True
        for si in range(n_states):
            s = env.index_to_state(si)
            if s == env.terminal:
                continue

            old_action = policy_arr[si]
            best_val = -np.inf
            best_a = 0
            for a in range(n_actions):
                r = env.get_reward(s, a)
                ns = env.get_next_state(s, a)
                nsi = env.state_to_index(ns)
                val = r + gamma * V[nsi]
                if val > best_val:
                    best_val = val
                    best_a = a

            policy_arr[si] = best_a
            if old_action != best_a:
                policy_stable = False

        iterations = iteration + 1
        if policy_stable:
            break

    # Convert to dict format
    policy = {}
    Q = {}
    for si in range(n_states):
        s = env.index_to_state(si)
        policy[s] = policy_arr[si]
        Q[s] = {}
        for a in range(n_actions):
            r = env.get_reward(s, a)
            ns = env.get_next_state(s, a)
            nsi = env.state_to_index(ns)
            Q[s][a] = r + gamma * V[nsi]

    wall_time = time.time() - t0

    # Evaluate
    mean_return, mean_steps = evaluate_policy(env, policy, n_episodes=100, horizon=100)

    metrics = AlgorithmMetrics(
        final_return=mean_return,
        final_steps=mean_steps,
        final_value_error=0.0,
        final_policy_agreement=1.0,
        states_visited=n_states,
        wall_time=wall_time,
        iterations=iterations
    )

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 3: Q-Learning (Tabular, Off-Policy TD)
# ---------------------------------------------------------------------------

def run_q_learning(env: GridworldEnv, gamma: float = 0.95,
                  num_episodes: int = 5000, horizon: int = 100,
                  alpha: float = 0.1, alpha_decay: float = 0.999,
                  epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                  epsilon_decay: float = 0.995,
                  eval_freq: int = 100, eval_episodes: int = 100,
                  V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                  seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Tabular Q-Learning with epsilon-greedy exploration.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions

    # Initialize Q-table
    Q = {s: {a: 0.0 for a in range(n_actions)} for s in env.get_all_states()}

    epsilon = epsilon_start
    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = max(Q[s], key=Q[s].get)

            ns, r, done = env.step(a)
            ep_return += r
            ep_length = t + 1

            # Q-Learning update (off-policy)
            if done:
                target = r
            else:
                target = r + gamma * max(Q[ns].values())

            Q[s][a] += alpha_current * (target - Q[s][a])

            s = ns
            if done:
                break

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        # Decay exploration and learning rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_current = max(0.01, alpha_current * alpha_decay)

        # Checkpoint evaluation
        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                val_err = compute_value_error(Q, V_optimal, env)
                metrics.checkpoint_value_errors.append(val_err)

            if optimal_policy is not None:
                agreement = compute_policy_agreement(policy, optimal_policy, env)
                metrics.checkpoint_policy_agreements.append(agreement)

    wall_time = time.time() - t0

    # Final evaluation
    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 4: SARSA (On-Policy TD)
# ---------------------------------------------------------------------------

def run_sarsa(env: GridworldEnv, gamma: float = 0.95,
             num_episodes: int = 5000, horizon: int = 100,
             alpha: float = 0.1, alpha_decay: float = 0.999,
             epsilon_start: float = 1.0, epsilon_end: float = 0.01,
             epsilon_decay: float = 0.995,
             eval_freq: int = 100, eval_episodes: int = 100,
             V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
             seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """SARSA: On-policy TD control with epsilon-greedy.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions

    # Initialize Q-table
    Q = {s: {a: 0.0 for a in range(n_actions)} for s in env.get_all_states()}

    def epsilon_greedy(state, eps):
        if np.random.random() < eps:
            return np.random.randint(n_actions)
        return max(Q[state], key=Q[state].get)

    epsilon = epsilon_start
    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        s = env.reset()
        a = epsilon_greedy(s, epsilon)
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            ns, r, done = env.step(a)
            ep_return += r
            ep_length = t + 1

            if done:
                # Terminal update
                Q[s][a] += alpha_current * (r - Q[s][a])
                break

            # Choose next action (on-policy)
            na = epsilon_greedy(ns, epsilon)

            # SARSA update
            target = r + gamma * Q[ns][na]
            Q[s][a] += alpha_current * (target - Q[s][a])

            s = ns
            a = na

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_current = max(0.01, alpha_current * alpha_decay)

        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 5: Expected SARSA
# ---------------------------------------------------------------------------

def run_expected_sarsa(env: GridworldEnv, gamma: float = 0.95,
                      num_episodes: int = 5000, horizon: int = 100,
                      alpha: float = 0.1, alpha_decay: float = 0.999,
                      epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                      epsilon_decay: float = 0.995,
                      eval_freq: int = 100, eval_episodes: int = 100,
                      V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                      seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Expected SARSA: Uses expected value under policy for update.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions

    Q = {s: {a: 0.0 for a in range(n_actions)} for s in env.get_all_states()}

    def expected_value(state, eps):
        """Compute expected Q-value under epsilon-greedy policy."""
        q_vals = Q[state]
        greedy_a = max(q_vals, key=q_vals.get)
        expected = 0.0
        for a in range(n_actions):
            if a == greedy_a:
                prob = (1 - eps) + eps / n_actions
            else:
                prob = eps / n_actions
            expected += prob * q_vals[a]
        return expected

    epsilon = epsilon_start
    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = max(Q[s], key=Q[s].get)

            ns, r, done = env.step(a)
            ep_return += r
            ep_length = t + 1

            # Expected SARSA update
            if done:
                target = r
            else:
                target = r + gamma * expected_value(ns, epsilon)

            Q[s][a] += alpha_current * (target - Q[s][a])

            s = ns
            if done:
                break

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_current = max(0.01, alpha_current * alpha_decay)

        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 6: Monte Carlo Control (First-Visit, Epsilon-Greedy)
# ---------------------------------------------------------------------------

def run_mc_control(env: GridworldEnv, gamma: float = 0.95,
                  num_episodes: int = 5000, horizon: int = 100,
                  epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                  epsilon_decay: float = 0.995,
                  eval_freq: int = 100, eval_episodes: int = 100,
                  V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                  seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """First-visit Monte Carlo Control with epsilon-greedy policy.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions

    # Initialize Q and visit counts
    Q = {s: {a: 0.0 for a in range(n_actions)} for s in env.get_all_states()}
    N = {s: {a: 0 for a in range(n_actions)} for s in env.get_all_states()}

    epsilon = epsilon_start
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        # Generate episode
        episode = []
        s = env.reset()

        for t in range(horizon):
            states_visited.add(s)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = max(Q[s], key=Q[s].get)

            ns, r, done = env.step(a)
            episode.append((s, a, r))

            if done:
                break
            s = ns

        # Compute returns and update Q (first-visit)
        G = 0.0
        visited = set()
        ep_return = sum(r for _, _, r in episode)
        ep_length = len(episode)

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G

            if (s, a) not in visited:
                visited.add((s, a))
                N[s][a] += 1
                # Incremental mean update
                Q[s][a] += (G - Q[s][a]) / N[s][a]

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 7: SARSA(λ) - Eligibility Traces
# ---------------------------------------------------------------------------

def run_sarsa_lambda(env: GridworldEnv, gamma: float = 0.95,
                    lmbda: float = 0.9,
                    num_episodes: int = 5000, horizon: int = 100,
                    alpha: float = 0.1, alpha_decay: float = 0.999,
                    epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                    epsilon_decay: float = 0.995,
                    eval_freq: int = 100, eval_episodes: int = 100,
                    V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                    seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """SARSA(λ) with accumulating eligibility traces.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions
    all_states = env.get_all_states()

    Q = {s: {a: 0.0 for a in range(n_actions)} for s in all_states}

    def epsilon_greedy(state, eps):
        if np.random.random() < eps:
            return np.random.randint(n_actions)
        return max(Q[state], key=Q[state].get)

    epsilon = epsilon_start
    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        # Initialize eligibility traces
        E = {s: {a: 0.0 for a in range(n_actions)} for s in all_states}

        s = env.reset()
        a = epsilon_greedy(s, epsilon)
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            ns, r, done = env.step(a)
            ep_return += r
            ep_length = t + 1

            if done:
                # TD error with terminal state
                delta = r - Q[s][a]
                E[s][a] += 1.0

                # Update all Q values
                for state in all_states:
                    for action in range(n_actions):
                        Q[state][action] += alpha_current * delta * E[state][action]
                break

            na = epsilon_greedy(ns, epsilon)

            # TD error
            delta = r + gamma * Q[ns][na] - Q[s][a]

            # Accumulating traces
            E[s][a] += 1.0

            # Update all Q values and decay traces
            for state in all_states:
                for action in range(n_actions):
                    Q[state][action] += alpha_current * delta * E[state][action]
                    E[state][action] *= gamma * lmbda

            s = ns
            a = na

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_current = max(0.01, alpha_current * alpha_decay)

        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 8: Watkins's Q(λ) - Off-policy with Eligibility Traces
# ---------------------------------------------------------------------------

def run_q_lambda(env: GridworldEnv, gamma: float = 0.95,
                lmbda: float = 0.9,
                num_episodes: int = 5000, horizon: int = 100,
                alpha: float = 0.1, alpha_decay: float = 0.999,
                epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                eval_freq: int = 100, eval_episodes: int = 100,
                V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Watkins's Q(λ): Off-policy TD with eligibility traces.

    Traces are cut when a non-greedy action is taken.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions
    all_states = env.get_all_states()

    Q = {s: {a: 0.0 for a in range(n_actions)} for s in all_states}

    epsilon = epsilon_start
    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        E = {s: {a: 0.0 for a in range(n_actions)} for s in all_states}

        s = env.reset()
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            # Epsilon-greedy action selection
            greedy_a = max(Q[s], key=Q[s].get)
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = greedy_a

            ns, r, done = env.step(a)
            ep_return += r
            ep_length = t + 1

            # Greedy action for next state (for Q-learning target)
            greedy_na = max(Q[ns], key=Q[ns].get)

            if done:
                delta = r - Q[s][a]
            else:
                delta = r + gamma * Q[ns][greedy_na] - Q[s][a]

            E[s][a] += 1.0

            # Update all Q values
            for state in all_states:
                for action in range(n_actions):
                    Q[state][action] += alpha_current * delta * E[state][action]

            if done:
                break

            # Watkins's Q(λ): cut traces if non-greedy action
            if a == greedy_a:
                for state in all_states:
                    for action in range(n_actions):
                        E[state][action] *= gamma * lmbda
            else:
                # Reset all traces to zero
                for state in all_states:
                    for action in range(n_actions):
                        E[state][action] = 0.0

            s = ns

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha_current = max(0.01, alpha_current * alpha_decay)

        if (ep + 1) % eval_freq == 0:
            policy = extract_policy_from_Q(Q, env)
            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    policy = extract_policy_from_Q(Q, env)
    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 9: REINFORCE (Policy Gradient)
# ---------------------------------------------------------------------------

def run_reinforce(env: GridworldEnv, gamma: float = 0.95,
                 num_episodes: int = 5000, horizon: int = 100,
                 alpha: float = 0.01, alpha_decay: float = 0.9995,
                 baseline: bool = True,
                 eval_freq: int = 100, eval_episodes: int = 100,
                 V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
                 seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """REINFORCE policy gradient with optional baseline.

    Uses softmax policy parameterization with tabular preferences.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value (estimated from returns)
        metrics: AlgorithmMetrics
    """
    np.random.seed(seed)
    t0 = time.time()

    n_actions = env.num_actions
    all_states = env.get_all_states()

    # Policy parameters (preferences)
    theta = {s: np.zeros(n_actions) for s in all_states}

    # Baseline (state value estimate)
    V_baseline = {s: 0.0 for s in all_states}
    V_count = {s: 0 for s in all_states}

    alpha_current = alpha
    states_visited = set()

    metrics = AlgorithmMetrics()

    def softmax_policy(state):
        """Return action probabilities for state."""
        prefs = theta[state]
        prefs = prefs - np.max(prefs)  # Numerical stability
        exp_prefs = np.exp(prefs)
        return exp_prefs / np.sum(exp_prefs)

    def sample_action(state):
        """Sample action from softmax policy."""
        probs = softmax_policy(state)
        return np.random.choice(n_actions, p=probs)

    for ep in range(num_episodes):
        # Generate episode
        episode = []
        s = env.reset()

        for t in range(horizon):
            states_visited.add(s)
            a = sample_action(s)
            ns, r, done = env.step(a)
            episode.append((s, a, r))
            if done:
                break
            s = ns

        ep_return = sum(r for _, _, r in episode)
        ep_length = len(episode)

        # Compute returns for each timestep
        returns = []
        G = 0.0
        for _, _, r in reversed(episode):
            G = r + gamma * G
            returns.insert(0, G)

        # Update baseline and policy
        for t, (s, a, r) in enumerate(episode):
            G_t = returns[t]

            # Update baseline
            V_count[s] += 1
            V_baseline[s] += (G_t - V_baseline[s]) / V_count[s]

            # Policy gradient update
            if baseline:
                advantage = G_t - V_baseline[s]
            else:
                advantage = G_t

            probs = softmax_policy(s)

            # Gradient of log softmax: δ_a - π(a|s)
            grad = -probs.copy()
            grad[a] += 1.0

            theta[s] += alpha_current * (gamma ** t) * advantage * grad

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        alpha_current = max(0.001, alpha_current * alpha_decay)

        if (ep + 1) % eval_freq == 0:
            # Extract greedy policy for evaluation
            policy = {}
            for s in all_states:
                policy[s] = np.argmax(theta[s])

            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            # Construct Q estimate from policy for value error
            Q_est = {s: {a: theta[s][a] for a in range(n_actions)} for s in all_states}

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q_est, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    # Final policy and Q-table
    policy = {}
    Q = {}
    for s in all_states:
        policy[s] = np.argmax(theta[s])
        Q[s] = {a: theta[s][a] for a in range(n_actions)}

    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm 10: DQN (Deep Q-Network)
# ---------------------------------------------------------------------------

def run_dqn(env: GridworldEnv, gamma: float = 0.95,
           num_episodes: int = 5000, horizon: int = 100,
           lr: float = 1e-3, replay_size: int = 10000, batch_size: int = 64,
           epsilon_start: float = 1.0, epsilon_end: float = 0.01,
           epsilon_decay: float = 0.995, target_update_freq: int = 50,
           hidden_dim: int = 64,
           eval_freq: int = 100, eval_episodes: int = 100,
           V_optimal: np.ndarray = None, optimal_policy: np.ndarray = None,
           seed: int = 42, **kwargs) -> Tuple[Dict, Dict, AlgorithmMetrics]:
    """Deep Q-Network for gridworld.

    Returns:
        policy: Dict[state] -> action
        Q: Dict[state][action] -> value (extracted from network)
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

    n_actions = env.num_actions
    state_dim = 2  # (row, col) normalized
    all_states = env.get_all_states()

    # Neural network
    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )

        def forward(self, x):
            return self.net(x)

    q_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = deque(maxlen=replay_size)

    epsilon = epsilon_start
    states_visited = set()

    metrics = AlgorithmMetrics()

    for ep in range(num_episodes):
        s = env.reset()
        s_feat = env.state_to_features(s)
        ep_return = 0.0
        ep_length = 0

        for t in range(horizon):
            states_visited.add(s)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0))
                    a = q_vals.argmax(dim=1).item()

            ns, r, done = env.step(a)
            ns_feat = env.state_to_features(ns)
            buffer.append((s_feat, a, r, ns_feat, float(done)))

            ep_return += r
            ep_length = t + 1

            # Train
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)

                s_b = torch.FloatTensor(np.array(s_b))
                a_b = torch.LongTensor(a_b)
                r_b = torch.FloatTensor(r_b)
                ns_b = torch.FloatTensor(np.array(ns_b))
                d_b = torch.FloatTensor(d_b)

                q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_b).max(dim=1)[0]
                    target = r_b + gamma * next_q * (1.0 - d_b)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            s = ns
            s_feat = ns_feat
            if done:
                break

        metrics.episode_returns.append(ep_return)
        metrics.episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % eval_freq == 0:
            # Extract policy from network
            q_net.eval()
            policy = {}
            Q = {}
            for s in all_states:
                s_feat = env.state_to_features(s)
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0)).squeeze(0).numpy()
                policy[s] = np.argmax(q_vals)
                Q[s] = {a: float(q_vals[a]) for a in range(n_actions)}
            q_net.train()

            mean_ret, mean_steps = evaluate_policy(env, policy, eval_episodes, horizon)
            metrics.checkpoint_episodes.append(ep + 1)
            metrics.checkpoint_returns.append(mean_ret)

            if V_optimal is not None:
                metrics.checkpoint_value_errors.append(compute_value_error(Q, V_optimal, env))
            if optimal_policy is not None:
                metrics.checkpoint_policy_agreements.append(
                    compute_policy_agreement(policy, optimal_policy, env))

    wall_time = time.time() - t0

    # Final policy extraction
    q_net.eval()
    policy = {}
    Q = {}
    for s in all_states:
        s_feat = env.state_to_features(s)
        with torch.no_grad():
            q_vals = q_net(torch.FloatTensor(s_feat).unsqueeze(0)).squeeze(0).numpy()
        policy[s] = np.argmax(q_vals)
        Q[s] = {a: float(q_vals[a]) for a in range(n_actions)}

    final_return, final_steps = evaluate_policy(env, policy, eval_episodes, horizon)

    metrics.final_return = final_return
    metrics.final_steps = final_steps
    metrics.states_visited = len(states_visited)
    metrics.wall_time = wall_time

    if V_optimal is not None:
        metrics.final_value_error = compute_value_error(Q, V_optimal, env)
    if optimal_policy is not None:
        metrics.final_policy_agreement = compute_policy_agreement(policy, optimal_policy, env)

    return policy, Q, metrics


# ---------------------------------------------------------------------------
# Algorithm Registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    'VI': run_value_iteration,
    'PI': run_policy_iteration,
    'Q-Learning': run_q_learning,
    'SARSA': run_sarsa,
    'Expected SARSA': run_expected_sarsa,
    'MC Control': run_mc_control,
    'SARSA(λ)': run_sarsa_lambda,
    'Q(λ)': run_q_lambda,
    'REINFORCE': run_reinforce,
    'DQN': run_dqn,
}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Gridworld Algorithms Module Test")
    print("=" * 50)

    env = GridworldEnv(N=5)
    print(f"Environment: {env.N}x{env.N} gridworld ({env.num_states} states)")

    # Run VI to get optimal
    print("\nRunning Value Iteration...")
    policy_vi, Q_vi, metrics_vi = run_value_iteration(env)
    print(f"  VI: return={metrics_vi.final_return:.3f}, steps={metrics_vi.final_steps:.1f}, "
          f"iterations={metrics_vi.iterations}, time={metrics_vi.wall_time:.3f}s")

    # Convert VI results to arrays for comparison
    V_optimal = np.array([max(Q_vi[env.index_to_state(i)].values())
                          for i in range(env.num_states)])
    optimal_policy = np.array([policy_vi[env.index_to_state(i)]
                               for i in range(env.num_states)])

    # Test Q-Learning
    print("\nRunning Q-Learning (1000 episodes)...")
    policy_ql, Q_ql, metrics_ql = run_q_learning(
        env, num_episodes=1000, seed=42,
        V_optimal=V_optimal, optimal_policy=optimal_policy
    )
    print(f"  Q-Learning: return={metrics_ql.final_return:.3f}, "
          f"agreement={metrics_ql.final_policy_agreement:.1%}, "
          f"value_error={metrics_ql.final_value_error:.3f}, "
          f"time={metrics_ql.wall_time:.3f}s")

    print("\nModule test complete.")
