# Dynamic Cournot Duopoly with Stochastic Demand — Chapter 5, RL in Games.
# Compares five MARL methods against exact MPE in a quantity-setting stochastic game.

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_STATES = 3                         # demand levels: Low, Medium, High
DEMAND_INTERCEPTS = np.array([8.0, 11.0, 14.0])
COST = 2.0                           # constant marginal cost
N_ACTIONS = 6                        # quantities 0..5
ACTIONS = np.arange(N_ACTIONS)
GAMMA = 0.95                         # discount factor
N_EPISODES = 20_000
N_SEEDS = 15
EVAL_EVERY = 500
EPISODE_LENGTH = 30                  # steps per episode

# Demand state transition matrix (persistent)
P_TRANSITION = np.array([
    [0.7, 0.25, 0.05],
    [0.15, 0.70, 0.15],
    [0.05, 0.25, 0.70],
])

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

METHOD_NAMES = ["IQL", "Nash-Q", "WoLF-PHC", "REINFORCE", "Fictitious Play"]
COLORS = {"IQL": "#1f77b4", "Nash-Q": "#ff7f0e", "WoLF-PHC": "#2ca02c",
          "REINFORCE": "#d62728", "Fictitious Play": "#9467bd"}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class DynamicCournotEnv:
    """Dynamic Cournot duopoly with stochastic demand as a Markov game."""

    def __init__(self):
        self.state = None

    def reset(self, rng):
        self.state = rng.choice(N_STATES)
        return self.state

    def step(self, q1, q2, rng):
        r1, r2 = self.profits(self.state, q1, q2)
        next_state = rng.choice(N_STATES, p=P_TRANSITION[self.state])
        self.state = next_state
        return next_state, r1, r2

    @staticmethod
    def profits(s, q1, q2):
        Q = q1 + q2
        a = DEMAND_INTERCEPTS[s]
        price = max(0.0, a - Q)
        r1 = price * q1 - COST * q1
        r2 = price * q2 - COST * q2
        return r1, r2


# ---------------------------------------------------------------------------
# Exact MPE via iterated best response + value iteration
# ---------------------------------------------------------------------------
def compute_mpe(tol=1e-8, max_iter=500):
    """
    Compute Markov Perfect Equilibrium strategies via iterated best response.
    Returns symmetric mixed-strategy policies pi[s, a] for each firm.
    Uses logit smoothing (temperature annealing) for convergence.
    """
    # Initialize uniform mixed strategies
    pi1 = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS
    pi2 = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    # Precompute stage-game profits for all (s, a1, a2)
    R = np.zeros((N_STATES, N_ACTIONS, N_ACTIONS, 2))
    for s in range(N_STATES):
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                r1, r2 = DynamicCournotEnv.profits(s, a1, a2)
                R[s, a1, a2, 0] = r1
                R[s, a1, a2, 1] = r2

    temperature = 1.0
    min_temp = 0.01

    for iteration in range(max_iter):
        pi1_old = pi1.copy()
        pi2_old = pi2.copy()

        # Solve firm 1's best response given pi2
        pi1 = _best_response_vi(R[:, :, :, 0], pi2, temperature, tol)
        # Solve firm 2's best response given pi1 (symmetric game, swap axes)
        R2_swapped = R[:, :, :, 1].transpose(0, 2, 1)  # R2(s, a2, a1)
        pi2 = _best_response_vi(R2_swapped, pi1, temperature, tol)

        diff = max(np.max(np.abs(pi1 - pi1_old)), np.max(np.abs(pi2 - pi2_old)))
        if diff < tol and temperature <= min_temp + 1e-12:
            break
        temperature = max(min_temp, temperature * 0.95)

    return pi1, pi2, R


def _best_response_vi(R_i, pi_opp, temperature, tol):
    """
    Value iteration for firm i given opponent's mixed strategy pi_opp.
    R_i[s, a_i, a_opp] = firm i's profit.
    Returns logit best-response policy pi_i[s, a_i].
    """
    V = np.zeros(N_STATES)

    for _ in range(5000):
        V_new = np.zeros(N_STATES)
        Q = np.zeros((N_STATES, N_ACTIONS))

        for s in range(N_STATES):
            for a_i in range(N_ACTIONS):
                # Expected immediate reward under opponent's strategy
                exp_r = np.dot(R_i[s, a_i, :], pi_opp[s, :])
                # Expected continuation
                exp_V = np.dot(P_TRANSITION[s], V)
                Q[s, a_i] = exp_r + GAMMA * exp_V

            # Logit (softmax) choice
            logits = Q[s, :] / max(temperature, 1e-10)
            logits -= logits.max()
            exp_logits = np.exp(logits)
            V_new[s] = temperature * np.log(exp_logits.sum()) + logits.max() * temperature
            # More precise: V = temp * log(sum(exp(Q/temp)))
            # Using the shifted version for stability:
            V_new[s] = max(temperature, 1e-10) * (logits.max() + np.log(exp_logits.sum()))
            # Actually recompute cleanly:
            raw = Q[s, :] / max(temperature, 1e-10)
            mx = raw.max()
            V_new[s] = max(temperature, 1e-10) * (mx + np.log(np.exp(raw - mx).sum()))

        if np.max(np.abs(V_new - V)) < tol * 0.01:
            break
        V = V_new

    # Extract policy
    pi = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a_i in range(N_ACTIONS):
            exp_r = np.dot(R_i[s, a_i, :], pi_opp[s, :])
            exp_V = np.dot(P_TRANSITION[s], V)
            Q[s, a_i] = exp_r + GAMMA * exp_V
        logits = Q[s, :] / max(temperature, 1e-10)
        logits -= logits.max()
        pi[s, :] = np.exp(logits) / np.exp(logits).sum()

    return pi


# ---------------------------------------------------------------------------
# Compute greedy (deterministic) MPE for evaluation
# ---------------------------------------------------------------------------
def compute_mpe_deterministic(tol=1e-10, max_iter=1000):
    """
    Compute deterministic MPE via iterated best response with value iteration.
    Returns deterministic policies as probability vectors (one-hot).
    """
    # Precompute stage-game profits
    R = np.zeros((N_STATES, N_ACTIONS, N_ACTIONS, 2))
    for s in range(N_STATES):
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                r1, r2 = DynamicCournotEnv.profits(s, a1, a2)
                R[s, a1, a2, 0] = r1
                R[s, a1, a2, 1] = r2

    # Initialize: each firm plays quantity 3 (near static Nash)
    policy1 = np.full(N_STATES, 3, dtype=int)
    policy2 = np.full(N_STATES, 3, dtype=int)

    for iteration in range(max_iter):
        p1_old = policy1.copy()
        p2_old = policy2.copy()

        # Firm 1 best response given firm 2's deterministic policy
        policy1 = _det_best_response(R[:, :, :, 0], policy2)
        # Firm 2 best response given firm 1's deterministic policy
        R2_swap = R[:, :, :, 1].transpose(0, 2, 1)
        policy2 = _det_best_response(R2_swap, policy1)

        if np.array_equal(policy1, p1_old) and np.array_equal(policy2, p2_old):
            break

    # Convert to probability vectors
    pi1 = np.zeros((N_STATES, N_ACTIONS))
    pi2 = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        pi1[s, policy1[s]] = 1.0
        pi2[s, policy2[s]] = 1.0

    return pi1, pi2, R, policy1, policy2


def _det_best_response(R_i, opp_policy):
    """Deterministic best response via value iteration given opponent's pure strategy."""
    V = np.zeros(N_STATES)

    for _ in range(5000):
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            a_opp = opp_policy[s]
            best_val = -np.inf
            for a_i in range(N_ACTIONS):
                q_val = R_i[s, a_i, a_opp] + GAMMA * np.dot(P_TRANSITION[s], V)
                if q_val > best_val:
                    best_val = q_val
            V_new[s] = best_val
        if np.max(np.abs(V_new - V)) < 1e-12:
            break
        V = V_new

    # Extract greedy policy
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        a_opp = opp_policy[s]
        best_val = -np.inf
        for a_i in range(N_ACTIONS):
            q_val = R_i[s, a_i, a_opp] + GAMMA * np.dot(P_TRANSITION[s], V)
            if q_val > best_val:
                best_val = q_val
                policy[s] = a_i
    return policy


# ---------------------------------------------------------------------------
# Policy distance metric
# ---------------------------------------------------------------------------
def policy_distance_from_mpe(pi_learned, mpe_pi):
    """Total variation distance averaged over states."""
    tv = 0.0
    for s in range(N_STATES):
        tv += 0.5 * np.sum(np.abs(pi_learned[s] - mpe_pi[s]))
    return tv / N_STATES


def extract_greedy_policy(Q):
    """Extract greedy policy from Q-table Q[s, a] -> pi[s, a] (one-hot)."""
    pi = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        best = np.argmax(Q[s])
        pi[s, best] = 1.0
    return pi


# ---------------------------------------------------------------------------
# MARL Method 1: Independent Q-Learning (IQL)
# ---------------------------------------------------------------------------
def run_iql(mpe_pi, seed):
    rng = np.random.RandomState(seed)
    env = DynamicCournotEnv()

    Q1 = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))

    alpha0 = 0.1
    eps0 = 1.0
    eps_decay = 0.9999

    distances = []
    rewards_log = []
    eps = eps0

    for ep in range(N_EPISODES):
        s = env.reset(rng)
        ep_reward = 0.0

        for t in range(EPISODE_LENGTH):
            # Epsilon-greedy for both firms
            if rng.uniform() < eps:
                a1 = rng.randint(N_ACTIONS)
            else:
                a1 = np.argmax(Q1[s])

            if rng.uniform() < eps:
                a2 = rng.randint(N_ACTIONS)
            else:
                a2 = np.argmax(Q2[s])

            s_next, r1, r2 = env.step(a1, a2, rng)
            alpha = alpha0 / (1 + ep * 0.0001)

            Q1[s, a1] += alpha * (r1 + GAMMA * np.max(Q1[s_next]) - Q1[s, a1])
            Q2[s, a2] += alpha * (r2 + GAMMA * np.max(Q2[s_next]) - Q2[s, a2])

            ep_reward += (r1 + r2) / 2.0
            s = s_next

        eps *= eps_decay

        if (ep + 1) % EVAL_EVERY == 0:
            pi1 = extract_greedy_policy(Q1)
            pi2 = extract_greedy_policy(Q2)
            d = (policy_distance_from_mpe(pi1, mpe_pi) +
                 policy_distance_from_mpe(pi2, mpe_pi)) / 2.0
            distances.append(d)
            rewards_log.append(ep_reward / EPISODE_LENGTH)

    return np.array(distances), np.array(rewards_log), Q1, Q2


# ---------------------------------------------------------------------------
# MARL Method 2: Nash-Q Learning
# ---------------------------------------------------------------------------
def _solve_nash_pure(R1_matrix, R2_matrix):
    """
    Find a pure-strategy Nash equilibrium of a bimatrix game (vectorized).
    R1_matrix[a1, a2], R2_matrix[a1, a2].
    Returns strategy pair as probability vectors.
    Falls back to maximin if no pure NE exists.
    """
    n1, n2 = R1_matrix.shape
    # Best response masks
    br1 = (R1_matrix == R1_matrix.max(axis=0, keepdims=True))  # BR of player 1 for each a2
    br2 = (R2_matrix == R2_matrix.max(axis=1, keepdims=True))  # BR of player 2 for each a1
    ne_mask = br1 & br2
    indices = np.argwhere(ne_mask)

    if len(indices) > 0:
        best_a1, best_a2 = indices[0]
    else:
        best_a1 = np.argmax(np.min(R1_matrix, axis=1))
        best_a2 = np.argmax(np.min(R2_matrix, axis=0))

    pi1 = np.zeros(n1)
    pi2 = np.zeros(n2)
    pi1[best_a1] = 1.0
    pi2[best_a2] = 1.0
    return pi1, pi2


def _solve_nash_pure_idx(R1_matrix, R2_matrix):
    """Fast version returning action indices instead of probability vectors."""
    br1 = (R1_matrix == R1_matrix.max(axis=0, keepdims=True))
    br2 = (R2_matrix == R2_matrix.max(axis=1, keepdims=True))
    ne_mask = br1 & br2
    indices = np.argwhere(ne_mask)
    if len(indices) > 0:
        return indices[0, 0], indices[0, 1]
    return (np.argmax(np.min(R1_matrix, axis=1)),
            np.argmax(np.min(R2_matrix, axis=0)))


def run_nash_q(mpe_pi, seed):
    rng = np.random.RandomState(seed)
    env = DynamicCournotEnv()

    # Joint Q-tables: Q_i(s, a1, a2)
    Q1 = np.zeros((N_STATES, N_ACTIONS, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS, N_ACTIONS))

    alpha0 = 0.1
    eps0 = 1.0
    eps_decay = 0.9999

    distances = []
    rewards_log = []
    eps = eps0

    for ep in range(N_EPISODES):
        s = env.reset(rng)
        ep_reward = 0.0

        for t in range(EPISODE_LENGTH):
            if rng.uniform() < eps:
                a1 = rng.randint(N_ACTIONS)
            else:
                a1 = np.argmax(np.sum(Q1[s] * 1.0, axis=1))  # marginalize
            if rng.uniform() < eps:
                a2 = rng.randint(N_ACTIONS)
            else:
                a2 = np.argmax(np.sum(Q2[s] * 1.0, axis=0))

            s_next, r1, r2 = env.step(a1, a2, rng)
            alpha = alpha0 / (1 + ep * 0.0001)

            # Compute Nash value of next state
            ne_a1, ne_a2 = _solve_nash_pure_idx(Q1[s_next], Q2[s_next])
            nash_v1 = Q1[s_next, ne_a1, ne_a2]
            nash_v2 = Q2[s_next, ne_a1, ne_a2]

            Q1[s, a1, a2] += alpha * (r1 + GAMMA * nash_v1 - Q1[s, a1, a2])
            Q2[s, a1, a2] += alpha * (r2 + GAMMA * nash_v2 - Q2[s, a1, a2])

            ep_reward += (r1 + r2) / 2.0
            s = s_next

        eps *= eps_decay

        if (ep + 1) % EVAL_EVERY == 0:
            # Extract marginal policies
            pi1 = np.zeros((N_STATES, N_ACTIONS))
            pi2 = np.zeros((N_STATES, N_ACTIONS))
            for ss in range(N_STATES):
                ne1, ne2 = _solve_nash_pure(Q1[ss], Q2[ss])
                pi1[ss] = ne1
                pi2[ss] = ne2
            d = (policy_distance_from_mpe(pi1, mpe_pi) +
                 policy_distance_from_mpe(pi2, mpe_pi)) / 2.0
            distances.append(d)
            rewards_log.append(ep_reward / EPISODE_LENGTH)

    return np.array(distances), np.array(rewards_log), Q1, Q2


# ---------------------------------------------------------------------------
# MARL Method 3: WoLF-PHC (Win or Learn Fast - Policy Hill Climbing)
# ---------------------------------------------------------------------------
def run_wolf_phc(mpe_pi, seed):
    rng = np.random.RandomState(seed)
    env = DynamicCournotEnv()

    Q1 = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))
    pi1 = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS
    pi2 = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS
    pi1_avg = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS
    pi2_avg = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    C = np.zeros((N_STATES, 2), dtype=int)  # visit counts per state per player

    alpha0 = 0.1
    delta_w = 0.002   # learning rate when winning
    delta_l = 0.02    # learning rate when losing (faster)

    distances = []
    rewards_log = []

    for ep in range(N_EPISODES):
        s = env.reset(rng)
        ep_reward = 0.0

        for t in range(EPISODE_LENGTH):
            # Sample actions from policies
            a1 = rng.choice(N_ACTIONS, p=pi1[s])
            a2 = rng.choice(N_ACTIONS, p=pi2[s])

            s_next, r1, r2 = env.step(a1, a2, rng)
            alpha = alpha0 / (1 + ep * 0.0001)

            # Q-learning update
            Q1[s, a1] += alpha * (r1 + GAMMA * np.max(Q1[s_next]) - Q1[s, a1])
            Q2[s, a2] += alpha * (r2 + GAMMA * np.max(Q2[s_next]) - Q2[s, a2])

            # Update average policies
            C[s, 0] += 1
            C[s, 1] += 1
            pi1_avg[s] += (pi1[s] - pi1_avg[s]) / C[s, 0]
            pi2_avg[s] += (pi2[s] - pi2_avg[s]) / C[s, 1]

            # WoLF policy update for firm 1
            val_current_1 = np.dot(pi1[s], Q1[s])
            val_avg_1 = np.dot(pi1_avg[s], Q1[s])
            delta1 = delta_w if val_current_1 > val_avg_1 else delta_l

            best_a1 = np.argmax(Q1[s])
            for a in range(N_ACTIONS):
                if a == best_a1:
                    pi1[s, a] = min(1.0, pi1[s, a] + delta1)
                else:
                    pi1[s, a] = max(0.0, pi1[s, a] - delta1 / (N_ACTIONS - 1))
            pi1[s] = np.maximum(pi1[s], 1e-10)
            pi1[s] /= pi1[s].sum()

            # WoLF policy update for firm 2
            val_current_2 = np.dot(pi2[s], Q2[s])
            val_avg_2 = np.dot(pi2_avg[s], Q2[s])
            delta2 = delta_w if val_current_2 > val_avg_2 else delta_l

            best_a2 = np.argmax(Q2[s])
            for a in range(N_ACTIONS):
                if a == best_a2:
                    pi2[s, a] = min(1.0, pi2[s, a] + delta2)
                else:
                    pi2[s, a] = max(0.0, pi2[s, a] - delta2 / (N_ACTIONS - 1))
            pi2[s] = np.maximum(pi2[s], 1e-10)
            pi2[s] /= pi2[s].sum()

            ep_reward += (r1 + r2) / 2.0
            s = s_next

        if (ep + 1) % EVAL_EVERY == 0:
            d = (policy_distance_from_mpe(pi1, mpe_pi) +
                 policy_distance_from_mpe(pi2, mpe_pi)) / 2.0
            distances.append(d)
            rewards_log.append(ep_reward / EPISODE_LENGTH)

    return np.array(distances), np.array(rewards_log), pi1, pi2


# ---------------------------------------------------------------------------
# MARL Method 4: REINFORCE (tabular softmax policy gradient)
# ---------------------------------------------------------------------------
def run_reinforce(mpe_pi, seed):
    rng = np.random.RandomState(seed)
    env = DynamicCournotEnv()

    # Tabular logits
    theta1 = np.zeros((N_STATES, N_ACTIONS))
    theta2 = np.zeros((N_STATES, N_ACTIONS))

    lr = 0.01
    baseline1 = np.zeros(N_STATES)
    baseline2 = np.zeros(N_STATES)
    baseline_count = np.zeros(N_STATES) + 1e-8

    distances = []
    rewards_log = []

    def softmax(logits):
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()

    for ep in range(N_EPISODES):
        s = env.reset(rng)

        states = []
        actions1 = []
        actions2 = []
        rewards1 = []
        rewards2 = []

        for t in range(EPISODE_LENGTH):
            pi1_s = softmax(theta1[s])
            pi2_s = softmax(theta2[s])
            a1 = rng.choice(N_ACTIONS, p=pi1_s)
            a2 = rng.choice(N_ACTIONS, p=pi2_s)

            s_next, r1, r2 = env.step(a1, a2, rng)

            states.append(s)
            actions1.append(a1)
            actions2.append(a2)
            rewards1.append(r1)
            rewards2.append(r2)
            s = s_next

        # Compute discounted returns
        G1 = np.zeros(EPISODE_LENGTH)
        G2 = np.zeros(EPISODE_LENGTH)
        g1, g2 = 0.0, 0.0
        for t in reversed(range(EPISODE_LENGTH)):
            g1 = rewards1[t] + GAMMA * g1
            g2 = rewards2[t] + GAMMA * g2
            G1[t] = g1
            G2[t] = g2

        # Policy gradient update
        for t in range(EPISODE_LENGTH):
            s_t = states[t]
            a1_t = actions1[t]
            a2_t = actions2[t]

            # Update baselines
            baseline_count[s_t] += 1
            baseline1[s_t] += (G1[t] - baseline1[s_t]) / baseline_count[s_t]
            baseline2[s_t] += (G2[t] - baseline2[s_t]) / baseline_count[s_t]

            adv1 = G1[t] - baseline1[s_t]
            adv2 = G2[t] - baseline2[s_t]

            pi1_s = softmax(theta1[s_t])
            pi2_s = softmax(theta2[s_t])

            # Gradient of log pi: e_{a} - pi
            grad1 = -pi1_s.copy()
            grad1[a1_t] += 1.0
            grad2 = -pi2_s.copy()
            grad2[a2_t] += 1.0

            effective_lr = lr / (1 + ep * 0.00005)
            theta1[s_t] += effective_lr * adv1 * grad1
            theta2[s_t] += effective_lr * adv2 * grad2

        if (ep + 1) % EVAL_EVERY == 0:
            pi1 = np.zeros((N_STATES, N_ACTIONS))
            pi2 = np.zeros((N_STATES, N_ACTIONS))
            for ss in range(N_STATES):
                pi1[ss] = softmax(theta1[ss])
                pi2[ss] = softmax(theta2[ss])
            d = (policy_distance_from_mpe(pi1, mpe_pi) +
                 policy_distance_from_mpe(pi2, mpe_pi)) / 2.0
            distances.append(d)
            avg_r = (np.mean(rewards1) + np.mean(rewards2)) / 2.0
            rewards_log.append(avg_r)

    # Final policies
    pi1_final = np.zeros((N_STATES, N_ACTIONS))
    pi2_final = np.zeros((N_STATES, N_ACTIONS))
    for ss in range(N_STATES):
        pi1_final[ss] = softmax(theta1[ss])
        pi2_final[ss] = softmax(theta2[ss])

    return np.array(distances), np.array(rewards_log), pi1_final, pi2_final


# ---------------------------------------------------------------------------
# MARL Method 5: Fictitious Play with Q-value estimation
# ---------------------------------------------------------------------------
def run_fictitious_play(mpe_pi, seed):
    rng = np.random.RandomState(seed)
    env = DynamicCournotEnv()

    # Empirical opponent action counts per state
    counts1 = np.ones((N_STATES, N_ACTIONS))  # firm 1's belief about firm 2
    counts2 = np.ones((N_STATES, N_ACTIONS))  # firm 2's belief about firm 1

    Q1 = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))

    alpha0 = 0.1

    distances = []
    rewards_log = []

    for ep in range(N_EPISODES):
        s = env.reset(rng)
        ep_reward = 0.0

        for t in range(EPISODE_LENGTH):
            # Best respond to empirical frequencies
            a1 = np.argmax(Q1[s])
            a2 = np.argmax(Q2[s])

            # Add exploration noise
            if rng.uniform() < 0.05 / (1 + ep * 0.0001):
                a1 = rng.randint(N_ACTIONS)
            if rng.uniform() < 0.05 / (1 + ep * 0.0001):
                a2 = rng.randint(N_ACTIONS)

            s_next, r1, r2 = env.step(a1, a2, rng)
            alpha = alpha0 / (1 + ep * 0.0001)

            # Update beliefs (opponent action counts)
            counts1[s, a2] += 1
            counts2[s, a1] += 1

            # Q-value update (expected reward against empirical distribution)
            Q1[s, a1] += alpha * (r1 + GAMMA * np.max(Q1[s_next]) - Q1[s, a1])
            Q2[s, a2] += alpha * (r2 + GAMMA * np.max(Q2[s_next]) - Q2[s, a2])

            ep_reward += (r1 + r2) / 2.0
            s = s_next

        if (ep + 1) % EVAL_EVERY == 0:
            pi1 = extract_greedy_policy(Q1)
            pi2 = extract_greedy_policy(Q2)
            d = (policy_distance_from_mpe(pi1, mpe_pi) +
                 policy_distance_from_mpe(pi2, mpe_pi)) / 2.0
            distances.append(d)
            rewards_log.append(ep_reward / EPISODE_LENGTH)

    return np.array(distances), np.array(rewards_log), Q1, Q2


# ---------------------------------------------------------------------------
# Compute MPE reward benchmark
# ---------------------------------------------------------------------------
def compute_mpe_reward(pi1, pi2):
    """Average per-step reward under MPE policies (stationary distribution)."""
    # Compute stationary distribution over states under transition matrix
    eigvals, eigvecs = np.linalg.eig(P_TRANSITION.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    mu = np.real(eigvecs[:, idx])
    mu = mu / mu.sum()

    avg_reward = 0.0
    for s in range(N_STATES):
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                r1, r2 = DynamicCournotEnv.profits(s, a1, a2)
                avg_reward += mu[s] * pi1[s, a1] * pi2[s, a2] * (r1 + r2) / 2.0
    return avg_reward


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Dynamic Cournot Duopoly: MARL Methods vs. Exact MPE")
    print("=" * 72)

    # --- Compute MPE ---
    print("\nComputing deterministic MPE via iterated best response...")
    mpe_pi1, mpe_pi2, R_all, mpe_policy1, mpe_policy2 = compute_mpe_deterministic()

    print("\nMPE policies (deterministic):")
    state_labels = ["Low (a=8)", "Medium (a=11)", "High (a=14)"]
    for s in range(N_STATES):
        print(f"  {state_labels[s]:16s}: firm 1 -> q={mpe_policy1[s]}, "
              f"firm 2 -> q={mpe_policy2[s]}")

    # Sanity check: static Cournot Nash for a=11, c=2 => q* = (11-2)/3 = 3
    print(f"\nSanity check (static Nash, a=11, c=2): q* = (11-2)/3 = 3")
    print(f"  MPE quantity at Medium demand: {mpe_policy1[1]}")

    mpe_reward = compute_mpe_reward(mpe_pi1, mpe_pi2)
    print(f"\nMPE average per-step reward (per firm): {mpe_reward:.4f}")

    # Use symmetric MPE for distance comparison
    mpe_pi = mpe_pi1  # symmetric game, pi1 == pi2

    # --- Run all methods across seeds ---
    runners = {
        "IQL": run_iql,
        "Nash-Q": run_nash_q,
        "WoLF-PHC": run_wolf_phc,
        "REINFORCE": run_reinforce,
        "Fictitious Play": run_fictitious_play,
    }

    n_evals = N_EPISODES // EVAL_EVERY
    all_distances = {m: np.zeros((N_SEEDS, n_evals)) for m in METHOD_NAMES}
    all_rewards = {m: np.zeros((N_SEEDS, n_evals)) for m in METHOD_NAMES}
    final_policies = {m: [] for m in METHOD_NAMES}

    import sys
    for method in METHOD_NAMES:
        print(f"\nRunning {method}...", flush=True)
        for seed in range(N_SEEDS):
            dists, rews, *policies = runners[method](mpe_pi, seed)
            all_distances[method][seed] = dists
            all_rewards[method][seed] = rews
            if seed == 0:
                final_policies[method] = policies
            if (seed + 1) % 5 == 0:
                print(f"  Completed seed {seed + 1}/{N_SEEDS}", flush=True)
                sys.stdout.flush()

    eval_episodes = np.arange(EVAL_EVERY, N_EPISODES + 1, EVAL_EVERY)

    # --- Figure 1: Policy distance from MPE ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for method in METHOD_NAMES:
        mean = all_distances[method].mean(axis=0)
        se = all_distances[method].std(axis=0) / np.sqrt(N_SEEDS)
        ax1.plot(eval_episodes, mean, color=COLORS[method], label=method,
                 linewidth=1.5)
        ax1.fill_between(eval_episodes, mean - se, mean + se,
                         color=COLORS[method], alpha=0.15)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Policy Distance from MPE (Total Variation)")
    ax1.set_title("Convergence to Markov Perfect Equilibrium")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(EVAL_EVERY, N_EPISODES)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = os.path.join(OUT_DIR, "dynamic_cournot_convergence.png")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {path1}")

    # --- Figure 2: Learned policies vs MPE by demand state ---
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    bar_width = 0.13
    x = np.arange(N_ACTIONS)

    for s_idx, ax in enumerate(axes):
        # MPE bars
        ax.bar(x - 2.5 * bar_width, mpe_pi[s_idx], bar_width, color="black",
               alpha=0.7, label="MPE")
        # Method bars
        POLICY_METHODS = {"WoLF-PHC", "REINFORCE"}
        for i, method in enumerate(METHOD_NAMES):
            p = final_policies[method]
            arr = p[0]
            if method in POLICY_METHODS:
                pi = arr  # already a probability table (3 x N_ACTIONS)
            elif arr.ndim == 3:
                # Nash-Q: joint Q-table Q[s, a1, a2]
                pi = np.zeros((N_STATES, N_ACTIONS))
                for ss in range(N_STATES):
                    best_a1 = np.argmax(np.max(arr[ss], axis=1))
                    pi[ss, best_a1] = 1.0
            else:
                pi = extract_greedy_policy(arr)
            ax.bar(x + (i - 1.5) * bar_width, pi[s_idx], bar_width,
                   color=COLORS[method], alpha=0.8, label=method if s_idx == 0 else "")

        ax.set_xlabel("Quantity $q$")
        ax.set_title(f"{state_labels[s_idx]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ACTIONS])
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Policy Probability")
    axes[0].legend(fontsize=7, loc="upper right")
    fig2.suptitle("Learned Policies vs. MPE by Demand State", fontsize=13)
    fig2.tight_layout()
    path2 = os.path.join(OUT_DIR, "dynamic_cournot_policies.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # --- Figure 3: Average reward convergence ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.axhline(y=mpe_reward, color="black", linestyle="--", linewidth=1.5,
                label="MPE reward", alpha=0.7)
    for method in METHOD_NAMES:
        mean = all_rewards[method].mean(axis=0)
        se = all_rewards[method].std(axis=0) / np.sqrt(N_SEEDS)
        ax3.plot(eval_episodes, mean, color=COLORS[method], label=method,
                 linewidth=1.5)
        ax3.fill_between(eval_episodes, mean - se, mean + se,
                         color=COLORS[method], alpha=0.15)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Average Per-Step Reward (per firm)")
    ax3.set_title("Reward Convergence")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_xlim(EVAL_EVERY, N_EPISODES)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    path3 = os.path.join(OUT_DIR, "dynamic_cournot_rewards.png")
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {path3}")

    # --- LaTeX results table ---
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Final Distance & Avg.\ Reward & "
                 r"Converged (ep.) \\")
    lines.append(r"\hline")

    for method in METHOD_NAMES:
        final_d = all_distances[method][:, -1]
        final_r = all_rewards[method][:, -1]
        d_mean = final_d.mean()
        d_se = final_d.std() / np.sqrt(N_SEEDS)
        r_mean = final_r.mean()
        r_se = final_r.std() / np.sqrt(N_SEEDS)

        # Convergence episode: first eval where mean distance < 0.1
        mean_curve = all_distances[method].mean(axis=0)
        conv_idx = np.where(mean_curve < 0.1)[0]
        if len(conv_idx) > 0:
            conv_ep = f"{eval_episodes[conv_idx[0]]:,}"
        else:
            conv_ep = "---"

        d_str = f"{d_mean:.3f} $\\pm$ {d_se:.3f}"
        r_str = f"{r_mean:.2f} $\\pm$ {r_se:.2f}"
        lines.append(f"{method} & {d_str} & {r_str} & {conv_ep} \\\\")

    lines.append(r"\hline")
    lines.append(f"MPE & 0.000 & {mpe_reward:.2f} & --- \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    tex_path = os.path.join(OUT_DIR, "dynamic_cournot_results.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {tex_path}")

    print("\n" + "=" * 72)
    print("Done.")
