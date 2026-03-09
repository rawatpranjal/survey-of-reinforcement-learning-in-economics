"""
Extended version with 10x10 grid, hazards, traps, and neural net reward model.
The paper's published results use gridworld_rlhf_paper.py instead.

RLHF Simulation Study: Honest Comparison of Preference-Based Policy Learning
Chapter 8 - RLHF & Preference Learning

Compares six methods on a 10x10 stochastic gridworld with hazards and traps:
  1. DP Oracle (value iteration with true reward)
  2. Q-Learning (true scalar reward)
  3. RLHF - Neural Net reward model (MLP, ~4800 params, Bradley-Terry MLE)
  4. RLHF - Correct structural model (4 params matching true reward form)
  5. RLHF - Misspecified structural model (2 params, ignores hazards/traps)
  6. DPO (tabular softmax policy, 400 params, no reward model)

Ablation: Online vs offline data collection for neural net RLHF at K=1000.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
apply_style()

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

GRID_SIZE = 10
SLIP_PROB = 0.10
GAMMA = 0.99
MAX_STEPS = 500

GOAL = (9, 9)
HAZARDS = {(3, 3), (6, 7)}
TRAPS = {(5, 2)}

PHI_TRUE = np.array([-0.1, 10.1, -1.9, -4.9])

Q_LEARNING_EPISODES = 20000
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_EPSILON = 0.1

COMPARISON_COUNTS = [25, 50, 100, 200, 500, 1000, 2000, 5000]
SEGMENT_LENGTH = 25

N_SEEDS = 30
N_SEEDS_ABLATION = 20
EVAL_EPISODES = 200

NN_INPUT_DIM = 8
NN_HIDDEN = 64
NN_LR = 1e-3
NN_EPOCHS = 200
NN_BATCH = 64

DPO_LR = 5e-3
DPO_EPOCHS = 300
DPO_LAMBDA_CANDIDATES = [0.001, 0.01, 0.1]

MASTER_SEED = 42
OUTPUT_DIR = 'ch08_rlhf/sims'


# =============================================================================
# Environment
# =============================================================================

class GridworldEnv:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self):
        self.N = GRID_SIZE
        self.goal = GOAL
        self.hazards = HAZARDS
        self.traps = TRAPS
        self.goal_si = self.state_to_index(GOAL)
        self.state = None
        self._trans_cache = {}
        self._precompute_transitions()
        self.reset()

    def _precompute_transitions(self):
        for si in range(self.num_states):
            s = self.index_to_state(si)
            for a in range(self.num_actions):
                self._trans_cache[(si, a)] = self._compute_transitions(s, a)

    def _compute_transitions(self, s, a):
        if s == self.goal:
            return [(s, 1.0)]
        transitions = {}
        ns_intended = self._next_state_det(s, a)
        transitions[ns_intended] = transitions.get(ns_intended, 0.0) + (1.0 - SLIP_PROB)
        for a2 in range(self.num_actions):
            ns_slip = self._next_state_det(s, a2)
            transitions[ns_slip] = transitions.get(ns_slip, 0.0) + SLIP_PROB / self.num_actions
        return list(transitions.items())

    @property
    def num_states(self):
        return self.N * self.N

    @property
    def num_actions(self):
        return len(self.ACTIONS)

    def reset(self, start=None):
        self.state = (0, 0) if start is None else start
        return self.state

    def state_to_index(self, s):
        return s[0] * self.N + s[1]

    def index_to_state(self, idx):
        return (idx // self.N, idx % self.N)

    def _next_state_det(self, state, action):
        dr, dc = self.ACTIONS[action]
        return (max(0, min(self.N - 1, state[0] + dr)),
                max(0, min(self.N - 1, state[1] + dc)))

    def transition_features(self, s, a, ns):
        return np.array([1.0, float(ns == self.goal),
                         float(ns in self.hazards), float(ns in self.traps)])

    def true_reward(self, s, a, ns):
        if s == self.goal:
            return 0.0
        return float(PHI_TRUE @ self.transition_features(s, a, ns))

    def step(self, action):
        if self.state == self.goal:
            return self.state, 0.0, True
        if np.random.random() < SLIP_PROB:
            actual = np.random.randint(self.num_actions)
        else:
            actual = action
        ns = self._next_state_det(self.state, actual)
        r = self.true_reward(self.state, action, ns)
        self.state = ns
        return ns, r, (ns == self.goal)


# =============================================================================
# DP, Evaluation, Q-Learning
# =============================================================================

def value_iteration(env, reward_fn=None, tol=1e-10, max_iter=5000):
    nS, nA = env.num_states, env.num_actions
    V = np.zeros(nS)
    if reward_fn is None:
        reward_fn = env.true_reward
    for it in range(max_iter):
        V_new = np.zeros(nS)
        for si in range(nS):
            if si == env.goal_si:
                continue
            s = env.index_to_state(si)
            best = -1e30
            for a in range(nA):
                q = sum(p * (reward_fn(s, a, ns) + GAMMA * V[env.state_to_index(ns)])
                        for ns, p in env._trans_cache[(si, a)])
                if q > best:
                    best = q
            V_new[si] = best
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = np.zeros(nS, dtype=int)
    for si in range(nS):
        if si == env.goal_si:
            continue
        s = env.index_to_state(si)
        best_q, best_a = -1e30, 0
        for a in range(nA):
            q = sum(p * (reward_fn(s, a, ns) + GAMMA * V[env.state_to_index(ns)])
                    for ns, p in env._trans_cache[(si, a)])
            if q > best_q:
                best_q, best_a = q, a
        policy[si] = best_a
    return V, policy, it + 1


def evaluate_deterministic(env, policy, n_episodes=EVAL_EPISODES):
    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for t in range(MAX_STEPS):
            if s == env.goal:
                break
            ns, r, done = env.step(int(policy[env.state_to_index(s)]))
            total += (GAMMA ** t) * r
            s = ns
            if done:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns) / np.sqrt(len(returns))


def evaluate_stochastic(env, theta, n_episodes=EVAL_EPISODES):
    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for t in range(MAX_STEPS):
            if s == env.goal:
                break
            si = env.state_to_index(s)
            probs = np.exp(theta[si] - theta[si].max())
            probs /= probs.sum()
            ns, r, done = env.step(np.random.choice(env.num_actions, p=probs))
            total += (GAMMA ** t) * r
            s = ns
            if done:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns) / np.sqrt(len(returns))


def q_learning(env):
    nS, nA = env.num_states, env.num_actions
    Q = np.zeros((nS, nA))
    total_steps = 0
    for _ in range(Q_LEARNING_EPISODES):
        s = env.reset()
        for _ in range(MAX_STEPS):
            if s == env.goal:
                break
            si = env.state_to_index(s)
            a = np.random.randint(nA) if np.random.random() < Q_LEARNING_EPSILON else np.argmax(Q[si])
            ns, r, done = env.step(a)
            Q[si, a] += Q_LEARNING_ALPHA * (r + GAMMA * np.max(Q[env.state_to_index(ns)]) - Q[si, a])
            total_steps += 1
            s = ns
            if done:
                break
    return np.argmax(Q, axis=1), total_steps


# =============================================================================
# Segment Generation
# =============================================================================

def generate_segment(env, policy, length=SEGMENT_LENGTH):
    while True:
        start = (np.random.randint(env.N), np.random.randint(env.N))
        if start != env.goal:
            break
    transitions = []
    total_reward = 0.0
    s = env.reset(start)
    for _ in range(length):
        if s == env.goal:
            break
        si = env.state_to_index(s)
        if isinstance(policy, str) and policy == 'random':
            a = np.random.randint(env.num_actions)
        elif callable(policy):
            a = policy(si)
        else:
            a = int(policy[si])
        ns, r, done = env.step(a)
        transitions.append((s, a, ns))
        total_reward += r
        s = ns
        if done:
            break
    return transitions, total_reward


def generate_comparisons(env, K, policy='random'):
    comparisons = []
    for _ in range(K):
        t1, r1 = generate_segment(env, policy)
        t2, r2 = generate_segment(env, policy)
        if not t1 or not t2:
            continue
        diff = np.clip(r1 - r2, -500, 500)
        if np.random.random() < 1.0 / (1.0 + np.exp(-diff)):
            comparisons.append((t1, t2))
        else:
            comparisons.append((t2, t1))
    return comparisons


# =============================================================================
# Neural Net Reward Model
# =============================================================================

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NN_INPUT_DIM, NN_HIDDEN), nn.ReLU(),
            nn.Linear(NN_HIDDEN, NN_HIDDEN), nn.ReLU(),
            nn.Linear(NN_HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _precompute_nn_tensors(comparisons):
    """Pad all segments to SEGMENT_LENGTH and return (K, L, 8) tensors + masks."""
    K = len(comparisons)
    L = SEGMENT_LENGTH
    w_feats = np.zeros((K, L, NN_INPUT_DIM))
    l_feats = np.zeros((K, L, NN_INPUT_DIM))
    w_mask = np.zeros((K, L))
    l_mask = np.zeros((K, L))

    for i, (winner, loser) in enumerate(comparisons):
        for t, (s, a, ns) in enumerate(winner):
            a_oh = [0.0, 0.0, 0.0, 0.0]
            a_oh[a] = 1.0
            w_feats[i, t] = [s[0] / GRID_SIZE, s[1] / GRID_SIZE,
                              a_oh[0], a_oh[1], a_oh[2], a_oh[3],
                              ns[0] / GRID_SIZE, ns[1] / GRID_SIZE]
            w_mask[i, t] = 1.0
        for t, (s, a, ns) in enumerate(loser):
            a_oh = [0.0, 0.0, 0.0, 0.0]
            a_oh[a] = 1.0
            l_feats[i, t] = [s[0] / GRID_SIZE, s[1] / GRID_SIZE,
                              a_oh[0], a_oh[1], a_oh[2], a_oh[3],
                              ns[0] / GRID_SIZE, ns[1] / GRID_SIZE]
            l_mask[i, t] = 1.0

    return (torch.tensor(w_feats, dtype=torch.float32),
            torch.tensor(l_feats, dtype=torch.float32),
            torch.tensor(w_mask, dtype=torch.float32),
            torch.tensor(l_mask, dtype=torch.float32))


def train_reward_net(comparisons):
    model = RewardNet()
    optimizer = optim.Adam(model.parameters(), lr=NN_LR)

    w_feats, l_feats, w_mask, l_mask = _precompute_nn_tensors(comparisons)
    K = w_feats.shape[0]
    indices = np.arange(K)

    for epoch in range(NN_EPOCHS):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, K, NN_BATCH):
            batch = indices[start:start + NN_BATCH]
            bw = w_feats[batch]  # (B, L, 8)
            bl = l_feats[batch]
            bwm = w_mask[batch]  # (B, L)
            blm = l_mask[batch]

            B, L, D = bw.shape
            # Forward all transitions at once
            r_w_all = model(bw.reshape(B * L, D)).reshape(B, L)  # (B, L)
            r_l_all = model(bl.reshape(B * L, D)).reshape(B, L)
            r_w = (r_w_all * bwm).sum(dim=1)  # (B,)
            r_l = (r_l_all * blm).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(r_w - r_l).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

    return model, total_loss / K


def extract_reward_table(model, env):
    nS, nA = env.num_states, env.num_actions
    R = np.zeros((nS, nA))
    all_feats, mapping = [], []
    for si in range(nS):
        s = env.index_to_state(si)
        if si == env.goal_si:
            continue
        for a in range(nA):
            for ns, p in env._trans_cache[(si, a)]:
                a_oh = [0.0, 0.0, 0.0, 0.0]
                a_oh[a] = 1.0
                all_feats.append([s[0] / GRID_SIZE, s[1] / GRID_SIZE,
                                   a_oh[0], a_oh[1], a_oh[2], a_oh[3],
                                   ns[0] / GRID_SIZE, ns[1] / GRID_SIZE])
                mapping.append((si, a, p))
    model.eval()
    with torch.no_grad():
        rewards = model(torch.tensor(all_feats, dtype=torch.float32)).numpy()
    for idx, (si, a, p) in enumerate(mapping):
        R[si, a] += p * rewards[idx]
    return R


def vi_from_reward_table(env, R):
    nS, nA = env.num_states, env.num_actions
    V = np.zeros(nS)
    for it in range(5000):
        V_new = np.zeros(nS)
        for si in range(nS):
            if si == env.goal_si:
                continue
            best = -1e30
            for a in range(nA):
                q = sum(p * (R[si, a] + GAMMA * V[env.state_to_index(ns)])
                        for ns, p in env._trans_cache[(si, a)])
                if q > best:
                    best = q
            V_new[si] = best
        if np.max(np.abs(V_new - V)) < 1e-10:
            break
        V = V_new
    policy = np.zeros(nS, dtype=int)
    for si in range(nS):
        if si == env.goal_si:
            continue
        best_q, best_a = -1e30, 0
        for a in range(nA):
            q = sum(p * (R[si, a] + GAMMA * V[env.state_to_index(ns)])
                    for ns, p in env._trans_cache[(si, a)])
            if q > best_q:
                best_q, best_a = q, a
        policy[si] = best_a
    return V, policy


# =============================================================================
# Structural Models (vectorized BT MLE)
# =============================================================================

def _precompute_structural_data(comparisons, env, n_features):
    """Returns (K, n_features) arrays of feature sums for winner and loser."""
    K = len(comparisons)
    fw = np.zeros((K, n_features))
    fl = np.zeros((K, n_features))
    for i, (winner, loser) in enumerate(comparisons):
        for s, a, ns in winner:
            fw[i] += env.transition_features(s, a, ns)[:n_features]
        for s, a, ns in loser:
            fl[i] += env.transition_features(s, a, ns)[:n_features]
    return fw, fl


def _structural_nll_and_grad(phi, fw, fl):
    diff_feats = fw - fl  # (K, d)
    diff = np.clip(diff_feats @ phi, -500, 500)  # (K,)
    sigma = 1.0 / (1.0 + np.exp(-diff))
    nll = -np.sum(np.log(sigma + 1e-15))
    grad = -diff_feats.T @ (1.0 - sigma)  # (d,)
    return nll, grad


def fit_structural(comparisons, env, n_features):
    fw, fl = _precompute_structural_data(comparisons, env, n_features)
    phi0 = np.zeros(n_features)
    result = minimize(lambda phi: _structural_nll_and_grad(phi, fw, fl),
                      phi0, jac=True, method='L-BFGS-B')
    return result.x


def structural_reward_fn(env, phi, n_features):
    def rfn(s, a, ns):
        if s == env.goal:
            return 0.0
        return float(phi @ env.transition_features(s, a, ns)[:n_features])
    return rfn


# =============================================================================
# DPO (fully vectorized)
# =============================================================================

def _precompute_dpo_arrays(comparisons, env):
    """Pad all segments to SEGMENT_LENGTH and create (K, L) arrays."""
    K = len(comparisons)
    L = SEGMENT_LENGTH
    nA = env.num_actions
    # Use goal state as padding (its theta contribution will be masked out)
    pad_si = env.goal_si

    w_si = np.full((K, L), pad_si, dtype=np.intp)
    w_a = np.zeros((K, L), dtype=np.intp)
    w_mask = np.zeros((K, L), dtype=np.float64)

    l_si = np.full((K, L), pad_si, dtype=np.intp)
    l_a = np.zeros((K, L), dtype=np.intp)
    l_mask = np.zeros((K, L), dtype=np.float64)

    for i, (winner, loser) in enumerate(comparisons):
        for t, (s, a, ns) in enumerate(winner):
            w_si[i, t] = env.state_to_index(s)
            w_a[i, t] = a
            w_mask[i, t] = 1.0
        for t, (s, a, ns) in enumerate(loser):
            l_si[i, t] = env.state_to_index(s)
            l_a[i, t] = a
            l_mask[i, t] = 1.0

    return w_si, w_a, w_mask, l_si, l_a, l_mask


def _batch_log_prob(theta, si, a, mask):
    """Vectorized log pi(segment) for all K comparisons.
    theta: (nS, nA), si: (K, L), a: (K, L), mask: (K, L)
    Returns: (K,) log-probabilities.
    """
    logits = theta[si]  # (K, L, nA)
    log_z = np.logaddexp.reduce(logits, axis=2)  # (K, L)
    selected = logits[np.arange(si.shape[0])[:, None], np.arange(si.shape[1])[None, :], a]
    return ((selected - log_z) * mask).sum(axis=1)  # (K,)


def train_dpo(comparisons, env, lambda_kl, dpo_arrays=None):
    nS, nA = env.num_states, env.num_actions
    theta = np.zeros((nS, nA))
    log_nA = np.log(nA)

    if dpo_arrays is None:
        dpo_arrays = _precompute_dpo_arrays(comparisons, env)
    w_si, w_a, w_mask, l_si, l_a, l_mask = dpo_arrays

    K = w_si.shape[0]
    L = w_si.shape[1]
    w_len = w_mask.sum(axis=1)  # (K,)
    l_len = l_mask.sum(axis=1)  # (K,)

    # Adam state
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    final_loss = 0.0
    for epoch in range(DPO_EPOCHS):
        # Log-ratios (vectorized)
        log_pi_w = _batch_log_prob(theta, w_si, w_a, w_mask)  # (K,)
        log_pi_l = _batch_log_prob(theta, l_si, l_a, l_mask)
        log_ratio_w = log_pi_w + w_len * log_nA
        log_ratio_l = log_pi_l + l_len * log_nA

        h = np.clip(lambda_kl * (log_ratio_w - log_ratio_l), -500, 500)
        sigma_h = 1.0 / (1.0 + np.exp(-h))  # (K,)
        final_loss = -np.mean(np.log(sigma_h + 1e-15))

        coeff = -(1.0 - sigma_h) * lambda_kl / K  # (K,)

        # Gradient (vectorized)
        grad = np.zeros_like(theta)

        # Softmax probs at all positions
        w_logits = theta[w_si]  # (K, L, nA)
        w_probs = np.exp(w_logits - w_logits.max(axis=2, keepdims=True))
        w_probs /= w_probs.sum(axis=2, keepdims=True)

        l_logits = theta[l_si]
        l_probs = np.exp(l_logits - l_logits.max(axis=2, keepdims=True))
        l_probs /= l_probs.sum(axis=2, keepdims=True)

        # Winner gradient: coeff[k] * (indicator[a] - probs) * mask
        w_indicator = np.zeros((K, L, nA))
        w_indicator[np.arange(K)[:, None], np.arange(L)[None, :], w_a] = 1.0
        w_delta = (w_indicator - w_probs) * w_mask[:, :, None] * coeff[:, None, None]

        # Loser gradient: -coeff[k] * (indicator[a] - probs) * mask
        l_indicator = np.zeros((K, L, nA))
        l_indicator[np.arange(K)[:, None], np.arange(L)[None, :], l_a] = 1.0
        l_delta = -(l_indicator - l_probs) * l_mask[:, :, None] * coeff[:, None, None]

        # Accumulate: np.add.at for sparse accumulation
        np.add.at(grad, w_si.ravel(),
                  w_delta.reshape(-1, nA))
        np.add.at(grad, l_si.ravel(),
                  l_delta.reshape(-1, nA))

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        theta -= DPO_LR * m_hat / (np.sqrt(v_hat) + eps)

    return theta, final_loss


def train_dpo_best_lambda(comparisons, env):
    dpo_arrays = _precompute_dpo_arrays(comparisons, env)
    best_theta, best_loss, best_lam = None, float('inf'), None
    for lam in DPO_LAMBDA_CANDIDATES:
        theta, loss = train_dpo(comparisons, env, lam, dpo_arrays=dpo_arrays)
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
            best_lam = lam
    return best_theta, best_loss, best_lam


# =============================================================================
# Online vs Offline
# =============================================================================

def run_nn_online(env, K, n_rounds=4):
    all_comps = []
    k_per_round = K // n_rounds
    policy = 'random'
    for _ in range(n_rounds):
        all_comps.extend(generate_comparisons(env, k_per_round, policy=policy))
        model, _ = train_reward_net(list(all_comps))
        R = extract_reward_table(model, env)
        _, policy = vi_from_reward_table(env, R)
    return policy


def run_nn_offline(env, K):
    comps = generate_comparisons(env, K)
    model, _ = train_reward_net(comps)
    R = extract_reward_table(model, env)
    _, policy = vi_from_reward_table(env, R)
    return policy


# =============================================================================
# Utilities
# =============================================================================

def count_hazard_visits(env, policy):
    count = 0
    for si in range(env.num_states):
        if si == env.goal_si:
            continue
        s = env.index_to_state(si)
        ns = env._next_state_det(s, int(policy[si]))
        if ns in HAZARDS or ns in TRAPS:
            count += 1
    return count


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('=' * 80)
    print('RLHF SIMULATION STUDY: HONEST COMPARISON')
    print('=' * 80)
    print(f'\nTimestamp: {timestamp}')
    print(f'Python: {sys.version.split()[0]}, NumPy: {np.__version__}, PyTorch: {torch.__version__}')
    print(f'\nPARAMETERS:')
    print(f'  Grid: {GRID_SIZE}x{GRID_SIZE}, slip={SLIP_PROB}, gamma={GAMMA}')
    print(f'  Goal: {GOAL}, Hazards: {HAZARDS}, Traps: {TRAPS}')
    print(f'  True phi: {PHI_TRUE}')
    print(f'    step={PHI_TRUE[0]:.1f}, goal={PHI_TRUE[0]+PHI_TRUE[1]:.1f}, '
          f'hazard={PHI_TRUE[0]+PHI_TRUE[2]:.1f}, trap={PHI_TRUE[0]+PHI_TRUE[3]:.1f}')
    print(f'  Segment length: {SEGMENT_LENGTH}, K values: {COMPARISON_COUNTS}')
    print(f'  Seeds: {N_SEEDS} (main), {N_SEEDS_ABLATION} (ablation)')
    print(f'  NN: {NN_INPUT_DIM}->{NN_HIDDEN}->{NN_HIDDEN}->1, lr={NN_LR}, '
          f'epochs={NN_EPOCHS}, batch={NN_BATCH}')
    print(f'  DPO: lambda={DPO_LAMBDA_CANDIDATES}, lr={DPO_LR}, epochs={DPO_EPOCHS}')

    env = GridworldEnv()

    # --- 1. DP Oracle ---
    print('\n' + '=' * 80)
    print('1. DP ORACLE')
    print('=' * 80)
    V_star, pi_star, vi_iters = value_iteration(env)
    np.random.seed(MASTER_SEED)
    dp_ret, dp_se = evaluate_deterministic(env, pi_star)
    dp_hv = count_hazard_visits(env, pi_star)
    print(f'  Converged in {vi_iters} iterations')
    print(f'  V*(start): {V_star[0]:.4f}')
    print(f'  Return: {dp_ret:.3f} +/- {dp_se:.3f}')
    print(f'  Hazard/trap visits in policy: {dp_hv}')

    # --- 2. Q-Learning ---
    print('\n' + '=' * 80)
    print('2. Q-LEARNING')
    print('=' * 80)
    ql_returns, ql_steps_list = [], []
    for seed in range(N_SEEDS):
        np.random.seed(MASTER_SEED + seed)
        pol, steps = q_learning(env)
        np.random.seed(MASTER_SEED + 9000 + seed)
        ret, _ = evaluate_deterministic(env, pol)
        ql_returns.append(ret)
        ql_steps_list.append(steps)
        if seed < 3 or seed == N_SEEDS - 1:
            print(f'  Seed {seed:2d}: queries={steps}, return={ret:.3f}')
            if seed == 2 and N_SEEDS > 4:
                print('  ...')
    ql_mean = np.mean(ql_returns)
    ql_se = np.std(ql_returns) / np.sqrt(N_SEEDS)
    print(f'\n  Summary: return={ql_mean:.3f} +/- {ql_se:.3f}, queries={int(np.mean(ql_steps_list))}')

    # --- 3-6. Preference methods ---
    nn_res = {K: [] for K in COMPARISON_COUNTS}
    co_res = {K: [] for K in COMPARISON_COUNTS}
    mi_res = {K: [] for K in COMPARISON_COUNTS}
    dpo_res = {K: [] for K in COMPARISON_COUNTS}
    co_phis = {K: [] for K in COMPARISON_COUNTS}
    mi_phis = {K: [] for K in COMPARISON_COUNTS}
    mi_hvs = {K: [] for K in COMPARISON_COUNTS}

    for K in COMPARISON_COUNTS:
        print(f'\n{"=" * 80}')
        print(f'K = {K}')
        print('=' * 80)

        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000 + K
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)

            comps = generate_comparisons(env, K)

            # Neural net RLHF
            model, loss = train_reward_net(list(comps))
            R_tab = extract_reward_table(model, env)
            _, nn_pol = vi_from_reward_table(env, R_tab)
            np.random.seed(rng_seed + 50000)
            ret, _ = evaluate_deterministic(env, nn_pol)
            nn_res[K].append(ret)

            # Correct structural (4 params)
            phi_c = fit_structural(comps, env, 4)
            _, pi_c, _ = value_iteration(env, structural_reward_fn(env, phi_c, 4))
            np.random.seed(rng_seed + 60000)
            ret, _ = evaluate_deterministic(env, pi_c)
            co_res[K].append(ret)
            co_phis[K].append(phi_c.copy())

            # Misspecified structural (2 params)
            phi_m = fit_structural(comps, env, 2)
            _, pi_m, _ = value_iteration(env, structural_reward_fn(env, phi_m, 2))
            np.random.seed(rng_seed + 70000)
            ret, _ = evaluate_deterministic(env, pi_m)
            mi_res[K].append(ret)
            mi_phis[K].append(phi_m.copy())
            mi_hvs[K].append(count_hazard_visits(env, pi_m))

            # DPO
            theta, dpo_loss, best_lam = train_dpo_best_lambda(comps, env)
            np.random.seed(rng_seed + 80000)
            ret, _ = evaluate_stochastic(env, theta)
            dpo_res[K].append(ret)

        nn_m = np.mean(nn_res[K]); nn_s = np.std(nn_res[K]) / np.sqrt(N_SEEDS)
        co_m = np.mean(co_res[K]); co_s = np.std(co_res[K]) / np.sqrt(N_SEEDS)
        mi_m = np.mean(mi_res[K]); mi_s = np.std(mi_res[K]) / np.sqrt(N_SEEDS)
        dp_m = np.mean(dpo_res[K]); dp_s = np.std(dpo_res[K]) / np.sqrt(N_SEEDS)
        print(f'  NN RLHF:     {nn_m:7.3f} +/- {nn_s:.3f}')
        print(f'  Correct:     {co_m:7.3f} +/- {co_s:.3f}')
        print(f'  Misspecified:{mi_m:7.3f} +/- {mi_s:.3f}  (hazard visits: {np.mean(mi_hvs[K]):.1f})')
        print(f'  DPO:         {dp_m:7.3f} +/- {dp_s:.3f}')

    # --- Ablation ---
    print('\n' + '=' * 80)
    print('ABLATION: ONLINE VS OFFLINE (NN RLHF, K=1000)')
    print('=' * 80)
    K_abl = 1000
    on_rets, off_rets = [], []
    for seed in range(N_SEEDS_ABLATION):
        np.random.seed(MASTER_SEED + 5000 + seed)
        torch.manual_seed(MASTER_SEED + 5000 + seed)
        pol_on = run_nn_online(env, K_abl)
        np.random.seed(MASTER_SEED + 5500 + seed)
        ret, _ = evaluate_deterministic(env, pol_on)
        on_rets.append(ret)

        np.random.seed(MASTER_SEED + 6000 + seed)
        torch.manual_seed(MASTER_SEED + 6000 + seed)
        pol_off = run_nn_offline(env, K_abl)
        np.random.seed(MASTER_SEED + 6500 + seed)
        ret, _ = evaluate_deterministic(env, pol_off)
        off_rets.append(ret)

        if seed < 3 or seed == N_SEEDS_ABLATION - 1:
            print(f'  Seed {seed:2d}: online={on_rets[-1]:.3f}, offline={off_rets[-1]:.3f}')
            if seed == 2 and N_SEEDS_ABLATION > 4:
                print('  ...')

    on_mean = np.mean(on_rets); on_se = np.std(on_rets) / np.sqrt(N_SEEDS_ABLATION)
    off_mean = np.mean(off_rets); off_se = np.std(off_rets) / np.sqrt(N_SEEDS_ABLATION)
    _, p_val = stats.ttest_ind(on_rets, off_rets, alternative='greater')
    print(f'\n  Online:  {on_mean:.3f} +/- {on_se:.3f}')
    print(f'  Offline: {off_mean:.3f} +/- {off_se:.3f}')
    print(f'  p-value (online > offline): {p_val:.4f}')

    # --- Summary ---
    print('\n' + '=' * 80)
    print('MAIN RESULTS TABLE')
    print('=' * 80)
    hdr = f'{"Method":<28s} | {"K":>5s} | {"Return":>18s} | {"% DP":>6s}'
    print(hdr)
    print('-' * len(hdr))
    print(f'{"DP Oracle":<28s} | {"--":>5s} | {dp_ret:7.3f} +/- {dp_se:.3f} | {100.0:6.1f}')
    print(f'{"Q-Learning":<28s} | {"--":>5s} | {ql_mean:7.3f} +/- {ql_se:.3f} | {100*ql_mean/dp_ret:6.1f}')
    print('-' * len(hdr))
    for K in COMPARISON_COUNTS:
        for name, res in [('NN RLHF', nn_res), ('Correct Structural', co_res),
                          ('Misspecified', mi_res), ('DPO', dpo_res)]:
            m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
            print(f'{name:<28s} | {K:5d} | {m:7.3f} +/- {s:.3f} | {100*m/dp_ret:6.1f}')
        if K != COMPARISON_COUNTS[-1]:
            print('-' * len(hdr))

    # --- Verification ---
    print('\n' + '=' * 80)
    print('VERIFICATION')
    print('=' * 80)
    print(f'  1. DP return positive: {dp_ret:.3f} -> {"PASS" if dp_ret > 0 else "FAIL"}')
    print(f'  2. Q-learning near DP: {100*ql_mean/dp_ret:.1f}% -> '
          f'{"PASS" if ql_mean/dp_ret > 0.85 else "FAIL"}')
    nn_hi = np.mean(nn_res[COMPARISON_COUNTS[-1]])
    print(f'  3. NN at K={COMPARISON_COUNTS[-1]}: {100*nn_hi/dp_ret:.1f}% -> '
          f'{"PASS" if nn_hi/dp_ret > 0.85 else "FAIL"}')
    co_hi = np.mean(co_res[COMPARISON_COUNTS[-1]])
    print(f'  4. Correct at K={COMPARISON_COUNTS[-1]}: {100*co_hi/dp_ret:.1f}% -> '
          f'{"PASS" if co_hi/dp_ret > 0.90 else "FAIL"}')
    mi_hi = np.mean(mi_res[COMPARISON_COUNTS[-1]])
    mi_hv_hi = np.mean(mi_hvs[COMPARISON_COUNTS[-1]])
    print(f'  5. Misspec suboptimal: {100*mi_hi/dp_ret:.1f}%, hv={mi_hv_hi:.1f} -> '
          f'{"PASS" if mi_hi < dp_ret * 0.98 else "CHECK"}')
    dpo_lo = np.mean(dpo_res[COMPARISON_COUNTS[0]])
    dpo_hi = np.mean(dpo_res[COMPARISON_COUNTS[-1]])
    print(f'  6. DPO monotonic: K={COMPARISON_COUNTS[0]}:{dpo_lo:.1f}, '
          f'K={COMPARISON_COUNTS[-1]}:{dpo_hi:.1f} -> '
          f'{"PASS" if dpo_hi > dpo_lo else "FAIL"}')
    print(f'  7. Online >= offline: {on_mean:.3f} vs {off_mean:.3f}, p={p_val:.4f} -> '
          f'{"PASS" if on_mean >= off_mean else "CHECK"}')

    # --- Parameter Recovery ---
    print('\n' + '=' * 80)
    print('PARAMETER RECOVERY (CORRECT STRUCTURAL)')
    print('=' * 80)
    print(f'     K | {"phi_0":>8s} | {"phi_1":>8s} | {"phi_2":>8s} | {"phi_3":>8s}')
    print(f'  true | {PHI_TRUE[0]:>8.3f} | {PHI_TRUE[1]:>8.3f} | {PHI_TRUE[2]:>8.3f} | {PHI_TRUE[3]:>8.3f}')
    print('-' * 52)
    for K in COMPARISON_COUNTS:
        means = np.mean(co_phis[K], axis=0)
        print(f'{K:6d} | {means[0]:>8.3f} | {means[1]:>8.3f} | {means[2]:>8.3f} | {means[3]:>8.3f}')

    print('\n' + '=' * 80)
    print('MISSPECIFIED HAZARD ANALYSIS')
    print('=' * 80)
    for K in COMPARISON_COUNTS:
        hv = np.mean(mi_hvs[K])
        mm = np.mean(mi_res[K])
        print(f'  K={K:5d}: hazard visits={hv:.1f}, return={mm:.3f}')

    # --- Figure ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    K_arr = np.array(COMPARISON_COUNTS)
    ax.axhline(dp_ret, color=COLORS['black'], ls='--', label='DP Oracle', alpha=0.8)
    ax.axhline(ql_mean, color=COLORS['blue'], ls=':', label='Q-Learning', alpha=0.8)
    for name, res, color, marker in [
        ('RLHF (Neural Net)', nn_res, COLORS['green'], 'o'),
        ('RLHF (Correct)', co_res, COLORS['purple'], 's'),
        ('RLHF (Misspecified)', mi_res, COLORS['red'], '^'),
        ('DPO', dpo_res, COLORS['orange'], 'D'),
    ]:
        means = [np.mean(res[K]) for K in COMPARISON_COUNTS]
        ses = [np.std(res[K]) / np.sqrt(N_SEEDS) for K in COMPARISON_COUNTS]
        ax.errorbar(K_arr, means, yerr=ses, marker=marker, color=color,
                     label=name, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Preference Comparisons ($K$)')
    ax.set_ylabel('Policy Return')
    ax.set_title('Sample Complexity of Preference-Based Policy Learning')
    ax.legend(loc='lower right', fontsize=8)
    fig_path = os.path.join(OUTPUT_DIR, 'gridworld_sample_complexity.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {fig_path}')

    # --- Tables ---
    tex1 = os.path.join(OUTPUT_DIR, 'gridworld_rlhf_results.tex')
    with open(tex1, 'w') as f:
        f.write('\\begin{tabular}{llrr}\n\\hline\n')
        f.write('Method & $K$ & Return & \\% of DP \\\\\n\\hline\n')
        f.write(f'DP Oracle & --- & ${dp_ret:.2f} \\pm {dp_se:.2f}$ & $100.0$ \\\\\n')
        f.write(f'Q-Learning & --- & ${ql_mean:.2f} \\pm {ql_se:.2f}$ & '
                f'${100*ql_mean/dp_ret:.1f}$ \\\\\n\\hline\n')
        for K in COMPARISON_COUNTS:
            for name, res in [('NN RLHF', nn_res), ('Correct', co_res),
                              ('Misspecified', mi_res), ('DPO', dpo_res)]:
                m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
                f.write(f'{name} & ${K}$ & ${m:.2f} \\pm {s:.2f}$ & '
                        f'${100*m/dp_ret:.1f}$ \\\\\n')
            if K != COMPARISON_COUNTS[-1]:
                f.write('\\hline\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex1}')

    tex2 = os.path.join(OUTPUT_DIR, 'gridworld_online_offline.tex')
    with open(tex2, 'w') as f:
        f.write('\\begin{tabular}{lrr}\n\\hline\n')
        f.write('Collection & Return & $p$-value \\\\\n\\hline\n')
        f.write(f'Online (4 rounds) & ${on_mean:.2f} \\pm {on_se:.2f}$ & \\\\\n')
        f.write(f'Offline & ${off_mean:.2f} \\pm {off_se:.2f}$ & ${p_val:.4f}$ \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex2}')

    print(f'\nOUTPUT FILES:')
    print(f'  {fig_path}')
    print(f'  {tex1}')
    print(f'  {tex2}')


if __name__ == '__main__':
    run_experiment()
