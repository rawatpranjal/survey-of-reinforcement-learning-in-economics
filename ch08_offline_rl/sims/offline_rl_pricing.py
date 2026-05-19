# Offline RL for Perishable Inventory Pricing — Chapter 8, Offline Reinforcement Learning.
# Compares seven methods on a finite-horizon perishable inventory pricing MDP with
# demand regime switching: behavior cloning, FQI, CQL, IQL, BCQ (the pessimism family),
# Decision Transformer, and return-conditioned supervised learning (the supervised-
# conditioning family). Per-paradigm caching via sims.sim_cache.compute_or_load so
# tweaking one method's hyperparameters does not invalidate the others.

import argparse
import copy
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, DOMAIN_COLORS, ALGO_COLORS, BENCH_STYLE, FIG_SINGLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'offline_rl_pricing'

# MDP parameters
MAX_INVENTORY = 30
N_DEMAND_REGIMES = 4
HORIZON = 20
N_PRICES = 10
PRICE_GRID = np.arange(1, N_PRICES + 1, dtype=float)  # {1,...,10}
LAMBDA_0 = np.array([1.5, 3.0, 5.0, 8.0])
PRICE_SENSITIVITY = 0.15
SALVAGE_VALUE = -2.0  # spoilage cost per unsold unit
GAMMA = 1.0  # finite horizon, no discounting

# Demand regime transition matrix (mildly persistent)
DEMAND_TRANS = np.array([
    [0.6, 0.2, 0.15, 0.05],
    [0.15, 0.6, 0.15, 0.1],
    [0.1, 0.15, 0.6, 0.15],
    [0.05, 0.15, 0.2, 0.6],
])

# Behavioral policy
BEHAVIORAL_MARKUPS = np.array([10, 10, 10, 10], dtype=float)
BEHAVIORAL_NOISE = 0.15

# Experiment parameters
N_OFFLINE_EPISODES = 500
N_EVAL_EPISODES = 1000
N_SEEDS = 20

# Coverage sensitivity experiment
EPSILON_B_VALUES = [0.05, 0.3, 0.9]

# Neural net common
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 256

# Q-method training
N_FQI_ITERATIONS = 200
N_GRADIENT_STEPS = 300
CQL_ALPHA = 0.1
IQL_TAU = 0.7
BCQ_THRESHOLD = 0.3

# DT (Decision Transformer) hyperparameters
DT_HIDDEN_DIM = 64
DT_N_LAYERS = 2
DT_N_HEADS = 4
DT_CONTEXT_K = 10
DT_N_GRADIENT_STEPS = 500
DT_LEARNING_RATE = 3e-4
DT_BATCH_SIZE = 32
DT_RETURN_NORM = 300.0  # divisor for normalizing return-to-go inputs

# RvS hyperparameters
RVS_HIDDEN_DIM = 128
RVS_N_GRADIENT_STEPS = 500
RVS_LEARNING_RATE = 1e-3
RVS_RETURN_NORM = 300.0

# Config version (bump to invalidate all caches)
CONFIG_VERSION = 13

ENV_PARAMS = {
    'MAX_INVENTORY': MAX_INVENTORY,
    'N_DEMAND_REGIMES': N_DEMAND_REGIMES,
    'HORIZON': HORIZON,
    'N_PRICES': N_PRICES,
    'LAMBDA_0': LAMBDA_0.tolist(),
    'PRICE_SENSITIVITY': PRICE_SENSITIVITY,
    'SALVAGE_VALUE': SALVAGE_VALUE,
    'GAMMA': GAMMA,
    'DEMAND_TRANS': DEMAND_TRANS.tolist(),
    'BEHAVIORAL_MARKUPS': BEHAVIORAL_MARKUPS.tolist(),
    'BEHAVIORAL_NOISE': BEHAVIORAL_NOISE,
}

SHARED_CONFIG = {
    **ENV_PARAMS,
    'N_OFFLINE_EPISODES': N_OFFLINE_EPISODES,
    'N_EVAL_EPISODES': N_EVAL_EPISODES,
    'N_SEEDS': N_SEEDS,
    'version': CONFIG_VERSION,
}

BC_CONFIG = {
    **SHARED_CONFIG,
    'HIDDEN_DIM': HIDDEN_DIM,
    'LEARNING_RATE': LEARNING_RATE,
    'BATCH_SIZE': BATCH_SIZE,
    'N_GRADIENT_STEPS': N_GRADIENT_STEPS,
}

FQI_CONFIG = {
    **BC_CONFIG,
    'N_FQI_ITERATIONS': N_FQI_ITERATIONS,
}

CQL_CONFIG = {**FQI_CONFIG, 'CQL_ALPHA': CQL_ALPHA}
IQL_CONFIG = {**FQI_CONFIG, 'IQL_TAU': IQL_TAU}
BCQ_CONFIG = {**FQI_CONFIG, 'BCQ_THRESHOLD': BCQ_THRESHOLD}

DT_CONFIG = {
    **SHARED_CONFIG,
    'DT_HIDDEN_DIM': DT_HIDDEN_DIM,
    'DT_N_LAYERS': DT_N_LAYERS,
    'DT_N_HEADS': DT_N_HEADS,
    'DT_CONTEXT_K': DT_CONTEXT_K,
    'DT_N_GRADIENT_STEPS': DT_N_GRADIENT_STEPS,
    'DT_LEARNING_RATE': DT_LEARNING_RATE,
    'DT_BATCH_SIZE': DT_BATCH_SIZE,
    'DT_RETURN_NORM': DT_RETURN_NORM,
}

RVS_CONFIG = {
    **SHARED_CONFIG,
    'RVS_HIDDEN_DIM': RVS_HIDDEN_DIM,
    'RVS_N_GRADIENT_STEPS': RVS_N_GRADIENT_STEPS,
    'RVS_LEARNING_RATE': RVS_LEARNING_RATE,
    'RVS_RETURN_NORM': RVS_RETURN_NORM,
}


# ---------------------------------------------------------------------------
# MDP: Perishable Inventory Pricing
# ---------------------------------------------------------------------------
def demand_rate(demand_regime, price):
    return LAMBDA_0[demand_regime] * np.exp(-PRICE_SENSITIVITY * price)


def sample_demand(demand_regime, price, rng):
    return rng.poisson(demand_rate(demand_regime, price))


def step(inventory, demand_regime, time_remaining, price, rng):
    if time_remaining <= 0:
        return inventory * SALVAGE_VALUE, inventory, demand_regime, 0
    q = sample_demand(demand_regime, price, rng)
    sold = min(q, inventory)
    reward = price * sold
    next_inv = inventory - sold
    next_d = rng.choice(N_DEMAND_REGIMES, p=DEMAND_TRANS[demand_regime])
    next_t = time_remaining - 1
    return reward, next_inv, next_d, next_t


def featurize(inventory, demand_regime, time_remaining):
    inv_norm = inventory / MAX_INVENTORY
    d_norm = demand_regime / (N_DEMAND_REGIMES - 1)
    t_norm = time_remaining / HORIZON
    return np.array([inv_norm, d_norm, t_norm], dtype=np.float32)


def featurize_sa(inventory, demand_regime, time_remaining, action_idx):
    s = featurize(inventory, demand_regime, time_remaining)
    a_norm = np.float32(action_idx / (N_PRICES - 1))
    return np.concatenate([s, [a_norm]]).astype(np.float32)


# ---------------------------------------------------------------------------
# DP Oracle (backward induction)
# ---------------------------------------------------------------------------
def solve_dp():
    V = np.zeros((MAX_INVENTORY + 1, N_DEMAND_REGIMES, HORIZON + 1))
    policy = np.zeros((MAX_INVENTORY + 1, N_DEMAND_REGIMES, HORIZON + 1), dtype=int)
    for i in range(MAX_INVENTORY + 1):
        for d in range(N_DEMAND_REGIMES):
            V[i, d, 0] = i * SALVAGE_VALUE
    for t in range(1, HORIZON + 1):
        for i in range(MAX_INVENTORY + 1):
            for d in range(N_DEMAND_REGIMES):
                best_val = -np.inf
                best_a = 0
                for a_idx in range(N_PRICES):
                    p = PRICE_GRID[a_idx]
                    rate = demand_rate(d, p)
                    ev = 0.0
                    max_q = min(i, int(rate * 5) + 10)
                    for q in range(max_q + 1):
                        prob_q = np.exp(-rate) * (rate ** q) / math.factorial(q)
                        sold = min(q, i)
                        reward = p * sold
                        next_inv = i - sold
                        cont = 0.0
                        for d_next in range(N_DEMAND_REGIMES):
                            cont += DEMAND_TRANS[d][d_next] * V[next_inv, d_next, t - 1]
                        ev += prob_q * (reward + cont)
                    if ev > best_val:
                        best_val = ev
                        best_a = a_idx
                V[i, d, t] = best_val
                policy[i, d, t] = best_a
    return V, policy


# ---------------------------------------------------------------------------
# Behavioral policy & offline data
# ---------------------------------------------------------------------------
def behavioral_action(demand_regime, rng, noise_prob=BEHAVIORAL_NOISE):
    if rng.random() < noise_prob:
        return rng.randint(N_PRICES)
    base_price = BEHAVIORAL_MARKUPS[demand_regime]
    return int(np.clip(base_price - 1, 0, N_PRICES - 1))


def generate_offline_data(n_episodes, rng, noise_prob=BEHAVIORAL_NOISE):
    """Generate offline dataset as a flat list of transitions, episode boundaries preserved
    via the 'episode_id' field on each transition."""
    data = []
    for ep in range(n_episodes):
        inv = MAX_INVENTORY
        d = rng.choice(N_DEMAND_REGIMES)
        for t in range(HORIZON, 0, -1):
            a_idx = behavioral_action(d, rng, noise_prob=noise_prob)
            p = PRICE_GRID[a_idx]
            reward, next_inv, next_d, next_t = step(inv, d, t, p, rng)
            data.append({
                'episode_id': ep, 'inv': inv, 'd': d, 't': t,
                'a_idx': a_idx, 'price': p,
                'reward': reward,
                'next_inv': next_inv, 'next_d': next_d, 'next_t': next_t,
                'done': (next_t == 0),
            })
            inv, d = next_inv, next_d
    return data


def prepare_tensors(data):
    """Flat-transition tensor form used by Q-methods and BC."""
    n = len(data)
    states = np.zeros((n, 3), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n, dtype=np.float32)
    next_states = np.zeros((n, 3), dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    sa_features = np.zeros((n, 4), dtype=np.float32)
    for idx, d in enumerate(data):
        states[idx] = featurize(d['inv'], d['d'], d['t'])
        actions[idx] = d['a_idx']
        rewards[idx] = d['reward']
        next_states[idx] = featurize(d['next_inv'], d['next_d'], d['next_t'])
        dones[idx] = float(d['done'])
        sa_features[idx] = featurize_sa(d['inv'], d['d'], d['t'], d['a_idx'])
    return {
        'states': torch.tensor(states),
        'actions': torch.tensor(actions),
        'rewards': torch.tensor(rewards),
        'next_states': torch.tensor(next_states),
        'dones': torch.tensor(dones),
        'sa_features': torch.tensor(sa_features),
    }


def prepare_trajectories(data):
    """Trajectory-form data for DT/RvS: list of episodes, each a dict of arrays."""
    by_ep = {}
    for d in data:
        by_ep.setdefault(d['episode_id'], []).append(d)
    episodes = []
    for ep_id in sorted(by_ep.keys()):
        steps = by_ep[ep_id]
        T_ep = len(steps)
        rewards = np.array([s['reward'] for s in steps], dtype=np.float32)
        # Absorb terminal salvage into the last reward
        terminal_salvage = steps[-1]['next_inv'] * SALVAGE_VALUE
        rewards_aug = rewards.copy()
        rewards_aug[-1] += terminal_salvage
        # Return-to-go at each step (sum from t to end, including augmented terminal)
        R_to_go = np.cumsum(rewards_aug[::-1])[::-1].copy()
        # States and actions
        states_arr = np.stack(
            [featurize(s['inv'], s['d'], s['t']) for s in steps]
        ).astype(np.float32)
        actions_arr = np.array([s['a_idx'] for s in steps], dtype=np.int64)
        # Previous-action sequence: PAD at t=0, then a_{t-1}
        prev_actions_arr = np.empty(T_ep, dtype=np.int64)
        prev_actions_arr[0] = N_PRICES  # PAD index
        prev_actions_arr[1:] = actions_arr[:-1]
        episodes.append({
            'returns': R_to_go.astype(np.float32),
            'states': states_arr,
            'actions': actions_arr,
            'prev_actions': prev_actions_arr,
            'T': T_ep,
        })
    return episodes


# ---------------------------------------------------------------------------
# Neural net components
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BehaviorCloner(nn.Module):
    def __init__(self, input_dim=3, n_actions=N_PRICES, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DecisionTransformer(nn.Module):
    """Fused-token Decision Transformer. Each position embeds (R_t, s_t, a_{t-1})
    with shared positional embeddings; predicts a_t from the position's hidden state."""
    def __init__(self, state_dim=3, n_actions=N_PRICES,
                 d_model=DT_HIDDEN_DIM, n_heads=DT_N_HEADS, n_layers=DT_N_LAYERS,
                 max_T=HORIZON):
        super().__init__()
        self.return_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(n_actions + 1, d_model)  # +1 for PAD
        self.pos_embed = nn.Embedding(max_T, d_model)
        self.norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, n_actions)

    def forward(self, returns, states, prev_actions, positions):
        # returns: (B, T, 1), states: (B, T, state_dim), prev_actions: (B, T), positions: (B, T)
        B, T = returns.shape[:2]
        x = (self.return_embed(returns)
             + self.state_embed(states)
             + self.action_embed(prev_actions)
             + self.pos_embed(positions))
        x = self.norm(x)
        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.action_head(x)  # (B, T, n_actions)


class RvSNetwork(nn.Module):
    """Return-conditioned MLP per Emmons et al. 2022. Input: (state, return-to-go).
    Output: action logits."""
    def __init__(self, state_dim=3, n_actions=N_PRICES, hidden_dim=RVS_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state, return_to_go):
        x = torch.cat([state, return_to_go], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Per-method training functions (one model per seed)
# ---------------------------------------------------------------------------
def train_bc(tensors, seed):
    torch.manual_seed(seed)
    model = BehaviorCloner()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    states = tensors['states']
    actions = tensors['actions']
    n = len(states)
    for _ in range(N_GRADIENT_STEPS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        logits = model(states[idx])
        loss = criterion(logits, actions[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def policy_fn(inv, d, t):
        s = torch.tensor(featurize(inv, d, t)).unsqueeze(0)
        with torch.no_grad():
            logits = model(s)
        return logits.argmax(dim=1).item()
    return policy_fn


def train_fqi(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    sa_features = tensors['sa_features']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)
    for _ in range(N_FQI_ITERATIONS):
        with torch.no_grad():
            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(q_net(sa_next))
            q_next_max = torch.stack(q_next_all, dim=1).max(dim=1).values
            targets = rewards + (1 - dones) * q_next_max
        for _ in range(3):
            idx = torch.randint(0, n, (BATCH_SIZE,))
            q_pred = q_net(sa_features[idx])
            loss = nn.MSELoss()(q_pred, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def policy_fn(inv, d, t):
        with torch.no_grad():
            q_vals = []
            for a in range(N_PRICES):
                sa = torch.tensor(featurize_sa(inv, d, t, a)).unsqueeze(0)
                q_vals.append(q_net(sa).item())
        return int(np.argmax(q_vals))
    return policy_fn


def train_cql(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    target_q_net = copy.deepcopy(q_net)
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    sa_features = tensors['sa_features']
    states = tensors['states']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)
    for _ in range(N_FQI_ITERATIONS):
        with torch.no_grad():
            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(target_q_net(sa_next))
            q_next_max = torch.stack(q_next_all, dim=1).max(dim=1).values
            targets = rewards + (1 - dones) * q_next_max
        for _ in range(8):
            idx = torch.randint(0, n, (BATCH_SIZE,))
            q_pred = q_net(sa_features[idx])
            bellman_loss = nn.MSELoss()(q_pred, targets[idx])
            s_batch = states[idx]
            q_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((len(idx), 1), a / (N_PRICES - 1))
                sa = torch.cat([s_batch, a_norm], dim=1)
                q_all.append(q_net(sa))
            q_all = torch.stack(q_all, dim=1)
            logsumexp = torch.logsumexp(q_all, dim=1)
            cql_penalty = (logsumexp - q_pred).mean()
            loss = bellman_loss + CQL_ALPHA * cql_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for p, tp in zip(q_net.parameters(), target_q_net.parameters()):
            tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

    def policy_fn(inv, d, t):
        with torch.no_grad():
            q_vals = []
            for a in range(N_PRICES):
                sa = torch.tensor(featurize_sa(inv, d, t, a)).unsqueeze(0)
                q_vals.append(q_net(sa).item())
        return int(np.argmax(q_vals))
    return policy_fn


def train_iql(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    target_q_net = copy.deepcopy(q_net)
    v_net = VNetwork()
    opt_q = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    opt_v = optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
    sa_features = tensors['sa_features']
    states = tensors['states']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)
    for _ in range(N_FQI_ITERATIONS):
        for _ in range(8):
            idx = torch.randint(0, n, (BATCH_SIZE,))
            with torch.no_grad():
                q_vals = target_q_net(sa_features[idx])
            v_vals = v_net(states[idx])
            diff = q_vals - v_vals
            weight = torch.where(diff > 0, IQL_TAU, 1 - IQL_TAU)
            v_loss = (weight * diff ** 2).mean()
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()
            with torch.no_grad():
                v_next = v_net(next_states[idx])
                q_targets = rewards[idx] + (1 - dones[idx]) * v_next
            q_pred = q_net(sa_features[idx])
            q_loss = nn.MSELoss()(q_pred, q_targets)
            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()
            for p, tp in zip(q_net.parameters(), target_q_net.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

    def policy_fn(inv, d, t):
        with torch.no_grad():
            q_vals = []
            for a in range(N_PRICES):
                sa = torch.tensor(featurize_sa(inv, d, t, a)).unsqueeze(0)
                q_vals.append(q_net(sa).item())
        return int(np.argmax(q_vals))
    return policy_fn


def train_bcq(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    bc_model = BehaviorCloner()
    opt_q = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    opt_bc = optim.Adam(bc_model.parameters(), lr=LEARNING_RATE)
    sa_features = tensors['sa_features']
    states = tensors['states']
    actions = tensors['actions']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)
    criterion = nn.CrossEntropyLoss()
    for _ in range(N_GRADIENT_STEPS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        logits = bc_model(states[idx])
        loss = criterion(logits, actions[idx])
        opt_bc.zero_grad()
        loss.backward()
        opt_bc.step()
    for _ in range(N_FQI_ITERATIONS):
        with torch.no_grad():
            bc_logits_next = bc_model(next_states)
            bc_probs_next = torch.softmax(bc_logits_next, dim=1)
            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(q_net(sa_next))
            q_next_all = torch.stack(q_next_all, dim=1)
            max_prob = bc_probs_next.max(dim=1, keepdim=True).values
            mask = (bc_probs_next >= BCQ_THRESHOLD * max_prob).float()
            q_next_masked = q_next_all - 1e8 * (1 - mask)
            q_next_max = q_next_masked.max(dim=1).values
            targets = rewards + (1 - dones) * q_next_max
        for _ in range(3):
            idx = torch.randint(0, n, (BATCH_SIZE,))
            q_pred = q_net(sa_features[idx])
            loss = nn.MSELoss()(q_pred, targets[idx])
            opt_q.zero_grad()
            loss.backward()
            opt_q.step()

    def policy_fn(inv, d, t):
        with torch.no_grad():
            s = torch.tensor(featurize(inv, d, t)).unsqueeze(0)
            bc_logits = bc_model(s)
            bc_probs = torch.softmax(bc_logits, dim=1).squeeze(0)
            max_prob = bc_probs.max()
            mask = (bc_probs >= BCQ_THRESHOLD * max_prob)
            q_vals = []
            for a in range(N_PRICES):
                sa = torch.tensor(featurize_sa(inv, d, t, a)).unsqueeze(0)
                q_vals.append(q_net(sa).item())
            q_vals = np.array(q_vals)
            q_vals[~mask.numpy()] = -np.inf
        return int(np.argmax(q_vals))
    return policy_fn


def train_dt(trajectories, seed):
    """Train a Decision Transformer on the trajectory dataset. Returns a stateful
    policy_fn that maintains return-to-go internally across calls."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DecisionTransformer()
    optimizer = optim.Adam(model.parameters(), lr=DT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # Build tensors from trajectories
    n_eps = len(trajectories)
    T = trajectories[0]['T']
    R_all = np.stack([e['returns'] for e in trajectories]) / DT_RETURN_NORM
    S_all = np.stack([e['states'] for e in trajectories])
    A_prev_all = np.stack([e['prev_actions'] for e in trajectories])
    A_target_all = np.stack([e['actions'] for e in trajectories])
    R_t = torch.tensor(R_all, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
    S_t = torch.tensor(S_all, dtype=torch.float32)
    A_prev_t = torch.tensor(A_prev_all, dtype=torch.long)
    A_tgt_t = torch.tensor(A_target_all, dtype=torch.long)
    Pos_t = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(n_eps, T)

    for _ in range(DT_N_GRADIENT_STEPS):
        idx = torch.randint(0, n_eps, (DT_BATCH_SIZE,))
        logits = model(R_t[idx], S_t[idx], A_prev_t[idx], Pos_t[idx])
        loss = criterion(logits.reshape(-1, N_PRICES), A_tgt_t[idx].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def train_rvs(trajectories, seed):
    """Train RvS network. Returns an RvSNetwork model; evaluation uses the
    return-conditioned protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RvSNetwork()
    optimizer = optim.Adam(model.parameters(), lr=RVS_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # Flatten trajectories into (state, return-to-go) → action triples
    S_flat = np.concatenate([e['states'] for e in trajectories], axis=0)
    R_flat = np.concatenate([e['returns'] for e in trajectories], axis=0) / RVS_RETURN_NORM
    A_flat = np.concatenate([e['actions'] for e in trajectories], axis=0)
    S_t = torch.tensor(S_flat, dtype=torch.float32)
    R_t = torch.tensor(R_flat, dtype=torch.float32).unsqueeze(-1)
    A_t = torch.tensor(A_flat, dtype=torch.long)
    n = len(A_t)

    for _ in range(RVS_N_GRADIENT_STEPS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        logits = model(S_t[idx], R_t[idx])
        loss = criterion(logits, A_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------
def evaluate_policy(policy_fn, n_episodes, rng, dp_policy=None):
    """For Q-methods and BC. policy_fn(inv, d, t) → action_idx."""
    total_rewards = []
    for _ in range(n_episodes):
        inv = MAX_INVENTORY
        d = rng.choice(N_DEMAND_REGIMES)
        episode_reward = 0.0
        for t in range(HORIZON, 0, -1):
            if policy_fn == 'dp':
                a_idx = dp_policy[inv, d, t]
            else:
                a_idx = policy_fn(inv, d, t)
            p = PRICE_GRID[a_idx]
            reward, inv, d, _ = step(inv, d, t, p, rng)
            episode_reward += reward
        episode_reward += inv * SALVAGE_VALUE
        total_rewards.append(episode_reward)
    return np.array(total_rewards)


def evaluate_dt(model, R_star, n_episodes, rng, context_K=DT_CONTEXT_K):
    """Decision Transformer evaluation with growing context and return-to-go tracking."""
    model.eval()
    total_rewards = []
    for _ in range(n_episodes):
        inv = MAX_INVENTORY
        d = rng.choice(N_DEMAND_REGIMES)
        episode_reward = 0.0
        R_to_go = R_star
        prev_action = N_PRICES  # PAD
        returns_seq, states_seq, prev_actions_seq = [], [], []
        for step_idx, t in enumerate(range(HORIZON, 0, -1)):
            s_features = featurize(inv, d, t).astype(np.float32)
            returns_seq.append(R_to_go / DT_RETURN_NORM)
            states_seq.append(s_features)
            prev_actions_seq.append(prev_action)
            start = max(0, len(returns_seq) - context_K)
            R_in = torch.tensor(returns_seq[start:], dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
            S_in = torch.tensor(np.stack(states_seq[start:]), dtype=torch.float32).unsqueeze(0)
            A_in = torch.tensor(prev_actions_seq[start:], dtype=torch.long).unsqueeze(0)
            Pos_in = torch.arange(R_in.size(1), dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                logits = model(R_in, S_in, A_in, Pos_in)
            action_idx = logits[0, -1].argmax().item()
            p = PRICE_GRID[action_idx]
            reward, next_inv, next_d, _ = step(inv, d, t, p, rng)
            episode_reward += reward
            inv = next_inv
            d = next_d
            R_to_go -= reward
            prev_action = action_idx
        episode_reward += inv * SALVAGE_VALUE
        total_rewards.append(episode_reward)
    return np.array(total_rewards)


def evaluate_rvs(model, R_star, n_episodes, rng):
    """RvS evaluation: per-step (state, return-to-go) → action argmax."""
    model.eval()
    total_rewards = []
    for _ in range(n_episodes):
        inv = MAX_INVENTORY
        d = rng.choice(N_DEMAND_REGIMES)
        episode_reward = 0.0
        R_to_go = R_star
        for t in range(HORIZON, 0, -1):
            s_features = featurize(inv, d, t).astype(np.float32)
            S_in = torch.tensor(s_features, dtype=torch.float32).unsqueeze(0)
            R_in = torch.tensor([[R_to_go / RVS_RETURN_NORM]], dtype=torch.float32)
            with torch.no_grad():
                logits = model(S_in, R_in)
            action_idx = logits[0].argmax().item()
            p = PRICE_GRID[action_idx]
            reward, next_inv, next_d, _ = step(inv, d, t, p, rng)
            episode_reward += reward
            inv = next_inv
            d = next_d
            R_to_go -= reward
        episode_reward += inv * SALVAGE_VALUE
        total_rewards.append(episode_reward)
    return np.array(total_rewards)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def compute_shared():
    print("Solving DP oracle...")
    V, dp_policy = solve_dp()
    dp_init_val = V[MAX_INVENTORY, 1, HORIZON]
    print(f"  DP value at (inv={MAX_INVENTORY}, d=1, t={HORIZON}): {dp_init_val:.2f}")
    print(f"Generating {N_SEEDS} offline datasets ({N_OFFLINE_EPISODES} episodes each)...")
    datasets = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        ds = generate_offline_data(N_OFFLINE_EPISODES, rng)
        datasets.append(ds)
    return {
        'dp_policy': dp_policy,
        'dp_value': V,
        'dp_init_val': float(dp_init_val),
        'offline_datasets': datasets,
    }


# ---------------------------------------------------------------------------
# Per-method runners (across seeds)
# ---------------------------------------------------------------------------
def evaluate_dp_seeds(shared):
    """Evaluate the DP oracle on the same eval seeds used by the trained methods."""
    dp_policy = shared['dp_policy']
    returns_per_seed = []
    for seed in range(N_SEEDS):
        rng_eval = np.random.RandomState(seed + 10000)
        r = evaluate_policy('dp', N_EVAL_EPISODES, rng_eval, dp_policy=dp_policy)
        returns_per_seed.append(r.mean())
    returns_per_seed = np.array(returns_per_seed)
    return {
        'returns_per_seed': returns_per_seed,
        'mean': float(returns_per_seed.mean()),
        'se': float(returns_per_seed.std() / np.sqrt(N_SEEDS)),
    }


def _aggregate(returns_per_seed, dp_mean):
    rps = np.array(returns_per_seed)
    return {
        'returns_per_seed': rps,
        'mean': float(rps.mean()),
        'se': float(rps.std() / np.sqrt(N_SEEDS)),
        'pct_optimal': float(rps.mean() / dp_mean * 100),
    }


def run_q_method(shared, train_fn, name):
    """Generic runner for Q-methods and BC. train_fn(tensors, seed) → policy_fn."""
    dp_mean = shared['dp_mean']
    returns_per_seed = []
    for seed in tqdm(range(N_SEEDS), desc=name, leave=False):
        tensors = prepare_tensors(shared['offline_datasets'][seed])
        rng_eval = np.random.RandomState(seed + 10000)
        policy = train_fn(tensors, seed)
        returns = evaluate_policy(policy, N_EVAL_EPISODES, rng_eval)
        returns_per_seed.append(returns.mean())
    return _aggregate(returns_per_seed, dp_mean)


def run_dt_method(shared):
    dp_mean = shared['dp_mean']
    R_star = shared['dp_init_val']
    returns_per_seed = []
    for seed in tqdm(range(N_SEEDS), desc='DT', leave=False):
        trajectories = prepare_trajectories(shared['offline_datasets'][seed])
        rng_eval = np.random.RandomState(seed + 10000)
        model = train_dt(trajectories, seed)
        returns = evaluate_dt(model, R_star, N_EVAL_EPISODES, rng_eval)
        returns_per_seed.append(returns.mean())
    return _aggregate(returns_per_seed, dp_mean)


def run_rvs_method(shared):
    dp_mean = shared['dp_mean']
    R_star = shared['dp_init_val']
    returns_per_seed = []
    for seed in tqdm(range(N_SEEDS), desc='RvS', leave=False):
        trajectories = prepare_trajectories(shared['offline_datasets'][seed])
        rng_eval = np.random.RandomState(seed + 10000)
        model = train_rvs(trajectories, seed)
        returns = evaluate_rvs(model, R_star, N_EVAL_EPISODES, rng_eval)
        returns_per_seed.append(returns.mean())
    return _aggregate(returns_per_seed, dp_mean)


# ---------------------------------------------------------------------------
# Coverage experiment per method
# ---------------------------------------------------------------------------
def run_q_method_coverage(shared, train_fn, name):
    dp_mean = shared['dp_mean']
    result = {}
    for eps_b in EPSILON_B_VALUES:
        pct_vals = []
        for seed in tqdm(range(N_SEEDS), desc=f"{name} eps={eps_b}", leave=False):
            rng_data = np.random.RandomState(seed + 20000)
            rng_eval = np.random.RandomState(seed + 30000)
            offline_data = generate_offline_data(N_OFFLINE_EPISODES, rng_data, noise_prob=eps_b)
            tensors = prepare_tensors(offline_data)
            policy = train_fn(tensors, seed)
            returns = evaluate_policy(policy, N_EVAL_EPISODES, rng_eval)
            pct_vals.append(returns.mean() / dp_mean * 100)
        result[eps_b] = {'mean': float(np.mean(pct_vals)),
                         'se': float(np.std(pct_vals) / np.sqrt(N_SEEDS))}
    return result


def run_dt_coverage(shared):
    dp_mean = shared['dp_mean']
    R_star = shared['dp_init_val']
    result = {}
    for eps_b in EPSILON_B_VALUES:
        pct_vals = []
        for seed in tqdm(range(N_SEEDS), desc=f"DT eps={eps_b}", leave=False):
            rng_data = np.random.RandomState(seed + 20000)
            rng_eval = np.random.RandomState(seed + 30000)
            offline_data = generate_offline_data(N_OFFLINE_EPISODES, rng_data, noise_prob=eps_b)
            trajectories = prepare_trajectories(offline_data)
            model = train_dt(trajectories, seed)
            returns = evaluate_dt(model, R_star, N_EVAL_EPISODES, rng_eval)
            pct_vals.append(returns.mean() / dp_mean * 100)
        result[eps_b] = {'mean': float(np.mean(pct_vals)),
                         'se': float(np.std(pct_vals) / np.sqrt(N_SEEDS))}
    return result


def run_rvs_coverage(shared):
    dp_mean = shared['dp_mean']
    R_star = shared['dp_init_val']
    result = {}
    for eps_b in EPSILON_B_VALUES:
        pct_vals = []
        for seed in tqdm(range(N_SEEDS), desc=f"RvS eps={eps_b}", leave=False):
            rng_data = np.random.RandomState(seed + 20000)
            rng_eval = np.random.RandomState(seed + 30000)
            offline_data = generate_offline_data(N_OFFLINE_EPISODES, rng_data, noise_prob=eps_b)
            trajectories = prepare_trajectories(offline_data)
            model = train_rvs(trajectories, seed)
            returns = evaluate_rvs(model, R_star, N_EVAL_EPISODES, rng_eval)
            pct_vals.append(returns.mean() / dp_mean * 100)
        result[eps_b] = {'mean': float(np.mean(pct_vals)),
                         'se': float(np.std(pct_vals) / np.sqrt(N_SEEDS))}
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
Q_METHODS = {
    'BC':  (train_bc,  BC_CONFIG),
    'FQI': (train_fqi, FQI_CONFIG),
    'CQL': (train_cql, CQL_CONFIG),
    'IQL': (train_iql, IQL_CONFIG),
    'BCQ': (train_bcq, BCQ_CONFIG),
}


# ---------------------------------------------------------------------------
# Top-level compute_data
# ---------------------------------------------------------------------------
def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'shared', SHARED_CONFIG,
        compute_shared, force=('shared' in force))
    # DP-oracle baseline (cached separately so it survives if DP_policy changes)
    dp_result = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'DP_Oracle', SHARED_CONFIG,
        evaluate_dp_seeds, shared, force=('DP_Oracle' in force or 'shared' in force))
    shared['dp_mean'] = dp_result['mean']
    main_results = {'DP Oracle': {**dp_result, 'pct_optimal': 100.0}}
    # Q-methods + BC
    for name, (train_fn, cfg) in Q_METHODS.items():
        r = compute_or_load(
            CACHE_DIR, SCRIPT_NAME, name, cfg,
            run_q_method, shared, train_fn, name,
            force=(name in force or 'shared' in force))
        main_results[name] = r
    # DT
    r = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'DT', DT_CONFIG,
        run_dt_method, shared,
        force=('DT' in force or 'shared' in force))
    main_results['DT'] = r
    # RvS
    r = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'RvS', RVS_CONFIG,
        run_rvs_method, shared,
        force=('RvS' in force or 'shared' in force))
    main_results['RvS'] = r
    # Coverage experiment (per-method caching)
    coverage_results = {}
    for name, (train_fn, cfg) in Q_METHODS.items():
        cov_key = f'coverage_{name}'
        r = compute_or_load(
            CACHE_DIR, SCRIPT_NAME, cov_key, cfg,
            run_q_method_coverage, shared, train_fn, name,
            force=(cov_key in force or name in force or 'shared' in force))
        coverage_results[name] = r
    coverage_results['DT'] = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'coverage_DT', DT_CONFIG,
        run_dt_coverage, shared,
        force=('coverage_DT' in force or 'DT' in force or 'shared' in force))
    coverage_results['RvS'] = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'coverage_RvS', RVS_CONFIG,
        run_rvs_coverage, shared,
        force=('coverage_RvS' in force or 'RvS' in force or 'shared' in force))
    return {
        'shared': {k: v for k, v in shared.items() if k != 'offline_datasets'},
        'main_results': main_results,
        'coverage_results': coverage_results,
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
ALL_METHODS = ['DP Oracle', 'BC', 'FQI', 'CQL', 'IQL', 'BCQ', 'DT', 'RvS']


def _rank_ordered(main_results):
    """Return method names sorted by mean return descending, with DP Oracle first."""
    trained = [m for m in ALL_METHODS if m != 'DP Oracle' and m in main_results]
    trained.sort(key=lambda m: -main_results[m]['mean'])
    return ['DP Oracle'] + trained


def generate_outputs(data):
    main_results = data['main_results']
    coverage_results = data['coverage_results']
    order = _rank_ordered(main_results)

    # --- Table ---
    tex_lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Method & Mean Return & \% of Optimal \\",
        r"\hline",
    ]
    for name in order:
        r = main_results[name]
        ret_str = f"${r['mean']:.2f} \\pm {r['se']:.2f}$"
        pct_str = f"${r.get('pct_optimal', 100.0):.1f}\\%$"
        tex_lines.append(f"{name} & {ret_str} & {pct_str} \\\\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_path = os.path.join(OUTPUT_DIR, "offline_rl_pricing_results.tex")
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"Saved table: {tex_path}")

    # --- Coverage figure ---
    method_names = [m for m in order if m != 'DP Oracle']
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    color_map = {
        'BC':  DOMAIN_COLORS.get('BC',  COLORS['gray']),
        'FQI': DOMAIN_COLORS.get('FQI', COLORS.get('blue', '#1f77b4')),
        'CQL': DOMAIN_COLORS.get('CQL', COLORS.get('red', '#d62728')),
        'IQL': DOMAIN_COLORS.get('IQL', COLORS.get('green', '#2ca02c')),
        'BCQ': DOMAIN_COLORS.get('BCQ', COLORS.get('purple', '#9467bd')),
        'DT':  COLORS.get('orange', '#ff7f0e'),
        'RvS': COLORS.get('brown', '#8c564b'),
    }
    for name in method_names:
        if name not in coverage_results:
            continue
        means = [coverage_results[name][eps]['mean'] for eps in EPSILON_B_VALUES]
        ses = [coverage_results[name][eps]['se'] for eps in EPSILON_B_VALUES]
        c = color_map.get(name, COLORS['gray'])
        ax.plot(EPSILON_B_VALUES, means, 'o-', color=c, label=name, linewidth=1.8)
        ax.fill_between(EPSILON_B_VALUES,
                        np.array(means) - np.array(ses),
                        np.array(means) + np.array(ses),
                        color=c, alpha=0.15)
    ax.axhline(100, **BENCH_STYLE, label='DP Optimal')
    ax.set_xlabel(r"Behavioral policy randomness $\epsilon_b$")
    ax.set_ylabel(r"Policy value (\% of DP optimal)")
    ax.set_title("Coverage Sensitivity")
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim(EPSILON_B_VALUES[0] - 0.02, EPSILON_B_VALUES[-1] + 0.02)
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(OUTPUT_DIR, "offline_rl_pricing_coverage.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # --- Stdout summary ---
    print("\n=== Main comparison (rank order) ===")
    print(f"{'Method':<12} {'Mean':>10} {'SE':>8} {'% Optimal':>10}")
    for name in order:
        r = main_results[name]
        print(f"{name:<12} {r['mean']:>10.2f} {r['se']:>8.2f} {r.get('pct_optimal', 100.0):>9.1f}%")
    print("\n=== Coverage sensitivity (% of DP optimal) ===")
    print(f"{'Method':<6} " + " ".join(f"eps={e:<5}" for e in EPSILON_B_VALUES))
    for name in method_names:
        if name not in coverage_results:
            continue
        cells = [f"{coverage_results[name][e]['mean']:>5.1f}±{coverage_results[name][e]['se']:.1f}"
                 for e in EPSILON_B_VALUES]
        print(f"{name:<6} " + " ".join(cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    if args.plots_only:
        # Load from cache without recomputing
        data = compute_data(force=set())
    else:
        data = compute_data(force=force)
    if not args.data_only:
        generate_outputs(data)
