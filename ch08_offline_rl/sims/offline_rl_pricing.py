# Offline RL for Perishable Inventory Pricing — Chapter 8, Offline Reinforcement Learning.
# Compares FQI, CQL, IQL, BCQ against DP optimal and behavioral cloning on a
# finite-horizon perishable inventory pricing MDP with demand regime switching.

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
from sims.plot_style import apply_style, COLORS, DOMAIN_COLORS, BENCH_STYLE, FIG_SINGLE
from sims.sim_cache import load_results, save_results, add_cache_args
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

# Behavioral policy parameters
BEHAVIORAL_MARKUPS = np.array([10, 10, 10, 10], dtype=float)  # always maximum price
BEHAVIORAL_NOISE = 0.15  # 85% at price 10, 15% uniform exploration

# Experiment parameters
N_OFFLINE_EPISODES = 500
N_EVAL_EPISODES = 1000
N_SEEDS = 20

# Neural net parameters
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
N_FQI_ITERATIONS = 200
BATCH_SIZE = 256
N_GRADIENT_STEPS = 300

# CQL alpha
CQL_ALPHA = 0.1

# IQL expectile
IQL_TAU = 0.7

# BCQ threshold
BCQ_THRESHOLD = 0.3

# Coverage sensitivity experiment
EPSILON_B_VALUES = [0.05, 0.3, 0.9]

CONFIG = {
    'MAX_INVENTORY': MAX_INVENTORY,
    'N_DEMAND_REGIMES': N_DEMAND_REGIMES,
    'HORIZON': HORIZON,
    'N_PRICES': N_PRICES,
    'LAMBDA_0': LAMBDA_0.tolist(),
    'PRICE_SENSITIVITY': PRICE_SENSITIVITY,
    'SALVAGE_VALUE': SALVAGE_VALUE,
    'N_OFFLINE_EPISODES': N_OFFLINE_EPISODES,
    'N_EVAL_EPISODES': N_EVAL_EPISODES,
    'N_SEEDS': N_SEEDS,
    'HIDDEN_DIM': HIDDEN_DIM,
    'N_FQI_ITERATIONS': N_FQI_ITERATIONS,
    'N_GRADIENT_STEPS': N_GRADIENT_STEPS,
    'CQL_ALPHA': CQL_ALPHA,
    'IQL_TAU': IQL_TAU,
    'BCQ_THRESHOLD': BCQ_THRESHOLD,
    'EPSILON_B_VALUES': EPSILON_B_VALUES,
    'version': 12,
}


# ---------------------------------------------------------------------------
# MDP: Perishable Inventory Pricing
# ---------------------------------------------------------------------------
def demand_rate(demand_regime, price):
    """Expected demand: Poisson rate lambda_0[d] * exp(-alpha * p)."""
    return LAMBDA_0[demand_regime] * np.exp(-PRICE_SENSITIVITY * price)


def sample_demand(demand_regime, price, rng):
    """Draw demand from Poisson(lambda(d, p))."""
    rate = demand_rate(demand_regime, price)
    return rng.poisson(rate)


def step(inventory, demand_regime, time_remaining, price, rng):
    """Execute one step of the MDP. Returns (reward, next_inv, next_d, next_t)."""
    if time_remaining <= 0:
        return inventory * SALVAGE_VALUE, inventory, demand_regime, 0

    q = sample_demand(demand_regime, price, rng)
    sold = min(q, inventory)
    reward = price * sold
    next_inv = inventory - sold
    next_d = rng.choice(N_DEMAND_REGIMES, p=DEMAND_TRANS[demand_regime])
    next_t = time_remaining - 1
    return reward, next_inv, next_d, next_t


# ---------------------------------------------------------------------------
# State featurization (normalized to [0, 1])
# ---------------------------------------------------------------------------
def featurize(inventory, demand_regime, time_remaining):
    """Convert state to normalized feature vector."""
    inv_norm = inventory / MAX_INVENTORY
    d_norm = demand_regime / (N_DEMAND_REGIMES - 1)
    t_norm = time_remaining / HORIZON
    return np.array([inv_norm, d_norm, t_norm], dtype=np.float32)


def featurize_sa(inventory, demand_regime, time_remaining, action_idx):
    """State-action features for Q-network input."""
    s = featurize(inventory, demand_regime, time_remaining)
    a_norm = np.float32(action_idx / (N_PRICES - 1))
    return np.concatenate([s, [a_norm]]).astype(np.float32)


# ---------------------------------------------------------------------------
# DP Oracle: Backward Induction (exact tabular)
# ---------------------------------------------------------------------------
def solve_dp():
    """Compute exact optimal value function and policy via backward induction."""
    # V[i, d, t] = optimal value from state (inventory=i, demand_regime=d, time_remaining=t)
    V = np.zeros((MAX_INVENTORY + 1, N_DEMAND_REGIMES, HORIZON + 1))
    policy = np.zeros((MAX_INVENTORY + 1, N_DEMAND_REGIMES, HORIZON + 1), dtype=int)

    # Terminal: salvage value
    for i in range(MAX_INVENTORY + 1):
        for d in range(N_DEMAND_REGIMES):
            V[i, d, 0] = i * SALVAGE_VALUE

    # Backward induction
    for t in range(1, HORIZON + 1):
        for i in range(MAX_INVENTORY + 1):
            for d in range(N_DEMAND_REGIMES):
                best_val = -np.inf
                best_a = 0
                for a_idx in range(N_PRICES):
                    p = PRICE_GRID[a_idx]
                    rate = demand_rate(d, p)
                    # Expected reward and continuation
                    ev = 0.0
                    # Sum over possible demand realizations
                    max_q = min(i, int(rate * 5) + 10)  # truncate Poisson
                    for q in range(max_q + 1):
                        prob_q = np.exp(-rate) * (rate ** q) / math.factorial(q)
                        sold = min(q, i)
                        reward = p * sold
                        next_inv = i - sold
                        # Expected continuation over next demand regime
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
# Behavioral Policy (data generation)
# ---------------------------------------------------------------------------
def behavioral_action(demand_regime, rng, noise_prob=BEHAVIORAL_NOISE):
    """Stochastic markup rule ignoring inventory and time."""
    if rng.random() < noise_prob:
        return rng.randint(N_PRICES)
    else:
        base_price = BEHAVIORAL_MARKUPS[demand_regime]
        a_idx = int(np.clip(base_price - 1, 0, N_PRICES - 1))
        return a_idx


def generate_offline_data(n_episodes, rng, noise_prob=BEHAVIORAL_NOISE):
    """Generate offline dataset from behavioral policy."""
    data = []
    for _ in range(n_episodes):
        inv = MAX_INVENTORY  # always start at full stock
        d = rng.choice(N_DEMAND_REGIMES)
        for t in range(HORIZON, 0, -1):
            a_idx = behavioral_action(d, rng, noise_prob=noise_prob)
            p = PRICE_GRID[a_idx]
            reward, next_inv, next_d, next_t = step(inv, d, t, p, rng)
            data.append({
                'inv': inv, 'd': d, 't': t,
                'a_idx': a_idx, 'price': p,
                'reward': reward,
                'next_inv': next_inv, 'next_d': next_d, 'next_t': next_t,
                'done': (next_t == 0),
            })
            inv, d = next_inv, next_d
    return data


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BehaviorCloner(nn.Module):
    def __init__(self, input_dim=3, n_actions=N_PRICES, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_tensors(data):
    """Convert list-of-dicts offline data to tensors."""
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


# ---------------------------------------------------------------------------
# Method 1: Behavioral Cloning (BC)
# ---------------------------------------------------------------------------
def train_bc(tensors, seed):
    torch.manual_seed(seed)
    model = BehaviorCloner()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    states = tensors['states']
    actions = tensors['actions']
    n = len(states)

    for epoch in range(N_GRADIENT_STEPS):
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


# ---------------------------------------------------------------------------
# Method 2: Fitted Q-Iteration (FQI)
# ---------------------------------------------------------------------------
def train_fqi(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    sa_features = tensors['sa_features']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)

    for fqi_iter in range(N_FQI_ITERATIONS):
        # Compute targets: max_a Q(s', a) — the source of overestimation
        with torch.no_grad():
            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(q_net(sa_next))
            q_next_max = torch.stack(q_next_all, dim=1).max(dim=1).values
            targets = rewards + (1 - dones) * q_next_max

        # Fit Q to targets
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


# ---------------------------------------------------------------------------
# Method 3: Conservative Q-Learning (CQL)
# ---------------------------------------------------------------------------
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

    for fqi_iter in range(N_FQI_ITERATIONS):
        with torch.no_grad():
            # Use TARGET network for stable next-state Q-values
            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(target_q_net(sa_next))
            q_next_max = torch.stack(q_next_all, dim=1).max(dim=1).values
            targets = rewards + (1 - dones) * q_next_max

        for _ in range(8):
            idx = torch.randint(0, n, (BATCH_SIZE,))

            # Standard Bellman loss
            q_pred = q_net(sa_features[idx])
            bellman_loss = nn.MSELoss()(q_pred, targets[idx])

            # CQL penalty: log-sum-exp over all actions minus Q at data action
            s_batch = states[idx]
            q_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((len(idx), 1), a / (N_PRICES - 1))
                sa = torch.cat([s_batch, a_norm], dim=1)
                q_all.append(q_net(sa))
            q_all = torch.stack(q_all, dim=1)  # (batch, n_actions)
            logsumexp = torch.logsumexp(q_all, dim=1)
            cql_penalty = (logsumexp - q_pred).mean()

            loss = bellman_loss + CQL_ALPHA * cql_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Soft-update target Q-network
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


# ---------------------------------------------------------------------------
# Method 4: Implicit Q-Learning (IQL)
# ---------------------------------------------------------------------------
def train_iql(tensors, seed):
    torch.manual_seed(seed)
    q_net = QNetwork()
    target_q_net = copy.deepcopy(q_net)
    v_net = VNetwork()
    opt_q = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    opt_v = optim.Adam(v_net.parameters(), lr=LEARNING_RATE)

    sa_features = tensors['sa_features']
    states = tensors['states']
    actions = tensors['actions']
    rewards = tensors['rewards']
    next_states = tensors['next_states']
    dones = tensors['dones']
    n = len(rewards)

    for iql_iter in range(N_FQI_ITERATIONS):
        for _ in range(8):
            idx = torch.randint(0, n, (BATCH_SIZE,))

            # V update: expectile regression on Q_target(s,a) - V(s)
            with torch.no_grad():
                q_vals = target_q_net(sa_features[idx])
            v_vals = v_net(states[idx])
            diff = q_vals - v_vals
            weight = torch.where(diff > 0, IQL_TAU, 1 - IQL_TAU)
            v_loss = (weight * diff ** 2).mean()
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()

            # Q update: standard Bellman with V as continuation
            with torch.no_grad():
                v_next = v_net(next_states[idx])
                q_targets = rewards[idx] + (1 - dones[idx]) * v_next
            q_pred = q_net(sa_features[idx])
            q_loss = nn.MSELoss()(q_pred, q_targets)
            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()

            # Soft-update target Q-network (EMA)
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


# ---------------------------------------------------------------------------
# Method 5: Batch-Constrained Q-learning (BCQ)
# ---------------------------------------------------------------------------
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

    # Pre-train behavioral model
    criterion = nn.CrossEntropyLoss()
    for _ in range(N_GRADIENT_STEPS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        logits = bc_model(states[idx])
        loss = criterion(logits, actions[idx])
        opt_bc.zero_grad()
        loss.backward()
        opt_bc.step()

    # FQI with BCQ constraint
    for fqi_iter in range(N_FQI_ITERATIONS):
        with torch.no_grad():
            # Behavioral action probabilities at next states
            bc_logits_next = bc_model(next_states)
            bc_probs_next = torch.softmax(bc_logits_next, dim=1)

            q_next_all = []
            for a in range(N_PRICES):
                a_norm = torch.full((n, 1), a / (N_PRICES - 1))
                sa_next = torch.cat([next_states, a_norm], dim=1)
                q_next_all.append(q_net(sa_next))
            q_next_all = torch.stack(q_next_all, dim=1)  # (n, n_actions)

            # Mask: only consider actions with prob > threshold * max_prob
            max_prob = bc_probs_next.max(dim=1, keepdim=True).values
            mask = (bc_probs_next >= BCQ_THRESHOLD * max_prob).float()
            # Set masked-out Q-values to -inf
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


# ---------------------------------------------------------------------------
# Policy Evaluation
# ---------------------------------------------------------------------------
def evaluate_policy(policy_fn, n_episodes, rng, dp_policy=None):
    """Evaluate a policy over n_episodes. Returns mean total reward."""
    total_rewards = []
    for _ in range(n_episodes):
        inv = MAX_INVENTORY  # match data-generating process
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
        # Terminal salvage
        episode_reward += inv * SALVAGE_VALUE
        total_rewards.append(episode_reward)
    return np.array(total_rewards)


# ---------------------------------------------------------------------------
# Experiment 1: Main comparison
# ---------------------------------------------------------------------------
def run_main_comparison(dp_policy, dp_value):
    print("\n=== Experiment 1: Main Comparison ===")
    print(f"Offline episodes: {N_OFFLINE_EPISODES}, Eval episodes: {N_EVAL_EPISODES}, Seeds: {N_SEEDS}")

    method_names = ['DP Oracle', 'BC', 'FQI', 'CQL', 'IQL', 'BCQ']
    train_fns = [None, train_bc, train_fqi, train_cql, train_iql, train_bcq]

    all_returns = {name: [] for name in method_names}

    for seed in tqdm(range(N_SEEDS), desc="Main comparison"):
        rng_data = np.random.RandomState(seed)
        rng_eval = np.random.RandomState(seed + 10000)

        # Generate offline data
        offline_data = generate_offline_data(N_OFFLINE_EPISODES, rng_data)
        tensors = prepare_tensors(offline_data)

        # Evaluate DP oracle
        dp_returns = evaluate_policy('dp', N_EVAL_EPISODES, rng_eval, dp_policy=dp_policy)
        all_returns['DP Oracle'].append(dp_returns.mean())

        # Train and evaluate each method
        for name, train_fn in zip(method_names[1:], train_fns[1:]):
            rng_eval_method = np.random.RandomState(seed + 10000)
            policy = train_fn(tensors, seed)
            returns = evaluate_policy(policy, N_EVAL_EPISODES, rng_eval_method)
            all_returns[name].append(returns.mean())

    # Compute statistics
    results = {}
    dp_mean = np.mean(all_returns['DP Oracle'])
    for name in method_names:
        vals = np.array(all_returns[name])
        results[name] = {
            'mean': vals.mean(),
            'se': vals.std() / np.sqrt(N_SEEDS),
            'pct_optimal': vals.mean() / dp_mean * 100,
        }
        print(f"  {name:<12}: {vals.mean():.2f} ± {vals.std()/np.sqrt(N_SEEDS):.2f}  "
              f"({vals.mean()/dp_mean*100:.1f}% of optimal)")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Coverage sensitivity
# ---------------------------------------------------------------------------
def run_coverage_experiment(dp_policy):
    print("\n=== Experiment 2: Coverage Sensitivity ===")
    method_names = ['BC', 'FQI', 'CQL', 'IQL', 'BCQ']
    train_fns = [train_bc, train_fqi, train_cql, train_iql, train_bcq]

    # Also compute DP optimal for normalization
    dp_returns_all = []
    for seed in range(N_SEEDS):
        rng_eval = np.random.RandomState(seed + 10000)
        dp_returns_all.append(evaluate_policy('dp', N_EVAL_EPISODES, rng_eval, dp_policy=dp_policy).mean())
    dp_mean = np.mean(dp_returns_all)

    coverage_results = {name: {} for name in method_names}

    for eps_b in EPSILON_B_VALUES:
        print(f"\n  epsilon_b = {eps_b}")
        for name, train_fn in zip(method_names, train_fns):
            pct_vals = []
            for seed in tqdm(range(N_SEEDS), desc=f"  {name} eps={eps_b}", leave=False):
                rng_data = np.random.RandomState(seed + 20000)
                rng_eval = np.random.RandomState(seed + 30000)

                offline_data = generate_offline_data(N_OFFLINE_EPISODES, rng_data, noise_prob=eps_b)
                tensors = prepare_tensors(offline_data)
                policy = train_fn(tensors, seed)
                returns = evaluate_policy(policy, N_EVAL_EPISODES, rng_eval)
                pct_vals.append(returns.mean() / dp_mean * 100)

            coverage_results[name][eps_b] = {
                'mean': np.mean(pct_vals),
                'se': np.std(pct_vals) / np.sqrt(N_SEEDS),
            }
            print(f"    {name:<6}: {np.mean(pct_vals):.1f}% ± {np.std(pct_vals)/np.sqrt(N_SEEDS):.1f}%")

    return coverage_results


# ---------------------------------------------------------------------------
# Compute all experiments
# ---------------------------------------------------------------------------
def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("Solving DP oracle...")
    V, dp_policy = solve_dp()
    # Report DP value at a representative initial state
    init_val = V[25, 1, HORIZON]
    print(f"  DP value at (inv=25, d=1, t={HORIZON}): {init_val:.2f}")

    # Experiment 1
    main_results = run_main_comparison(dp_policy, V)

    # Experiment 2
    coverage_results = run_coverage_experiment(dp_policy)

    data = {
        'main_results': main_results,
        'coverage_results': coverage_results,
        'dp_init_val': init_val,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    print(f"  Cache saved: {os.path.join(CACHE_DIR, SCRIPT_NAME + '.pkl')}")
    return data


# ---------------------------------------------------------------------------
# Generate outputs
# ---------------------------------------------------------------------------
def generate_outputs(data):
    main_results = data['main_results']
    coverage_results = data['coverage_results']

    # --- Table 1: Main comparison ---
    method_order = ['DP Oracle', 'BC', 'FQI', 'CQL', 'IQL', 'BCQ']
    tex_lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Method & Mean Return & \% of Optimal \\",
        r"\hline",
    ]
    for name in method_order:
        r = main_results[name]
        ret_str = f"${r['mean']:.2f} \\pm {r['se']:.2f}$"
        pct_str = f"${r['pct_optimal']:.1f}\\%$"
        tex_lines.append(f"{name} & {ret_str} & {pct_str} \\\\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")

    tex_path = os.path.join(OUTPUT_DIR, "offline_rl_pricing_results.tex")
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"Saved table: {tex_path}")

    # --- Figure: Coverage sensitivity ---
    method_names = ['BC', 'FQI', 'CQL', 'IQL', 'BCQ']
    method_colors = [DOMAIN_COLORS.get(n, COLORS['gray']) for n in method_names]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name, color in zip(method_names, method_colors):
        means = [coverage_results[name][eps]['mean'] for eps in EPSILON_B_VALUES]
        ses = [coverage_results[name][eps]['se'] for eps in EPSILON_B_VALUES]
        ax.plot(EPSILON_B_VALUES, means, 'o-', color=color, label=name, linewidth=1.8)
        ax.fill_between(EPSILON_B_VALUES,
                        np.array(means) - np.array(ses),
                        np.array(means) + np.array(ses),
                        color=color, alpha=0.15)

    ax.axhline(100, **BENCH_STYLE, label='DP Optimal')
    ax.set_xlabel(r"Behavioral policy randomness $\epsilon_b$")
    ax.set_ylabel("Policy value (\\% of DP optimal)")
    ax.set_title("Coverage Sensitivity")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(EPSILON_B_VALUES[0] - 0.02, EPSILON_B_VALUES[-1] + 0.02)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(OUTPUT_DIR, "offline_rl_pricing_coverage.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()
    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
