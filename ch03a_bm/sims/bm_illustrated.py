"""
Brock-Mirman Illustrated Example: 8 Algorithms on a Stochastic Growth Model
Chapter 3a — VI, PI, LP, Q-Learning, SARSA, DQN, REINFORCE, PPO on BM economy.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, COLORS, FIG_SINGLE, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()

import argparse
import time

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ============================================================================
# Parameters
# ============================================================================

ALPHA_PROD = 0.36       # capital share
BETA = 0.96             # discount factor
Z_VALS = np.array([0.9, 1.1])
PI_TRANS = np.array([[0.8, 0.2],
                     [0.2, 0.8]])

N_K = 50                # capital grid points
N_Z = len(Z_VALS)
N_S = N_K * N_Z         # 100 states
N_A = N_K               # 50 actions

SEED = 42
TOL = 1e-10
MAX_ITER_VI = 5000
MAX_ITER_PI = 50

# RL hyperparameters
# Visit-count learning rate: alpha(s,a) = QL_ALPHA_C / (QL_ALPHA_C + N(s,a))
QL_EPISODES = 500_000
QL_HORIZON = 100
QL_ALPHA_C = 100.0        # visit-count learning rate constant
QL_EPS_START = 1.0
QL_EPS_END = 0.05
QL_EPS_DECAY = 0.99999
QL_EVAL_FREQ = 100

SARSA_EPISODES = 500_000
SARSA_HORIZON = 100
SARSA_ALPHA_C = 100.0
SARSA_EPS_START = 1.0
SARSA_EPS_END = 0.05
SARSA_EPS_DECAY = 0.99999
SARSA_EVAL_FREQ = 100

DQN_EPISODES = 50_000
DQN_HORIZON = 100
DQN_REPLAY_SIZE = 50_000
DQN_BATCH_SIZE = 128
DQN_LR = 3e-4
DQN_EPS_START = 1.0
DQN_EPS_END = 0.01
DQN_EPS_DECAY = 0.9999
DQN_TARGET_UPDATE = 100
DQN_HIDDEN = 128
DQN_TRAIN_FREQ = 4          # train every N steps
DQN_EVAL_FREQ = 500

REINFORCE_EPISODES = 500_000
REINFORCE_HORIZON = 100
REINFORCE_ALPHA = 0.005
REINFORCE_ALPHA_DECAY = 0.99999
REINFORCE_TAU = 1.0
REINFORCE_EVAL_FREQ = 200

PPO_EPISODES = 500_000
PPO_HORIZON = 100
PPO_ALPHA = 0.005
PPO_ALPHA_DECAY = 0.99999
PPO_TAU = 1.0
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_GAE_LAMBDA = 0.95
PPO_EVAL_FREQ = 200

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

ALGO_NAMES = ['VI', 'PI', 'LP', 'Q-Learning', 'SARSA', 'DQN', 'REINFORCE', 'PPO']
ALGO_CACHE_KEYS = {
    'VI': 'vi', 'PI': 'pi', 'LP': 'lp',
    'Q-Learning': 'ql', 'SARSA': 'sarsa', 'DQN': 'dqn',
    'REINFORCE': 'reinforce', 'PPO': 'ppo',
}

# ============================================================================
# Model helpers (ported from brock_mirman_newton.py)
# ============================================================================

def build_grid(n_k):
    k_ss_high = (ALPHA_PROD * BETA * Z_VALS.max()) ** (1 / (1 - ALPHA_PROD))
    return np.linspace(0.01, 1.5 * k_ss_high, n_k)


def build_reward_and_transitions(k_grid):
    n_k = len(k_grid)
    n_z = len(Z_VALS)
    n_s = n_k * n_z
    n_a = n_k

    R = np.full((n_s, n_a), -np.inf)
    P = np.zeros((n_s, n_a, n_s))

    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            output = Z_VALS[iz] * k_grid[ik] ** ALPHA_PROD
            for ia in range(n_a):
                c = output - k_grid[ia]
                if c > 1e-12:
                    R[s, ia] = np.log(c)
                    for iz_next in range(n_z):
                        s_next = ia * n_z + iz_next
                        P[s, ia, s_next] = PI_TRANS[iz, iz_next]

    return R, P


def closed_form_policy(k_grid):
    n_k = len(k_grid)
    n_z = len(Z_VALS)
    policy = np.zeros(n_k * n_z, dtype=int)
    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            k_next = ALPHA_PROD * BETA * Z_VALS[iz] * k_grid[ik] ** ALPHA_PROD
            policy[s] = np.argmin(np.abs(k_grid - k_next))
    return policy


def feasible_mask(R):
    return R > -1e30


# ============================================================================
# BrockMirmanEnv (gym-like interface for RL algorithms)
# ============================================================================

class BrockMirmanEnv:
    def __init__(self, k_grid, R, P):
        self.k_grid = k_grid
        self.R = R
        self.P = P
        self.n_k = len(k_grid)
        self.n_z = N_Z
        self.num_states = self.n_k * self.n_z
        self.num_actions = self.n_k
        self.gamma = BETA
        self._feasible = feasible_mask(R)

    def state_to_index(self, s):
        ik, iz = s
        return ik * self.n_z + iz

    def index_to_state(self, idx):
        ik = idx // self.n_z
        iz = idx % self.n_z
        return (ik, iz)

    def reset(self):
        ik = np.random.randint(self.n_k)
        iz = np.random.randint(self.n_z)
        self._state = (ik, iz)
        return self._state

    def step(self, action):
        ik, iz = self._state
        s_idx = ik * self.n_z + iz
        r = self.R[s_idx, action]
        # Sample next z
        iz_next = np.random.choice(self.n_z, p=PI_TRANS[iz])
        self._state = (action, iz_next)  # action = ik'
        return self._state, r, False  # never done

    def get_feasible_actions(self, s):
        s_idx = self.state_to_index(s)
        return np.where(self._feasible[s_idx])[0]


# ============================================================================
# Caching (via sims.sim_cache)
# ============================================================================

def _algo_full_config(algo_name):
    """Full config dict for cache hashing (shared params + algo-specific)."""
    configs = {
        'VI': dict(n_k=N_K, tol=TOL),
        'PI': dict(n_k=N_K),
        'LP': dict(n_k=N_K),
        'Q-Learning': dict(n_k=N_K, ep=QL_EPISODES, h=QL_HORIZON,
                           ac=QL_ALPHA_C,
                           es=QL_EPS_START, ee=QL_EPS_END, ed=QL_EPS_DECAY),
        'SARSA': dict(n_k=N_K, ep=SARSA_EPISODES, h=SARSA_HORIZON,
                      ac=SARSA_ALPHA_C,
                      es=SARSA_EPS_START, ee=SARSA_EPS_END, ed=SARSA_EPS_DECAY),
        'DQN': dict(n_k=N_K, ep=DQN_EPISODES, h=DQN_HORIZON,
                    rs=DQN_REPLAY_SIZE, bs=DQN_BATCH_SIZE, lr=DQN_LR,
                    hd=DQN_HIDDEN, tu=DQN_TARGET_UPDATE, tf=DQN_TRAIN_FREQ),
        'REINFORCE': dict(n_k=N_K, ep=REINFORCE_EPISODES, h=REINFORCE_HORIZON,
                          a=REINFORCE_ALPHA, ad=REINFORCE_ALPHA_DECAY, tau=REINFORCE_TAU),
        'PPO': dict(n_k=N_K, ep=PPO_EPISODES, h=PPO_HORIZON,
                    a=PPO_ALPHA, ad=PPO_ALPHA_DECAY, tau=PPO_TAU,
                    clip=PPO_CLIP, epochs=PPO_EPOCHS, gae=PPO_GAE_LAMBDA),
    }
    return {'algo': algo_name, 'alpha': ALPHA_PROD, 'beta': BETA,
            'z': Z_VALS.tolist(), 'pi': PI_TRANS.tolist(), 'seed': SEED,
            'version': 1, **configs[algo_name]}


def load_cache(algo_name):
    key = ALGO_CACHE_KEYS[algo_name]
    return load_results(CACHE_DIR, key + '_results', _algo_full_config(algo_name))


def save_cache(algo_name, data):
    key = ALGO_CACHE_KEYS[algo_name]
    save_results(CACHE_DIR, key + '_results', _algo_full_config(algo_name), data)


# ============================================================================
# Closed-form solution and evaluation helpers
# ============================================================================

def compute_closed_form_V(k_grid, R, P):
    """Get V* from VI (used as ground truth)."""
    V, _, _ = value_iteration(R, P, BETA)
    return V


def policy_agreement(pol, cf_pol):
    return np.mean(pol == cf_pol) * 100


def policy_agreement_soft(pol, cf_pol, tol=1):
    """Agreement allowing ±tol grid points."""
    return np.mean(np.abs(pol.astype(int) - cf_pol.astype(int)) <= tol) * 100


def value_rmse(V, V_star):
    return np.sqrt(np.mean((V - V_star) ** 2))


def value_max_error(V, V_star):
    return np.max(np.abs(V - V_star))


def extract_V_from_Q(Q, n_s, n_a):
    return np.max(Q, axis=1)


def extract_policy_from_Q(Q, n_s, n_a):
    return np.argmax(Q, axis=1)


# ============================================================================
# Planning: VI
# ============================================================================

def value_iteration(R, P, gamma, tol=TOL, max_iter=MAX_ITER_VI):
    n_s, n_a = R.shape
    V = np.zeros(n_s)
    errors = []
    for it in range(max_iter):
        Q = R + gamma * (P @ V)
        V_new = np.max(Q, axis=1)
        err = np.max(np.abs(V_new - V))
        errors.append(err)
        V = V_new
        if err < tol:
            break
    policy = np.argmax(R + gamma * (P @ V), axis=1)
    return V, policy, errors


def run_vi(R, P):
    print("  Running VI...", end=" ", flush=True)
    t0 = time.perf_counter()
    V, pol, errs = value_iteration(R, P, BETA)
    t = time.perf_counter() - t0
    print(f"{len(errs)} iters, {t:.2f}s")
    return {'V': V, 'policy': pol, 'errors': errs, 'time': t, 'iters': len(errs)}


# ============================================================================
# Planning: PI
# ============================================================================

def policy_iteration(R, P, gamma, max_iter=MAX_ITER_PI):
    n_s, n_a = R.shape
    policy = np.argmax(R, axis=1)
    errors = []

    V_history = []
    t0_total = time.perf_counter()
    for it in range(max_iter):
        r_pi = R[np.arange(n_s), policy]
        P_pi = P[np.arange(n_s), policy, :]
        A_mat = np.eye(n_s) - gamma * P_pi
        V = np.linalg.solve(A_mat, r_pi)
        Q = R + gamma * (P @ V)
        new_policy = np.argmax(Q, axis=1)
        V_history.append(V.copy())
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    V_star = V_history[-1]
    for V_h in V_history:
        errors.append(np.max(np.abs(V_h - V_star)))

    t_total = time.perf_counter() - t0_total
    return V, policy, errors, t_total


def run_pi(R, P):
    print("  Running PI...", end=" ", flush=True)
    V, pol, errs, t = policy_iteration(R, P, BETA)
    print(f"{len(errs)} iters, {t:.2f}s")
    return {'V': V, 'policy': pol, 'errors': errs, 'time': t, 'iters': len(errs)}


# ============================================================================
# Planning: LP (Manne 1960)
# ============================================================================

def run_lp(R, P):
    print("  Running LP...", end=" ", flush=True)
    t0 = time.perf_counter()
    n_s, n_a = R.shape
    gamma = BETA

    feasible = [(s, a) for s in range(n_s) for a in range(n_a) if R[s, a] > -1e30]
    n_con = len(feasible)

    A_ub = np.zeros((n_con, n_s))
    b_ub = np.zeros(n_con)
    for idx, (s, a) in enumerate(feasible):
        A_ub[idx, s] = -1.0
        A_ub[idx, :] += gamma * P[s, a, :]
        b_ub[idx] = -R[s, a]

    c = np.ones(n_s) / n_s
    v_bounds = [(-1e6, 1e6)] * n_s

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=v_bounds, method='highs')
    assert result.success, f"LP failed: {result.message}"
    V_lp = result.x

    policy = np.argmax(R + gamma * (P @ V_lp), axis=1)
    t = time.perf_counter() - t0
    print(f"{t:.2f}s")
    return {'V': V_lp, 'policy': policy, 'time': t}


# ============================================================================
# Q-Learning
# ============================================================================

def run_q_learning(env, V_star, cf_pol):
    cached = load_cache('Q-Learning')
    if cached is not None:
        print("  Q-Learning: loaded from cache")
        return cached

    print("  Running Q-Learning...", flush=True)
    np.random.seed(SEED)
    t0 = time.perf_counter()

    feas = feasible_mask(env.R)
    Q = np.full((N_S, N_A), -1e6)
    for s in range(N_S):
        Q[s, feas[s]] = 0.0

    counts = np.zeros((N_S, N_A))
    epsilon = QL_EPS_START

    eval_episodes_list = []
    eval_v_errors = []
    eval_policy_agr = []

    for ep in range(QL_EPISODES):
        s = env.reset()
        for t in range(QL_HORIZON):
            s_idx = env.state_to_index(s)
            if np.random.random() < epsilon:
                feasible_a = np.where(feas[s_idx])[0]
                a = np.random.choice(feasible_a)
            else:
                a = np.argmax(Q[s_idx])

            ns, r, done = env.step(a)
            ns_idx = env.state_to_index(ns)

            if r > -1e30:
                counts[s_idx, a] += 1
                alpha_sa = QL_ALPHA_C / (QL_ALPHA_C + counts[s_idx, a])
                max_q_next = np.max(Q[ns_idx])
                td_target = r + BETA * max_q_next
                Q[s_idx, a] += alpha_sa * (td_target - Q[s_idx, a])

            s = ns

        epsilon = max(QL_EPS_END, epsilon * QL_EPS_DECAY)

        if (ep + 1) % QL_EVAL_FREQ == 0:
            V_learned = np.max(Q, axis=1)
            pol_learned = np.argmax(Q, axis=1)
            eval_episodes_list.append(ep + 1)
            eval_v_errors.append(value_max_error(V_learned, V_star))
            eval_policy_agr.append(policy_agreement_soft(pol_learned, cf_pol))

        if (ep + 1) % 50_000 == 0:
            V_l = np.max(Q, axis=1)
            p_l = np.argmax(Q, axis=1)
            print(f"    ep {ep+1:>7d}: ||V-V*||_inf={value_max_error(V_l, V_star):.4f}, "
                  f"agr±1={policy_agreement_soft(p_l, cf_pol):.1f}%")

    t = time.perf_counter() - t0
    V_final = np.max(Q, axis=1)
    pol_final = np.argmax(Q, axis=1)

    data = {
        'V': V_final, 'policy': pol_final, 'time': t,
        'eval_episodes': eval_episodes_list,
        'eval_v_errors': eval_v_errors,
        'eval_policy_agr': eval_policy_agr,
    }
    save_cache('Q-Learning', data)
    return data


# ============================================================================
# SARSA
# ============================================================================

def run_sarsa(env, V_star, cf_pol):
    cached = load_cache('SARSA')
    if cached is not None:
        print("  SARSA: loaded from cache")
        return cached

    print("  Running SARSA...", flush=True)
    np.random.seed(SEED)
    t0 = time.perf_counter()

    feas = feasible_mask(env.R)
    Q = np.full((N_S, N_A), -1e6)
    for s in range(N_S):
        Q[s, feas[s]] = 0.0

    counts = np.zeros((N_S, N_A))
    epsilon = SARSA_EPS_START

    eval_episodes_list = []
    eval_v_errors = []
    eval_policy_agr = []

    def pick_action(s_idx, eps):
        if np.random.random() < eps:
            feasible_a = np.where(feas[s_idx])[0]
            return np.random.choice(feasible_a)
        return np.argmax(Q[s_idx])

    for ep in range(SARSA_EPISODES):
        s = env.reset()
        s_idx = env.state_to_index(s)
        a = pick_action(s_idx, epsilon)

        for t in range(SARSA_HORIZON):
            ns, r, done = env.step(a)
            ns_idx = env.state_to_index(ns)
            na = pick_action(ns_idx, epsilon)

            if r > -1e30:
                counts[s_idx, a] += 1
                alpha_sa = SARSA_ALPHA_C / (SARSA_ALPHA_C + counts[s_idx, a])
                td_target = r + BETA * Q[ns_idx, na]
                Q[s_idx, a] += alpha_sa * (td_target - Q[s_idx, a])

            s_idx = ns_idx
            a = na

        epsilon = max(SARSA_EPS_END, epsilon * SARSA_EPS_DECAY)

        if (ep + 1) % SARSA_EVAL_FREQ == 0:
            V_learned = np.max(Q, axis=1)
            pol_learned = np.argmax(Q, axis=1)
            eval_episodes_list.append(ep + 1)
            eval_v_errors.append(value_max_error(V_learned, V_star))
            eval_policy_agr.append(policy_agreement_soft(pol_learned, cf_pol))

        if (ep + 1) % 50_000 == 0:
            V_l = np.max(Q, axis=1)
            p_l = np.argmax(Q, axis=1)
            print(f"    ep {ep+1:>7d}: ||V-V*||_inf={value_max_error(V_l, V_star):.4f}, "
                  f"agr±1={policy_agreement_soft(p_l, cf_pol):.1f}%")

    t = time.perf_counter() - t0
    V_final = np.max(Q, axis=1)
    pol_final = np.argmax(Q, axis=1)

    data = {
        'V': V_final, 'policy': pol_final, 'time': t,
        'eval_episodes': eval_episodes_list,
        'eval_v_errors': eval_v_errors,
        'eval_policy_agr': eval_policy_agr,
    }
    save_cache('SARSA', data)
    return data


# ============================================================================
# DQN
# ============================================================================

def run_dqn(env, V_star, cf_pol):
    cached = load_cache('DQN')
    if cached is not None:
        print("  DQN: loaded from cache")
        return cached

    print("  Running DQN...", flush=True)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t0 = time.perf_counter()
    feas = feasible_mask(env.R)
    k_max = env.k_grid[-1]
    feas_torch = torch.tensor(feas)

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, DQN_HIDDEN),
                nn.ReLU(),
                nn.Linear(DQN_HIDDEN, DQN_HIDDEN),
                nn.ReLU(),
                nn.Linear(DQN_HIDDEN, N_A),
            )

        def forward(self, x):
            return self.net(x)

    q_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=DQN_LR)

    # Pre-compute state features for all states (for fast eval)
    all_state_features = torch.zeros(N_S, 2)
    for s_idx in range(N_S):
        ik, iz = env.index_to_state(s_idx)
        all_state_features[s_idx] = torch.FloatTensor([env.k_grid[ik] / k_max, float(iz)])

    # Numpy replay buffer (pre-allocated arrays)
    rb_s = np.zeros((DQN_REPLAY_SIZE, 2), dtype=np.float32)
    rb_a = np.zeros(DQN_REPLAY_SIZE, dtype=np.int64)
    rb_r = np.zeros(DQN_REPLAY_SIZE, dtype=np.float32)
    rb_ns = np.zeros((DQN_REPLAY_SIZE, 2), dtype=np.float32)
    rb_size = 0
    rb_idx = 0

    epsilon = DQN_EPS_START

    eval_episodes_list = []
    eval_v_errors = []
    eval_policy_agr = []

    total_steps = 0
    for ep in range(DQN_EPISODES):
        s = env.reset()
        ik, iz = s
        s_feat = np.array([env.k_grid[ik] / k_max, float(iz)], dtype=np.float32)

        for t in range(DQN_HORIZON):
            s_idx = ik * N_Z + iz
            if np.random.random() < epsilon:
                feasible_a = np.where(feas[s_idx])[0]
                a = int(np.random.choice(feasible_a))
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.from_numpy(s_feat).unsqueeze(0)).squeeze()
                    q_vals[~feas_torch[s_idx]] = -1e6
                    a = q_vals.argmax().item()

            ns, r, done = env.step(a)
            r_store = max(r, -1000.0)
            nik, niz = ns
            ns_feat = np.array([env.k_grid[nik] / k_max, float(niz)], dtype=np.float32)

            # Store in replay buffer
            rb_s[rb_idx] = s_feat
            rb_a[rb_idx] = a
            rb_r[rb_idx] = r_store
            rb_ns[rb_idx] = ns_feat
            rb_idx = (rb_idx + 1) % DQN_REPLAY_SIZE
            rb_size = min(rb_size + 1, DQN_REPLAY_SIZE)
            total_steps += 1

            s_feat = ns_feat
            ik, iz = nik, niz

            # Train every TRAIN_FREQ steps
            if total_steps % DQN_TRAIN_FREQ == 0 and rb_size >= DQN_BATCH_SIZE:
                idx = np.random.randint(0, rb_size, size=DQN_BATCH_SIZE)
                s_b = torch.from_numpy(rb_s[idx])
                a_b = torch.from_numpy(rb_a[idx])
                r_b = torch.from_numpy(rb_r[idx])
                ns_b = torch.from_numpy(rb_ns[idx])

                q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_b).max(dim=1)[0]
                    target = r_b + BETA * next_q

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(DQN_EPS_END, epsilon * DQN_EPS_DECAY)

        if (ep + 1) % DQN_TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % DQN_EVAL_FREQ == 0:
            q_net.eval()
            with torch.no_grad():
                all_q = q_net(all_state_features)
                all_q_masked = all_q.clone()
                all_q_masked[~feas_torch] = -1e6
                V_learned = all_q_masked.max(dim=1)[0].numpy()
                pol_learned = all_q_masked.argmax(dim=1).numpy()
            q_net.train()

            eval_episodes_list.append(ep + 1)
            eval_v_errors.append(value_max_error(V_learned, V_star))
            eval_policy_agr.append(policy_agreement_soft(pol_learned, cf_pol))

        if (ep + 1) % 10_000 == 0:
            q_net.eval()
            with torch.no_grad():
                all_q = q_net(all_state_features)
                all_q[~feas_torch] = -1e6
                V_l = all_q.max(dim=1)[0].numpy()
                p_l = all_q.argmax(dim=1).numpy()
            q_net.train()
            print(f"    ep {ep+1:>7d}: ||V-V*||_inf={value_max_error(V_l, V_star):.4f}, "
                  f"agr±1={policy_agreement_soft(p_l, cf_pol):.1f}%")

    # Final extraction
    t = time.perf_counter() - t0
    q_net.eval()
    with torch.no_grad():
        all_q = q_net(all_state_features)
        all_q[~feas_torch] = -1e6
        V_final = all_q.max(dim=1)[0].numpy()
        pol_final = all_q.argmax(dim=1).numpy()

    data = {
        'V': V_final, 'policy': pol_final, 'time': t,
        'eval_episodes': eval_episodes_list,
        'eval_v_errors': eval_v_errors,
        'eval_policy_agr': eval_policy_agr,
    }
    save_cache('DQN', data)
    return data


# ============================================================================
# REINFORCE (tabular softmax)
# ============================================================================

def softmax_probs_array(theta, s_idx, feas_mask, tau=1.0):
    """Softmax probabilities with infeasible masking."""
    logits = theta[s_idx].copy()
    logits[~feas_mask[s_idx]] = -1e6
    logits = logits / tau
    logits -= logits.max()
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum()


def extract_policy_from_theta(theta, feas, tau=1.0):
    """Extract greedy policy from theta for all states (vectorized)."""
    masked = theta.copy()
    masked[~feas] = -1e6
    return np.argmax(masked, axis=1)


def run_reinforce(env, V_star, cf_pol):
    cached = load_cache('REINFORCE')
    if cached is not None:
        print("  REINFORCE: loaded from cache")
        return cached

    print("  Running REINFORCE...", flush=True)
    np.random.seed(SEED)
    t0 = time.perf_counter()

    feas = feasible_mask(env.R)

    # Tabular softmax parameters
    theta = np.zeros((N_S, N_A))
    theta[~feas] = -1e6

    # Baseline: incremental mean return per state
    V_base = np.zeros(N_S)
    V_count = np.zeros(N_S)

    alpha_t = REINFORCE_ALPHA

    eval_episodes_list = []
    eval_v_errors = []
    eval_policy_agr = []

    for ep in range(REINFORCE_EPISODES):
        # Collect episode
        trajectory = []
        s = env.reset()
        for t in range(REINFORCE_HORIZON):
            s_idx = env.state_to_index(s)
            probs = softmax_probs_array(theta, s_idx, feas, REINFORCE_TAU)
            a = np.random.choice(N_A, p=probs)
            ns, r, done = env.step(a)
            trajectory.append((s_idx, a, r))
            s = ns

        # Compute returns and update (vectorized over actions)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            s_idx, a, r = trajectory[t]
            G = BETA * G + r

            V_count[s_idx] += 1
            V_base[s_idx] += (G - V_base[s_idx]) / V_count[s_idx]
            advantage = G - V_base[s_idx]

            probs = softmax_probs_array(theta, s_idx, feas, REINFORCE_TAU)
            grad = -probs.copy()
            grad[a] += 1.0  # grad[a] = 1 - probs[a], grad[a'] = -probs[a']
            mask = feas[s_idx]
            theta[s_idx, mask] += alpha_t * advantage * grad[mask] / REINFORCE_TAU

        alpha_t = alpha_t * REINFORCE_ALPHA_DECAY

        if (ep + 1) % REINFORCE_EVAL_FREQ == 0:
            pol_learned = extract_policy_from_theta(theta, feas, REINFORCE_TAU)
            eval_episodes_list.append(ep + 1)
            eval_v_errors.append(value_max_error(V_base, V_star))
            eval_policy_agr.append(policy_agreement_soft(pol_learned, cf_pol))

        if (ep + 1) % 50_000 == 0:
            pol_l = extract_policy_from_theta(theta, feas, REINFORCE_TAU)
            print(f"    ep {ep+1:>7d}: ||V-V*||_inf={value_max_error(V_base, V_star):.4f}, "
                  f"agr±1={policy_agreement_soft(pol_l, cf_pol):.1f}%")

    t = time.perf_counter() - t0
    pol_final = extract_policy_from_theta(theta, feas, REINFORCE_TAU)

    data = {
        'V': V_base.copy(), 'policy': pol_final, 'time': t,
        'eval_episodes': eval_episodes_list,
        'eval_v_errors': eval_v_errors,
        'eval_policy_agr': eval_policy_agr,
    }
    save_cache('REINFORCE', data)
    return data


# ============================================================================
# PPO (tabular softmax, clipped surrogate)
# ============================================================================

def run_ppo(env, V_star, cf_pol):
    cached = load_cache('PPO')
    if cached is not None:
        print("  PPO: loaded from cache")
        return cached

    print("  Running PPO...", flush=True)
    np.random.seed(SEED)
    t0 = time.perf_counter()

    feas = feasible_mask(env.R)

    theta = np.zeros((N_S, N_A))
    theta[~feas] = -1e6

    V_base = np.zeros(N_S)
    V_count = np.zeros(N_S)

    alpha_t = PPO_ALPHA

    eval_episodes_list = []
    eval_v_errors = []
    eval_policy_agr = []

    for ep in range(PPO_EPISODES):
        # Collect trajectory
        trajectory = []
        s = env.reset()
        for t in range(PPO_HORIZON):
            s_idx = env.state_to_index(s)
            probs = softmax_probs_array(theta, s_idx, feas, PPO_TAU)
            a = np.random.choice(N_A, p=probs)
            log_prob_old = np.log(probs[a] + 1e-10)
            ns, r, done = env.step(a)
            trajectory.append((s_idx, a, r, log_prob_old))
            s = ns

        T = len(trajectory)
        returns = np.zeros(T)
        advantages = np.zeros(T)

        # Returns
        G = 0.0
        for t in range(T - 1, -1, -1):
            _, _, r, _ = trajectory[t]
            G = BETA * G + r
            returns[t] = G

        # GAE
        gae = 0.0
        for t in range(T - 1, -1, -1):
            s_idx_t = trajectory[t][0]
            r_t = trajectory[t][2]
            if t < T - 1:
                v_next = V_base[trajectory[t + 1][0]]
            else:
                v_next = 0.0
            delta = r_t + BETA * v_next - V_base[s_idx_t]
            gae = delta + BETA * PPO_GAE_LAMBDA * gae
            advantages[t] = gae

        # Update baseline
        for t in range(T):
            s_idx_t = trajectory[t][0]
            V_count[s_idx_t] += 1
            V_base[s_idx_t] += (returns[t] - V_base[s_idx_t]) / V_count[s_idx_t]

        # PPO epochs (vectorized over actions)
        for _ in range(PPO_EPOCHS):
            for t in range(T):
                s_idx_t, a_t, _, log_prob_old_t = trajectory[t]
                adv_t = advantages[t]

                probs_new = softmax_probs_array(theta, s_idx_t, feas, PPO_TAU)
                log_prob_new = np.log(probs_new[a_t] + 1e-10)
                ratio = np.exp(log_prob_new - log_prob_old_t)

                surr1 = ratio * adv_t
                surr2 = np.clip(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_t
                if surr1 <= surr2:
                    effective_adv = adv_t
                else:
                    if ratio > 1.0 + PPO_CLIP or ratio < 1.0 - PPO_CLIP:
                        effective_adv = 0.0
                    else:
                        effective_adv = adv_t

                grad = -probs_new.copy()
                grad[a_t] += 1.0
                mask = feas[s_idx_t]
                theta[s_idx_t, mask] += alpha_t * effective_adv * ratio * grad[mask] / PPO_TAU

        alpha_t = alpha_t * PPO_ALPHA_DECAY

        if (ep + 1) % PPO_EVAL_FREQ == 0:
            pol_learned = extract_policy_from_theta(theta, feas, PPO_TAU)
            eval_episodes_list.append(ep + 1)
            eval_v_errors.append(value_max_error(V_base, V_star))
            eval_policy_agr.append(policy_agreement_soft(pol_learned, cf_pol))

        if (ep + 1) % 50_000 == 0:
            pol_l = extract_policy_from_theta(theta, feas, PPO_TAU)
            print(f"    ep {ep+1:>7d}: ||V-V*||_inf={value_max_error(V_base, V_star):.4f}, "
                  f"agr±1={policy_agreement_soft(pol_l, cf_pol):.1f}%")

    t = time.perf_counter() - t0
    pol_final = extract_policy_from_theta(theta, feas, PPO_TAU)

    data = {
        'V': V_base.copy(), 'policy': pol_final, 'time': t,
        'eval_episodes': eval_episodes_list,
        'eval_v_errors': eval_v_errors,
        'eval_policy_agr': eval_policy_agr,
    }
    save_cache('PPO', data)
    return data


# ============================================================================
# Figures
# ============================================================================

def fig_learning_curves(results):
    """||V - V*||_inf over episodes for learning algorithms."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    planning_algos = ['VI', 'PI', 'LP']
    learning_algos = ['Q-Learning', 'SARSA', 'DQN', 'REINFORCE', 'PPO']

    # Planning baselines (horizontal dashed lines at their max error = ~0)
    for name in planning_algos:
        if name in results:
            ax.axhline(0.01, color=ALGO_COLORS[name], linestyle='--',
                       alpha=0.5, label=f'{name} (exact)')

    for name in learning_algos:
        if name in results and 'eval_episodes' in results[name]:
            eps = results[name]['eval_episodes']
            errs = results[name]['eval_v_errors']
            ax.plot(eps, errs, label=name, color=ALGO_COLORS[name])

    ax.set_xlabel('Episode')
    ax.set_ylabel('$\\|V - V^*\\|_\\infty$')
    ax.set_title('Brock-Mirman: Value Function Convergence')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'bm_learning_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_policy_curves(results, k_grid):
    """k'(k) savings policy for each z state."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Closed-form
    cf = closed_form_policy(k_grid)

    for iz, (ax, z_label) in enumerate(zip(axes, ['$z = 0.9$', '$z = 1.1$'])):
        # Closed-form curve
        cf_kprime = np.array([k_grid[cf[ik * N_Z + iz]] for ik in range(N_K)])
        ax.plot(k_grid, cf_kprime, 'k--', linewidth=2.5, label='Closed-form', zorder=10)

        for name in ALGO_NAMES:
            if name not in results:
                continue
            pol = results[name]['policy']
            kprime = np.array([k_grid[pol[ik * N_Z + iz]] for ik in range(N_K)])
            ax.plot(k_grid, kprime, color=ALGO_COLORS[name], label=name, alpha=0.8)

        ax.set_xlabel('Capital $k$')
        ax.set_ylabel("$k'(k,z)$")
        ax.set_title(f'Savings Policy ({z_label})')
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'bm_policy_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_value_functions(results, k_grid, V_star):
    """V(k,z) for each z state."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for iz, (ax, z_label) in enumerate(zip(axes, ['$z = 0.9$', '$z = 1.1$'])):
        # V* curve
        v_star_slice = np.array([V_star[ik * N_Z + iz] for ik in range(N_K)])
        ax.plot(k_grid, v_star_slice, 'k--', linewidth=2.5, label='$V^*$ (VI)', zorder=10)

        for name in ALGO_NAMES:
            if name not in results:
                continue
            if name == 'VI':
                continue  # already plotted as V*
            V = results[name]['V']
            v_slice = np.array([V[ik * N_Z + iz] for ik in range(N_K)])
            ax.plot(k_grid, v_slice, color=ALGO_COLORS[name], label=name, alpha=0.8)

        ax.set_xlabel('Capital $k$')
        ax.set_ylabel('$V(k,z)$')
        ax.set_title(f'Value Function ({z_label})')
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'bm_value_functions.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_convergence(results):
    """Policy agreement (% matching closed-form) over episodes."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    learning_algos = ['Q-Learning', 'SARSA', 'DQN', 'REINFORCE', 'PPO']
    for name in learning_algos:
        if name in results and 'eval_episodes' in results[name]:
            eps = results[name]['eval_episodes']
            agr = results[name]['eval_policy_agr']
            ax.plot(eps, agr, label=name, color=ALGO_COLORS[name])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Agreement $\\pm 1$ with Closed-Form (\\%)')
    ax.set_title('Brock-Mirman: Policy Convergence')
    ax.set_ylim(-5, 105)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'bm_convergence.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Table
# ============================================================================

def generate_table(results, V_star, cf_pol):
    lines = []
    lines.append(r'\begin{tabular}{lrrrr}')
    lines.append(r'\hline')
    lines.append(r'Algorithm & $\|V{-}V^*\|_\mathrm{RMSE}$ & $\|V{-}V^*\|_\infty$ '
                 r'& Agreement $\pm 1$ (\%) & Time (s) \\')
    lines.append(r'\hline')

    for name in ALGO_NAMES:
        if name not in results:
            continue
        r = results[name]
        V = r['V']
        pol = r['policy']
        rmse = value_rmse(V, V_star)
        maxe = value_max_error(V, V_star)
        agr = policy_agreement_soft(pol, cf_pol)
        t = r['time']
        lines.append(f'{name} & {rmse:.4f} & {maxe:.4f} & {agr:.1f} & {t:.1f} \\\\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    path = os.path.join(OUTPUT_DIR, 'bm_study_results.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Brock-Mirman 8-algorithm study')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated algo cache keys (e.g. --only ql,sarsa)')
    add_cache_args(parser)
    args = parser.parse_args()

    only_set = None
    if args.only:
        only_set = {x.strip() for x in args.only.split(',')}

    print("=" * 70)
    print("BROCK-MIRMAN ILLUSTRATED EXAMPLE: 8 ALGORITHMS")
    print("=" * 70)
    print(f"Parameters: alpha={ALPHA_PROD}, beta={BETA}, z={Z_VALS.tolist()}")
    print(f"Grid: n_k={N_K}, n_z={N_Z} -> {N_S} states, {N_A} actions")
    if only_set:
        print(f"Mode: selective re-run ({', '.join(only_set)})")
    elif args.plots_only:
        print("Mode: plots-only")
    elif args.data_only:
        print("Mode: data-only (compute, skip figures/tables)")
    else:
        print("Mode: full run (cache where valid)")
    print()

    # Build model
    k_grid = build_grid(N_K)
    R, P = build_reward_and_transitions(k_grid)
    cf_pol = closed_form_policy(k_grid)
    env = BrockMirmanEnv(k_grid, R, P)

    print("Phase 1: Planning")
    print("-" * 40)
    results = {}

    vi_data = run_vi(R, P)
    results['VI'] = vi_data
    V_star = vi_data['V']

    pi_data = run_pi(R, P)
    results['PI'] = pi_data

    lp_data = run_lp(R, P)
    results['LP'] = lp_data

    # Verify planning methods
    for name in ['VI', 'PI', 'LP']:
        agr = policy_agreement_soft(results[name]['policy'], cf_pol)
        print(f"  {name} policy agreement (±1) with closed-form: {agr:.1f}%")

    if args.plots_only:
        print("\nPhase 2: Loading learning algorithms from cache")
        print("-" * 40)
        for name in ['Q-Learning', 'SARSA', 'DQN', 'REINFORCE', 'PPO']:
            cached = load_cache(name)
            if cached is not None:
                results[name] = cached
                print(f"  {name}: loaded from cache")
            else:
                print(f"  {name}: no cache found, skipping")
    else:
        print("\nPhase 2: Learning algorithms")
        print("-" * 40)

        def should_run(name):
            if only_set is None:
                return True
            return ALGO_CACHE_KEYS[name] in only_set

        # For algorithms we're not re-running, try loading cache
        for name in ['Q-Learning', 'SARSA', 'DQN', 'REINFORCE', 'PPO']:
            if should_run(name):
                if name == 'Q-Learning':
                    results[name] = run_q_learning(env, V_star, cf_pol)
                elif name == 'SARSA':
                    results[name] = run_sarsa(env, V_star, cf_pol)
                elif name == 'DQN':
                    results[name] = run_dqn(env, V_star, cf_pol)
                elif name == 'REINFORCE':
                    results[name] = run_reinforce(env, V_star, cf_pol)
                elif name == 'PPO':
                    results[name] = run_ppo(env, V_star, cf_pol)
            else:
                cached = load_cache(name)
                if cached is not None:
                    results[name] = cached
                    print(f"  {name}: loaded from cache")

    if not args.data_only:
        # Phase 3: Figures
        print("\nPhase 3: Figures")
        print("-" * 40)
        fig_learning_curves(results)
        fig_policy_curves(results, k_grid)
        fig_value_functions(results, k_grid, V_star)
        fig_convergence(results)

        # Phase 4: Table
        print("\nPhase 4: Table")
        print("-" * 40)
        generate_table(results, V_star, cf_pol)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<12} {'RMSE':>8} {'Max Err':>8} {'Agr±1 %':>8} {'Time':>8}")
    print("-" * 52)
    for name in ALGO_NAMES:
        if name not in results:
            continue
        r = results[name]
        rmse = value_rmse(r['V'], V_star)
        maxe = value_max_error(r['V'], V_star)
        agr = policy_agreement_soft(r['policy'], cf_pol)
        print(f"{name:<12} {rmse:>8.4f} {maxe:>8.4f} {agr:>7.1f}% {r['time']:>7.1f}s")

    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png') or f.endswith('.tex'):
            print(f"  {f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
