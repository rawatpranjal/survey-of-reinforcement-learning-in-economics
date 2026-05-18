#!/usr/bin/env python3
"""
dtr_qlearning_vs_murphy.py
Chapter 15 (RL for Causal Inference), Section subsec:gmethods_bridge.

Simulation study on the Murphy/Q-learning equivalence. Three research
questions, three panels:

(Q1) Tabular sample-size sweep. On a synthetic two-stage DTR with finite
     state under sequential ignorability, do Murphy's batch backward
     regression (= Fitted Q-Iteration) and online Q-learning recover the
     same optimal regime as the cohort size grows?

(Q2) Tabular training-budget sweep. For Q-learning, which solves the
     recursion by stochastic approximation rather than one-shot regression,
     how does the recovered regime improve as the cohort is replayed more
     times at a fixed cohort size?

(Q3) High-dimensional case. When the state is continuous and high enough
     dimensional that tabular Q is infeasible, do the neural-network
     analogues -- Neural Fitted Q-Iteration and DQN -- recover the same
     optimal regime?

All three plots share the V(pi_hat)/V* axis. The oracle V* is computed
analytically in the tabular case and by Monte Carlo on the known DGP in
the high-dimensional case.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import (
    apply_style, COLORS, BENCH_STYLE,
)
from sims.sim_cache import (
    compute_or_load, add_component_args, parse_force_set,
)

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'dtr_qlearning_vs_murphy'


# ============================================================================
# TABULAR CASE
# ============================================================================
N_STATES = 5
S_TREAT = 2

ALPHA_0 = 1.0
ALPHA_1 = -0.4
BETA_S = 1.0
BETA_A = -0.3
BETA_SA = 1.5
SIGMA_Y = 0.5
P_IMPROVE = 0.6
P_WORSEN = 0.3
P_DRIFT = 0.15

N_GRID = [100, 300, 1000, 3000, 10000]
N_EPOCHS_GRID = [1, 3, 10, 30, 100]
N_AT_EPOCHS_PANEL = 300
N_EPOCHS_DEFAULT = 100
ALPHA_QLEARN = 0.1
N_SEEDS = 50

SHARED_CONFIG = {
    'N_STATES': N_STATES, 'S_TREAT': S_TREAT,
    'ALPHA_0': ALPHA_0, 'ALPHA_1': ALPHA_1,
    'BETA_S': BETA_S, 'BETA_A': BETA_A, 'BETA_SA': BETA_SA,
    'SIGMA_Y': SIGMA_Y,
    'P_IMPROVE': P_IMPROVE, 'P_WORSEN': P_WORSEN, 'P_DRIFT': P_DRIFT,
    'N_SEEDS': N_SEEDS,
}
ORACLE_CONFIG = {**SHARED_CONFIG}
MURPHY_CONFIG = {**SHARED_CONFIG, 'N_GRID': N_GRID}
QLEARN_N_CONFIG = {
    **SHARED_CONFIG, 'N_GRID': N_GRID,
    'N_EPOCHS': N_EPOCHS_DEFAULT, 'ALPHA_QLEARN': ALPHA_QLEARN,
}
QLEARN_EPOCHS_CONFIG = {
    **SHARED_CONFIG, 'N_FIXED': N_AT_EPOCHS_PANEL,
    'N_EPOCHS_GRID': N_EPOCHS_GRID, 'ALPHA_QLEARN': ALPHA_QLEARN,
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def behavior_prop_a1(s):
    return sigmoid(ALPHA_0 + ALPHA_1 * s)


def transition_dist(s1, a1):
    if s1 <= S_TREAT:
        if a1 == 1:
            s2_up = min(N_STATES, s1 + 1)
            if s2_up == s1:
                return {s1: 1.0}
            return {s1: 1.0 - P_IMPROVE, s2_up: P_IMPROVE}
        else:
            s2_dn = max(1, s1 - 1)
            if s2_dn == s1:
                return {s1: 1.0}
            return {s1: 1.0 - P_WORSEN, s2_dn: P_WORSEN}
    else:
        s2_dn = max(1, s1 - 1)
        s2_up = min(N_STATES, s1 + 1)
        dist = {}
        p_dn = P_DRIFT if s2_dn != s1 else 0.0
        p_up = P_DRIFT if s2_up != s1 else 0.0
        dist[s1] = 1.0 - p_dn - p_up
        if p_dn > 0:
            dist[s2_dn] = dist.get(s2_dn, 0.0) + p_dn
        if p_up > 0:
            dist[s2_up] = dist.get(s2_up, 0.0) + p_up
        return {k: v for k, v in dist.items() if v > 0}


def outcome_mean_tab(s2, a2):
    return BETA_S * s2 + BETA_A * a2 + BETA_SA * (1 if s2 <= S_TREAT else 0) * a2


def generate_cohort_tab(N, rng):
    S1 = rng.integers(1, N_STATES + 1, size=N)
    p1 = sigmoid(ALPHA_0 + ALPHA_1 * S1)
    A1 = (rng.random(N) < p1).astype(np.int32)
    S2 = np.empty(N, dtype=np.int32)
    mask_low_t = (S1 <= S_TREAT) & (A1 == 1)
    mask_low_n = (S1 <= S_TREAT) & (A1 == 0)
    mask_high = (S1 > S_TREAT)
    n_lt = int(mask_low_t.sum())
    improve = rng.random(n_lt) < P_IMPROVE
    S2[mask_low_t] = np.where(improve, np.minimum(N_STATES, S1[mask_low_t] + 1), S1[mask_low_t])
    n_ln = int(mask_low_n.sum())
    worsen = rng.random(n_ln) < P_WORSEN
    S2[mask_low_n] = np.where(worsen, np.maximum(1, S1[mask_low_n] - 1), S1[mask_low_n])
    drift = rng.random(int(mask_high.sum()))
    S2_high = S1[mask_high].copy()
    S2_high[drift < P_DRIFT] = np.maximum(1, S1[mask_high][drift < P_DRIFT] - 1)
    up_m = (drift >= P_DRIFT) & (drift < 2 * P_DRIFT)
    S2_high[up_m] = np.minimum(N_STATES, S1[mask_high][up_m] + 1)
    S2[mask_high] = S2_high
    p2 = sigmoid(ALPHA_0 + ALPHA_1 * S2)
    A2 = (rng.random(N) < p2).astype(np.int32)
    indicator = (S2 <= S_TREAT).astype(float)
    Y_mean = BETA_S * S2 + BETA_A * A2 + BETA_SA * indicator * A2
    Y = Y_mean + SIGMA_Y * rng.normal(size=N)
    return S1, A1, S2, A2, Y


def compute_oracle_tab(cfg):
    V2 = np.zeros(N_STATES + 1)
    pi2_star = {}
    for s2 in range(1, N_STATES + 1):
        q = {a: outcome_mean_tab(s2, a) for a in (0, 1)}
        a_star = 0 if q[0] >= q[1] else 1
        V2[s2] = q[a_star]
        pi2_star[s2] = a_star
    V1 = np.zeros(N_STATES + 1)
    pi1_star = {}
    for s1 in range(1, N_STATES + 1):
        q1 = {}
        for a1 in (0, 1):
            dist = transition_dist(s1, a1)
            q1[a1] = sum(p * V2[s2] for s2, p in dist.items())
        a_star = 0 if q1[0] >= q1[1] else 1
        V1[s1] = q1[a_star]
        pi1_star[s1] = a_star
    V_star = float(np.mean(V1[1:N_STATES + 1]))
    print(f"    Tabular Oracle V* = {V_star:.4f}")
    print("    Optimal stage-1 policy: " + ", ".join(f's={s}: a={pi1_star[s]}' for s in range(1, N_STATES + 1)))
    print("    Optimal stage-2 policy: " + ", ".join(f's2={s}: a={pi2_star[s]}' for s in range(1, N_STATES + 1)))
    return {
        'V_star': V_star, 'V1_star': V1.tolist(), 'V2_star': V2.tolist(),
        'pi1_star': pi1_star, 'pi2_star': pi2_star,
    }


def evaluate_policy_tab(pi_hat_1, pi_hat_2):
    V = 0.0
    for s1 in range(1, N_STATES + 1):
        a1 = pi_hat_1[s1]
        dist = transition_dist(s1, a1)
        ev_s1 = 0.0
        for s2, p_s2 in dist.items():
            a2 = pi_hat_2[(s1, a1, s2)]
            ev_s1 += p_s2 * outcome_mean_tab(s2, a2)
        V += ev_s1 / N_STATES
    return V


def murphy_estimate_tab(S1, A1, S2, A2, Y):
    sum2 = np.zeros((N_STATES + 1, 2, N_STATES + 1, 2))
    cnt2 = np.zeros((N_STATES + 1, 2, N_STATES + 1, 2), dtype=np.int64)
    np.add.at(sum2, (S1, A1, S2, A2), Y)
    np.add.at(cnt2, (S1, A1, S2, A2), 1)
    Q2 = np.zeros_like(sum2)
    nz = cnt2 > 0
    Q2[nz] = sum2[nz] / cnt2[nz]
    V2_hat = Q2.max(axis=-1)
    pi2_arr = Q2.argmax(axis=-1)
    V2_obs = V2_hat[S1, A1, S2]
    sum1 = np.zeros((N_STATES + 1, 2))
    cnt1 = np.zeros((N_STATES + 1, 2), dtype=np.int64)
    np.add.at(sum1, (S1, A1), V2_obs)
    np.add.at(cnt1, (S1, A1), 1)
    Q1 = np.zeros_like(sum1)
    nz1 = cnt1 > 0
    Q1[nz1] = sum1[nz1] / cnt1[nz1]
    pi1_arr = Q1.argmax(axis=-1)
    pi_hat_1 = {s1: int(pi1_arr[s1]) for s1 in range(1, N_STATES + 1)}
    pi_hat_2 = {
        (s1, a1, s2): int(pi2_arr[s1, a1, s2])
        for s1 in range(1, N_STATES + 1)
        for a1 in (0, 1)
        for s2 in range(1, N_STATES + 1)
    }
    return pi_hat_1, pi_hat_2


def qlearn_estimate_tab(S1, A1, S2, A2, Y, n_epochs, alpha, rng):
    Q1 = np.zeros((N_STATES + 1, 2))
    Q2 = np.zeros((N_STATES + 1, 2, N_STATES + 1, 2))
    n = len(Y)
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for idx in order:
            s1 = int(S1[idx]); a1 = int(A1[idx])
            s2 = int(S2[idx]); a2 = int(A2[idx])
            y = float(Y[idx])
            target1 = max(Q2[s1, a1, s2, 0], Q2[s1, a1, s2, 1])
            Q1[s1, a1] += alpha * (target1 - Q1[s1, a1])
            Q2[s1, a1, s2, a2] += alpha * (y - Q2[s1, a1, s2, a2])
    pi1_arr = Q1.argmax(axis=-1)
    pi2_arr = Q2.argmax(axis=-1)
    pi_hat_1 = {s1: int(pi1_arr[s1]) for s1 in range(1, N_STATES + 1)}
    pi_hat_2 = {
        (s1, a1, s2): int(pi2_arr[s1, a1, s2])
        for s1 in range(1, N_STATES + 1)
        for a1 in (0, 1)
        for s2 in range(1, N_STATES + 1)
    }
    return pi_hat_1, pi_hat_2


def run_murphy_sweep(cfg):
    V = np.zeros((len(cfg['N_GRID']), cfg['N_SEEDS']))
    for i, N in enumerate(cfg['N_GRID']):
        for s in tqdm(range(cfg['N_SEEDS']), desc=f'  Murphy N={N}', leave=False,
                      disable=not sys.stderr.isatty()):
            rng = np.random.default_rng(N * 1000 + s)
            S1, A1, S2, A2, Y = generate_cohort_tab(N, rng)
            pi1, pi2 = murphy_estimate_tab(S1, A1, S2, A2, Y)
            V[i, s] = evaluate_policy_tab(pi1, pi2)
    return {'V': V, 'N_grid': list(cfg['N_GRID'])}


def run_qlearn_N_sweep(cfg):
    V = np.zeros((len(cfg['N_GRID']), cfg['N_SEEDS']))
    for i, N in enumerate(cfg['N_GRID']):
        for s in tqdm(range(cfg['N_SEEDS']), desc=f'  Q-learn N={N}', leave=False,
                      disable=not sys.stderr.isatty()):
            rng = np.random.default_rng(N * 1000 + s + 7)
            S1, A1, S2, A2, Y = generate_cohort_tab(N, rng)
            pi1, pi2 = qlearn_estimate_tab(S1, A1, S2, A2, Y, cfg['N_EPOCHS'], cfg['ALPHA_QLEARN'], rng)
            V[i, s] = evaluate_policy_tab(pi1, pi2)
    return {'V': V, 'N_grid': list(cfg['N_GRID']), 'n_epochs': cfg['N_EPOCHS']}


def run_qlearn_epochs_sweep(cfg):
    V = np.zeros((len(cfg['N_EPOCHS_GRID']), cfg['N_SEEDS']))
    for i, ne in enumerate(cfg['N_EPOCHS_GRID']):
        for s in tqdm(range(cfg['N_SEEDS']), desc=f'  Q-learn epochs={ne}', leave=False,
                      disable=not sys.stderr.isatty()):
            rng = np.random.default_rng(cfg['N_FIXED'] * 1000 + s + 13)
            S1, A1, S2, A2, Y = generate_cohort_tab(cfg['N_FIXED'], rng)
            pi1, pi2 = qlearn_estimate_tab(S1, A1, S2, A2, Y, ne, cfg['ALPHA_QLEARN'], rng)
            V[i, s] = evaluate_policy_tab(pi1, pi2)
    return {'V': V, 'epochs_grid': list(cfg['N_EPOCHS_GRID']), 'N_fixed': cfg['N_FIXED']}


# ============================================================================
# HIGH-DIMENSIONAL CASE
# ============================================================================
P_FEAT = 10
T_DECAY = 0.5
DELTA_HD = 0.5
SIGMA_ETA_HD = 0.5
SIGMA_Y_HD = 0.5
BETA_OUTCOME = np.array([1.0, 0.5, -0.3, 0.2, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
GAMMA_BEH = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ALPHA_A_HD = -0.3
ALPHA_SA_HD = 1.5
THRESHOLD_HD = 0.0

N_GRID_HD = [500, 2000, 5000]
N_SEEDS_HD = 20
N_FQI_EPOCHS = 200
N_DQN_STEPS = 8000
BATCH_SIZE_HD = 64
LR_NN = 1e-3
HIDDEN_DIM = 64

SHARED_CONFIG_HD = {
    'P_FEAT': P_FEAT, 'T_DECAY': T_DECAY, 'DELTA_HD': DELTA_HD,
    'SIGMA_ETA_HD': SIGMA_ETA_HD, 'SIGMA_Y_HD': SIGMA_Y_HD,
    'BETA_OUTCOME': BETA_OUTCOME.tolist(),
    'GAMMA_BEH': GAMMA_BEH.tolist(),
    'ALPHA_A_HD': ALPHA_A_HD, 'ALPHA_SA_HD': ALPHA_SA_HD,
    'THRESHOLD_HD': THRESHOLD_HD,
    'N_SEEDS_HD': N_SEEDS_HD,
    'N_GRID_HD': N_GRID_HD,
    'HIDDEN_DIM': HIDDEN_DIM, 'LR_NN': LR_NN,
}
ORACLE_HD_CONFIG = {**SHARED_CONFIG_HD}
FQI_HD_CONFIG = {**SHARED_CONFIG_HD, 'N_FQI_EPOCHS': N_FQI_EPOCHS}
DQN_HD_CONFIG = {**SHARED_CONFIG_HD, 'N_DQN_STEPS': N_DQN_STEPS, 'BATCH_SIZE_HD': BATCH_SIZE_HD}


def generate_cohort_hd(N, rng):
    """High-dimensional two-stage cohort."""
    S1 = rng.normal(size=(N, P_FEAT)).astype(np.float32)
    p1 = sigmoid(S1 @ GAMMA_BEH)
    A1 = (rng.random(N) < p1).astype(np.int32)
    e0 = np.zeros(P_FEAT, dtype=np.float32); e0[0] = 1.0
    eta = SIGMA_ETA_HD * rng.normal(size=(N, P_FEAT)).astype(np.float32)
    S2 = T_DECAY * S1 + DELTA_HD * A1.reshape(-1, 1).astype(np.float32) * e0 + eta
    p2 = sigmoid(S2 @ GAMMA_BEH)
    A2 = (rng.random(N) < p2).astype(np.int32)
    indicator = (S2[:, 0] < THRESHOLD_HD).astype(np.float32)
    Y_mean = S2 @ BETA_OUTCOME + ALPHA_A_HD * A2 + ALPHA_SA_HD * indicator * A2
    Y = (Y_mean + SIGMA_Y_HD * rng.normal(size=N)).astype(np.float32)
    return S1, A1, S2, A2, Y


def compute_oracle_hd(cfg):
    """V* by MC under the known DGP applying the analytical optimal policy."""
    M = 200000
    rng = np.random.default_rng(0)
    S1 = rng.normal(size=(M, P_FEAT))
    # Optimal stage-1: maximize Q*_1(S_1, A_1).
    # Q*_1(S_1, A_1) = 0.5 * beta' S_1 + delta * beta[0] * A_1
    #                  + (alpha_A + alpha_SA) * P(S_2[0] < c | S_1, A_1)
    # with S_2[0] | S_1, A_1 ~ N(0.5 * S_1[0] + delta * A_1, sigma_eta^2).
    from scipy.stats import norm
    common = 0.5 * S1 @ BETA_OUTCOME
    Q1_a0 = common + (ALPHA_A_HD + ALPHA_SA_HD) * norm.cdf(
        (THRESHOLD_HD - 0.5 * S1[:, 0]) / SIGMA_ETA_HD)
    Q1_a1 = (common + DELTA_HD * BETA_OUTCOME[0]
             + (ALPHA_A_HD + ALPHA_SA_HD) * norm.cdf(
                 (THRESHOLD_HD - 0.5 * S1[:, 0] - DELTA_HD) / SIGMA_ETA_HD))
    A1_star = (Q1_a1 > Q1_a0).astype(np.int32)
    # Sample S_2 under A_1_star, then apply analytical optimal stage-2 (treat iff S_2[0] < threshold).
    e0 = np.zeros(P_FEAT); e0[0] = 1.0
    eta = SIGMA_ETA_HD * rng.normal(size=(M, P_FEAT))
    S2 = T_DECAY * S1 + DELTA_HD * A1_star.reshape(-1, 1) * e0 + eta
    A2_star = (S2[:, 0] < THRESHOLD_HD).astype(np.int32)
    indicator = (S2[:, 0] < THRESHOLD_HD).astype(float)
    Y_mean = S2 @ BETA_OUTCOME + ALPHA_A_HD * A2_star + ALPHA_SA_HD * indicator * A2_star
    V_star = float(Y_mean.mean())
    V_star_se = float(Y_mean.std() / np.sqrt(M))
    # Also compute V(behavior) for reference
    A1_b = (rng.random(M) < sigmoid(S1 @ GAMMA_BEH)).astype(np.int32)
    eta_b = SIGMA_ETA_HD * rng.normal(size=(M, P_FEAT))
    S2_b = T_DECAY * S1 + DELTA_HD * A1_b.reshape(-1, 1) * e0 + eta_b
    A2_b = (rng.random(M) < sigmoid(S2_b @ GAMMA_BEH)).astype(np.int32)
    indicator_b = (S2_b[:, 0] < THRESHOLD_HD).astype(float)
    Y_b = S2_b @ BETA_OUTCOME + ALPHA_A_HD * A2_b + ALPHA_SA_HD * indicator_b * A2_b
    V_behavior = float(Y_b.mean())
    print(f"    High-dim Oracle V* = {V_star:.4f} (MC SE {V_star_se:.4f})")
    print(f"    High-dim V(behavior policy) = {V_behavior:.4f}")
    return {'V_star': V_star, 'V_star_se': V_star_se, 'V_behavior': V_behavior}


class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_policy_hd(Q1, Q2, M=50000, seed=42):
    """V(pi_hat) under the true DGP by MC, given torch Q-networks."""
    rng = np.random.default_rng(seed)
    S1 = rng.normal(size=(M, P_FEAT)).astype(np.float32)
    s1_t = torch.from_numpy(S1)
    with torch.no_grad():
        q1_a0 = Q1(torch.cat([s1_t, torch.zeros(M, 1)], dim=1)).numpy()
        q1_a1 = Q1(torch.cat([s1_t, torch.ones(M, 1)], dim=1)).numpy()
    A1 = (q1_a1 > q1_a0).astype(np.int32)
    e0 = np.zeros(P_FEAT, dtype=np.float32); e0[0] = 1.0
    eta = SIGMA_ETA_HD * rng.normal(size=(M, P_FEAT)).astype(np.float32)
    S2 = T_DECAY * S1 + DELTA_HD * A1.reshape(-1, 1).astype(np.float32) * e0 + eta
    s2_t = torch.from_numpy(S2)
    with torch.no_grad():
        q2_a0 = Q2(torch.cat([s2_t, torch.zeros(M, 1)], dim=1)).numpy()
        q2_a1 = Q2(torch.cat([s2_t, torch.ones(M, 1)], dim=1)).numpy()
    A2 = (q2_a1 > q2_a0).astype(np.int32)
    indicator = (S2[:, 0] < THRESHOLD_HD).astype(np.float32)
    Y_mean = S2 @ BETA_OUTCOME + ALPHA_A_HD * A2 + ALPHA_SA_HD * indicator * A2
    return float(Y_mean.mean())


def neural_fqi_estimate(S1, A1, S2, A2, Y, n_epochs, lr, hidden_dim, seed):
    torch.manual_seed(seed)
    Q2 = QNet(P_FEAT + 1, hidden_dim)
    opt2 = optim.Adam(Q2.parameters(), lr=lr)
    s2_t = torch.from_numpy(S2)
    a2_t = torch.from_numpy(A2.reshape(-1, 1).astype(np.float32))
    y_t = torch.from_numpy(Y)
    X2 = torch.cat([s2_t, a2_t], dim=1)
    for _ in range(n_epochs):
        opt2.zero_grad()
        loss = ((Q2(X2) - y_t) ** 2).mean()
        loss.backward()
        opt2.step()
    N = len(S2)
    with torch.no_grad():
        q2_a0 = Q2(torch.cat([s2_t, torch.zeros(N, 1)], dim=1))
        q2_a1 = Q2(torch.cat([s2_t, torch.ones(N, 1)], dim=1))
        V2 = torch.maximum(q2_a0, q2_a1)
    Q1 = QNet(P_FEAT + 1, hidden_dim)
    opt1 = optim.Adam(Q1.parameters(), lr=lr)
    s1_t = torch.from_numpy(S1)
    a1_t = torch.from_numpy(A1.reshape(-1, 1).astype(np.float32))
    X1 = torch.cat([s1_t, a1_t], dim=1)
    for _ in range(n_epochs):
        opt1.zero_grad()
        loss = ((Q1(X1) - V2) ** 2).mean()
        loss.backward()
        opt1.step()
    return Q1, Q2


def dqn_estimate(S1, A1, S2, A2, Y, n_steps, batch_size, lr, hidden_dim, seed, rng):
    torch.manual_seed(seed)
    Q1 = QNet(P_FEAT + 1, hidden_dim)
    Q2 = QNet(P_FEAT + 1, hidden_dim)
    opt = optim.Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=lr)
    N = len(S1)
    s1_all = torch.from_numpy(S1); s2_all = torch.from_numpy(S2)
    a1_all = torch.from_numpy(A1.reshape(-1, 1).astype(np.float32))
    a2_all = torch.from_numpy(A2.reshape(-1, 1).astype(np.float32))
    y_all = torch.from_numpy(Y)
    for _ in range(n_steps):
        idx = rng.integers(0, N, size=batch_size)
        idx_t = torch.from_numpy(idx)
        s1_b = s1_all[idx_t]
        a1_b = a1_all[idx_t]
        s2_b = s2_all[idx_t]
        a2_b = a2_all[idx_t]
        y_b = y_all[idx_t]
        q2_pred = Q2(torch.cat([s2_b, a2_b], dim=1))
        loss2 = ((q2_pred - y_b) ** 2).mean()
        with torch.no_grad():
            q2_a0 = Q2(torch.cat([s2_b, torch.zeros(batch_size, 1)], dim=1))
            q2_a1 = Q2(torch.cat([s2_b, torch.ones(batch_size, 1)], dim=1))
            target1 = torch.maximum(q2_a0, q2_a1)
        q1_pred = Q1(torch.cat([s1_b, a1_b], dim=1))
        loss1 = ((q1_pred - target1) ** 2).mean()
        loss = loss1 + loss2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(Q1.parameters()) + list(Q2.parameters()), 5.0)
        opt.step()
    return Q1, Q2


def run_fqi_hd_sweep(cfg):
    V = np.zeros((len(cfg['N_GRID_HD']), cfg['N_SEEDS_HD']))
    for i, N in enumerate(cfg['N_GRID_HD']):
        for s in tqdm(range(cfg['N_SEEDS_HD']), desc=f'  NN-FQI N={N}', leave=False,
                      disable=not sys.stderr.isatty()):
            rng = np.random.default_rng(N * 100 + s)
            S1, A1, S2, A2, Y = generate_cohort_hd(N, rng)
            Q1, Q2 = neural_fqi_estimate(S1, A1, S2, A2, Y, cfg['N_FQI_EPOCHS'], cfg['LR_NN'], cfg['HIDDEN_DIM'], seed=s)
            V[i, s] = evaluate_policy_hd(Q1, Q2, seed=s + 100)
    return {'V': V, 'N_grid': list(cfg['N_GRID_HD'])}


def run_dqn_hd_sweep(cfg):
    V = np.zeros((len(cfg['N_GRID_HD']), cfg['N_SEEDS_HD']))
    for i, N in enumerate(cfg['N_GRID_HD']):
        for s in tqdm(range(cfg['N_SEEDS_HD']), desc=f'  DQN N={N}', leave=False,
                      disable=not sys.stderr.isatty()):
            rng = np.random.default_rng(N * 100 + s + 7)
            S1, A1, S2, A2, Y = generate_cohort_hd(N, rng)
            Q1, Q2 = dqn_estimate(S1, A1, S2, A2, Y, cfg['N_DQN_STEPS'], cfg['BATCH_SIZE_HD'],
                                   cfg['LR_NN'], cfg['HIDDEN_DIM'], seed=s, rng=rng)
            V[i, s] = evaluate_policy_hd(Q1, Q2, seed=s + 100)
    return {'V': V, 'N_grid': list(cfg['N_GRID_HD'])}


# ============================================================================
# Pipeline
# ============================================================================
def compute_data(force=None):
    force = force or set()
    oracle = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'oracle', ORACLE_CONFIG,
        compute_oracle_tab, ORACLE_CONFIG,
        force=('oracle' in force),
    )
    murphy = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'murphy', MURPHY_CONFIG,
        run_murphy_sweep, MURPHY_CONFIG,
        force=('murphy' in force),
    )
    qlearn_N = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'qlearn_N', QLEARN_N_CONFIG,
        run_qlearn_N_sweep, QLEARN_N_CONFIG,
        force=('qlearn_N' in force),
    )
    qlearn_epochs = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'qlearn_epochs', QLEARN_EPOCHS_CONFIG,
        run_qlearn_epochs_sweep, QLEARN_EPOCHS_CONFIG,
        force=('qlearn_epochs' in force),
    )
    oracle_hd = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'oracle_hd', ORACLE_HD_CONFIG,
        compute_oracle_hd, ORACLE_HD_CONFIG,
        force=('oracle_hd' in force),
    )
    fqi_hd = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'fqi_hd', FQI_HD_CONFIG,
        run_fqi_hd_sweep, FQI_HD_CONFIG,
        force=('fqi_hd' in force),
    )
    dqn_hd = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'dqn_hd', DQN_HD_CONFIG,
        run_dqn_hd_sweep, DQN_HD_CONFIG,
        force=('dqn_hd' in force),
    )
    return {
        'oracle': oracle, 'murphy': murphy,
        'qlearn_N': qlearn_N, 'qlearn_epochs': qlearn_epochs,
        'oracle_hd': oracle_hd, 'fqi_hd': fqi_hd, 'dqn_hd': dqn_hd,
    }


# ============================================================================
# Outputs
# ============================================================================
def generate_outputs(data):
    oracle = data['oracle']
    murphy = data['murphy']
    qlearn_N = data['qlearn_N']
    qlearn_epochs = data['qlearn_epochs']
    oracle_hd = data['oracle_hd']
    fqi_hd = data['fqi_hd']
    dqn_hd = data['dqn_hd']
    V_star = oracle['V_star']
    V_star_hd = oracle_hd['V_star']

    print()
    print('=' * 72)
    print('  Murphy vs Q-learning equivalence -- simulation study')
    print('=' * 72)
    print()
    print('  (Q1) Tabular V(pi_hat)/V* vs cohort size N '
          f'[Q-learn 100 replays, alpha={ALPHA_QLEARN}]')
    print(f'  Tabular Oracle V* = {V_star:.4f}')
    print(f"  {'N':>6}  {'Murphy mean (SE)':>22}  {'Q-learn mean (SE)':>22}")
    n_grid = murphy['N_grid']
    for i, N in enumerate(n_grid):
        m_mean = murphy['V'][i].mean() / V_star
        m_se = murphy['V'][i].std() / np.sqrt(N_SEEDS) / V_star
        q_mean = qlearn_N['V'][i].mean() / V_star
        q_se = qlearn_N['V'][i].std() / np.sqrt(N_SEEDS) / V_star
        print(f'  {N:>6d}  {m_mean:>13.4f} ({m_se:.4f})  {q_mean:>13.4f} ({q_se:.4f})')
    print()
    print(f'  (Q2) Tabular Q-learn V(pi_hat)/V* vs replay epochs at N={N_AT_EPOCHS_PANEL}')
    print(f"  {'epochs':>6}  {'Q-learn mean (SE)':>22}")
    for i, ep in enumerate(qlearn_epochs['epochs_grid']):
        m = qlearn_epochs['V'][i].mean() / V_star
        s = qlearn_epochs['V'][i].std() / np.sqrt(N_SEEDS) / V_star
        print(f'  {ep:>6d}  {m:>13.4f} ({s:.4f})')
    idx_panel = n_grid.index(N_AT_EPOCHS_PANEL)
    m_panel = murphy['V'][idx_panel].mean() / V_star
    print(f'  Murphy reference at N={N_AT_EPOCHS_PANEL}: {m_panel:.4f}')
    print()
    print(f'  (Q3) High-dim V(pi_hat)/V* vs cohort size N '
          f'[FQI {N_FQI_EPOCHS} full-batch epochs, DQN {N_DQN_STEPS} minibatch steps]')
    print(f'  High-dim Oracle V* = {V_star_hd:.4f} '
          f'(behavior policy {oracle_hd["V_behavior"]:.4f})')
    print(f"  {'N':>6}  {'NN-FQI mean (SE)':>22}  {'DQN mean (SE)':>22}")
    for i, N in enumerate(fqi_hd['N_grid']):
        f_mean = fqi_hd['V'][i].mean() / V_star_hd
        f_se = fqi_hd['V'][i].std() / np.sqrt(N_SEEDS_HD) / V_star_hd
        d_mean = dqn_hd['V'][i].mean() / V_star_hd
        d_se = dqn_hd['V'][i].std() / np.sqrt(N_SEEDS_HD) / V_star_hd
        print(f'  {N:>6d}  {f_mean:>13.4f} ({f_se:.4f})  {d_mean:>13.4f} ({d_se:.4f})')
    print()

    # ---- Figure: 1 x 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: tabular V vs N
    ax = axes[0]
    n_arr = np.array(n_grid)
    m_means = murphy['V'].mean(axis=1) / V_star
    m_ses = murphy['V'].std(axis=1) / np.sqrt(N_SEEDS) / V_star
    q_means = qlearn_N['V'].mean(axis=1) / V_star
    q_ses = qlearn_N['V'].std(axis=1) / np.sqrt(N_SEEDS) / V_star
    ax.errorbar(n_arr, m_means, yerr=1.96 * m_ses, marker='o',
                label='Murphy (FQI)', color=COLORS['blue'], capsize=3)
    ax.errorbar(n_arr, q_means, yerr=1.96 * q_ses, marker='s',
                label=fr'$Q$-learning', color=COLORS['orange'], capsize=3)
    ax.axhline(1.0, **BENCH_STYLE, label=r'Oracle $V^*$')
    ax.set_xscale('log')
    ax.set_xlabel(r'Cohort size $N$')
    ax.set_ylabel(r'$V(\hat\pi) / V^*$')
    ax.set_title(r'(Q1) Tabular: same $V^*$ from both estimators')
    ax.legend(frameon=False, loc='lower right', fontsize=9)

    # Panel 2: tabular V vs epochs at fixed N
    ax = axes[1]
    ep_arr = np.array(qlearn_epochs['epochs_grid'])
    qe_means = qlearn_epochs['V'].mean(axis=1) / V_star
    qe_ses = qlearn_epochs['V'].std(axis=1) / np.sqrt(N_SEEDS) / V_star
    ax.errorbar(ep_arr, qe_means, yerr=1.96 * qe_ses, marker='s',
                label=fr'$Q$-learning', color=COLORS['orange'], capsize=3)
    ax.axhline(m_panel, color=COLORS['blue'], linestyle='--', linewidth=1.2,
               label=fr'Murphy at $N={N_AT_EPOCHS_PANEL}$')
    ax.axhline(1.0, **BENCH_STYLE, label=r'Oracle $V^*$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$Q$-learning replay epochs')
    ax.set_ylabel(r'$V(\hat\pi) / V^*$')
    ax.set_title(fr'(Q2) Tabular: budget matters at $N={N_AT_EPOCHS_PANEL}$')
    ax.legend(frameon=False, loc='lower right', fontsize=9)

    # Panel 3: high-dim V vs N
    ax = axes[2]
    n_hd_arr = np.array(fqi_hd['N_grid'])
    f_means = fqi_hd['V'].mean(axis=1) / V_star_hd
    f_ses = fqi_hd['V'].std(axis=1) / np.sqrt(N_SEEDS_HD) / V_star_hd
    d_means = dqn_hd['V'].mean(axis=1) / V_star_hd
    d_ses = dqn_hd['V'].std(axis=1) / np.sqrt(N_SEEDS_HD) / V_star_hd
    ax.errorbar(n_hd_arr, f_means, yerr=1.96 * f_ses, marker='o',
                label='Neural-FQI (Murphy)', color=COLORS['blue'], capsize=3)
    ax.errorbar(n_hd_arr, d_means, yerr=1.96 * d_ses, marker='s',
                label='DQN', color=COLORS['orange'], capsize=3)
    ax.axhline(1.0, **BENCH_STYLE, label=r'Oracle $V^*$')
    V_beh_norm = oracle_hd['V_behavior'] / V_star_hd
    ax.axhline(V_beh_norm, color=COLORS['gray'], linestyle=':', linewidth=1.0,
               label='Behavior policy')
    ax.set_xscale('log')
    ax.set_xlabel(r'Cohort size $N$')
    ax.set_ylabel(r'$V(\hat\pi) / V^*$')
    ax.set_title(fr'(Q3) High dim ($p={P_FEAT}$): NN analogues agree')
    ax.legend(frameon=False, loc='lower right', fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'dtr_qlearning_vs_murphy.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure: {fig_path}')


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print('Config:')
    print(f'  TABULAR: N_seeds={N_SEEDS}, N_GRID={N_GRID}, '
          f'N_EPOCHS_GRID={N_EPOCHS_GRID}, alpha_qlearn={ALPHA_QLEARN}, '
          f'n_epochs_default={N_EPOCHS_DEFAULT}, N_panel={N_AT_EPOCHS_PANEL}')
    print(f'  TABULAR DGP: S in 1..{N_STATES}, beta=({BETA_S},{BETA_A},{BETA_SA}), '
          f'sigma_Y={SIGMA_Y}, S_TREAT={S_TREAT}')
    print(f'  HIGH-DIM: p={P_FEAT}, N_seeds={N_SEEDS_HD}, N_GRID_HD={N_GRID_HD}, '
          f'FQI epochs={N_FQI_EPOCHS}, DQN steps={N_DQN_STEPS}, hidden={HIDDEN_DIM}, lr={LR_NN}')
    print(f'  HIGH-DIM DGP: T_decay={T_DECAY}, delta={DELTA_HD}, '
          f'sigma_eta={SIGMA_ETA_HD}, sigma_Y={SIGMA_Y_HD}, threshold={THRESHOLD_HD}')
    if force:
        print(f'  forcing recompute of: {sorted(force)}')

    if args.plots_only:
        data = compute_data()
    else:
        data = compute_data(force=force)
    if not args.data_only:
        generate_outputs(data)


if __name__ == '__main__':
    main()
