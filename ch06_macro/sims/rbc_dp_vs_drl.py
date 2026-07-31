# RBC: DP vs DRL. Chapter: RL for Macroeconomic Models.
# Compares VFI (DP), PPO and DDPG (DRL), and a constant-savings-rate
# analytical baseline on a representative-agent stochastic RBC model.

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import (
    apply_style, COLORS, ALGO_COLORS, BENCH_STYLE, FIG_SINGLE,
)
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'rbc_dp_vs_drl'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_PARAMS = dict(
    beta=0.96, alpha=0.36, delta=0.10, rho=0.95, sigma=0.007,
    T_episode=200,
)

SHARED_CONFIG = {
    **ENV_PARAMS,
    'N_SEEDS': 10,
    'N_EVAL_EPISODES': 30,
    'EVAL_T': 200,
    'K_LOW': 0.5, 'K_HIGH': 8.0,
    'A_LOW': 0.95, 'A_HIGH': 1.05,
}

KPR_CONFIG = {**SHARED_CONFIG}

VFI_CONFIG = {
    **SHARED_CONFIG,
    'K_GRID_SIZE': 400, 'A_GRID_SIZE': 41,
    'K_GRID_MIN': 0.1, 'K_GRID_MAX': 12.0,
    'VFI_TOL': 1e-5, 'VFI_MAX_ITER': 800,
}

PPO_CONFIG = {
    **SHARED_CONFIG,
    'HIDDEN_DIM': 64,
    'LR': 3e-4,
    'GAE_LAMBDA': 0.95,
    'CLIP_EPS': 0.2,
    'N_UPDATES': 100,
    'N_STEPS_PER_UPDATE': 1024,
    'N_PPO_EPOCHS': 4,
    'MINIBATCH_SIZE': 64,
    'ENTROPY_COEF': 0.0,
}

DDPG_CONFIG = {
    **SHARED_CONFIG,
    'HIDDEN_DIM': 64,
    'LR_ACTOR': 1e-4,
    'LR_CRITIC': 1e-3,
    'TAU': 0.005,
    'BUFFER_SIZE': 50_000,
    'BATCH_SIZE': 64,
    'NOISE_STD': 0.1,
    'N_STEPS': 60_000,
    'WARMUP_STEPS': 1000,
    'UPDATE_EVERY': 1,
}

ALGO_LIST = ['VFI', 'PPO', 'DDPG']  # KPR is closed-form, computed alongside shared

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RBCEnv:
    """Standard representative-agent stochastic RBC.
    State: (K, A); action: c (consumption). Log utility, Cobb-Douglas, AR(1) TFP.
    """
    def __init__(self, params):
        self.beta = params['beta']
        self.alpha = params['alpha']
        self.delta = params['delta']
        self.rho = params['rho']
        self.sigma = params['sigma']
        self.T = params['T_episode']
        self.A_ss = 1.0
        self.K_ss = ((1.0 / self.beta - 1.0 + self.delta) / self.alpha) ** (1.0 / (self.alpha - 1.0))
        self.C_ss = self.A_ss * self.K_ss ** self.alpha - self.delta * self.K_ss

    def reset(self, K0, A0):
        self.K = float(K0)
        self.A = float(A0)
        self.t = 0
        return np.array([self.K, self.A], dtype=np.float32)

    def step(self, c, rng):
        Y = self.A * self.K ** self.alpha
        W = Y + (1.0 - self.delta) * self.K
        c = float(np.clip(c, 1e-4, W - 1e-4))
        K_next = W - c
        eps = rng.normal(0.0, self.sigma)
        A_next = float(np.exp(self.rho * np.log(self.A) + eps))
        reward = float(np.log(c))
        self.K, self.A = K_next, A_next
        self.t += 1
        done = self.t >= self.T
        return np.array([self.K, self.A], dtype=np.float32), reward, done


def wealth(K, A, params):
    return A * K ** params['alpha'] + (1.0 - params['delta']) * K

# ---------------------------------------------------------------------------
# KPR analytical baseline (log-linearisation around the deterministic steady state)
# ---------------------------------------------------------------------------

def compute_kpr_coefs(params):
    """Solve for the log-linearised policy coefficients (eta_k, eta_a) such that
    log(c_t/C*) ≈ eta_k * log(K_t/K*) + eta_a * log(A_t / A*).
    Stability is enforced by selecting the saddle-path root.
    """
    alpha = params['alpha']; beta = params['beta']; delta = params['delta']; rho = params['rho']
    K_ss = ((1.0 / beta - 1.0 + delta) / alpha) ** (1.0 / (alpha - 1.0))
    C_ss = K_ss ** alpha - delta * K_ss
    zeta = 1.0 - beta * (1.0 - delta)
    CK = C_ss / K_ss
    Kalpham1 = K_ss ** (alpha - 1.0)

    # Quadratic in eta_k: a * eta_k^2 + b * eta_k + c = 0
    a = -CK
    b = (1.0 / beta) - 1.0 + zeta * (alpha - 1.0) * CK
    c = -zeta * (alpha - 1.0) / beta
    disc = b * b - 4.0 * a * c
    if disc < 0:
        raise ValueError("Negative discriminant in KPR root selection.")
    r1 = (-b + np.sqrt(disc)) / (2.0 * a)
    r2 = (-b - np.sqrt(disc)) / (2.0 * a)
    eig1 = 1.0 / beta - CK * r1
    eig2 = 1.0 / beta - CK * r2
    if abs(eig1) < 1.0 and abs(eig2) >= 1.0:
        eta_k = r1
    elif abs(eig2) < 1.0 and abs(eig1) >= 1.0:
        eta_k = r2
    elif abs(eig1) < abs(eig2):
        eta_k = r1
    else:
        eta_k = r2

    # eta_a from the matching equation on tilde a_t
    coef = 1.0 + (eta_k - zeta * (alpha - 1.0)) * CK - rho
    rhs = (eta_k - zeta * (alpha - 1.0)) * Kalpham1 - zeta * rho
    eta_a = rhs / coef
    return eta_k, eta_a, K_ss, C_ss


def kpr_policy_factory(params):
    eta_k, eta_a, K_ss, C_ss = compute_kpr_coefs(params)

    def policy(K, A):
        tilde_k = np.log(max(K, 1e-6) / K_ss)
        tilde_a = np.log(max(A, 1e-6))
        tilde_c = eta_k * tilde_k + eta_a * tilde_a
        c = C_ss * np.exp(tilde_c)
        W = wealth(K, A, params)
        return float(np.clip(c, 1e-4, W - 1e-4))

    return policy, eta_k, eta_a


def kpr_solution(params):
    policy, eta_k, eta_a = kpr_policy_factory(params)
    return {'policy_fn': policy, 'eta_k': eta_k, 'eta_a': eta_a}

# ---------------------------------------------------------------------------
# Tauchen discretisation for AR(1) TFP
# ---------------------------------------------------------------------------

def tauchen(N, rho, sigma, mu=0.0, m=3.0):
    """Discretise log(A_t) = rho * log(A_{t-1}) + eps_t into an N-state Markov chain."""
    sd = sigma / np.sqrt(1.0 - rho ** 2)
    grid = np.linspace(mu - m * sd, mu + m * sd, N)
    step = grid[1] - grid[0]
    P = np.zeros((N, N))
    from scipy.stats import norm
    for i in range(N):
        for j in range(N):
            if j == 0:
                P[i, j] = norm.cdf((grid[0] - rho * grid[i] + step / 2) / sigma)
            elif j == N - 1:
                P[i, j] = 1.0 - norm.cdf((grid[-1] - rho * grid[i] - step / 2) / sigma)
            else:
                P[i, j] = (norm.cdf((grid[j] - rho * grid[i] + step / 2) / sigma)
                           - norm.cdf((grid[j] - rho * grid[i] - step / 2) / sigma))
    return grid, P

# ---------------------------------------------------------------------------
# VFI on a (K, log A) grid
# ---------------------------------------------------------------------------

def vfi_solve(config):
    """Value-function iteration on a discretised RBC. Returns (K_grid, A_grid, policy_C, V)."""
    alpha = config['alpha']; beta = config['beta']; delta = config['delta']
    rho = config['rho']; sigma = config['sigma']

    K_grid = np.linspace(config['K_GRID_MIN'], config['K_GRID_MAX'], config['K_GRID_SIZE'])
    logA_grid, P_A = tauchen(config['A_GRID_SIZE'], rho, sigma)
    A_grid = np.exp(logA_grid)

    nK, nA = len(K_grid), len(A_grid)
    Y = np.einsum('a,k->ak', A_grid, K_grid ** alpha)
    W = Y + (1.0 - delta) * K_grid[np.newaxis, :]

    V = np.log(np.maximum(W - delta * K_grid[np.newaxis, :], 1e-6))
    policy_C = np.zeros_like(V)

    for it in range(config['VFI_MAX_ITER']):
        EV = P_A @ V
        C_grid = W[:, :, np.newaxis] - K_grid[np.newaxis, np.newaxis, :]
        feasible = C_grid > 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            util = np.where(feasible, np.log(np.where(feasible, C_grid, 1.0)), -1e10)
            cont = beta * EV[:, np.newaxis, :]
            total = util + cont
        best_idx = np.argmax(total, axis=2)
        V_new = np.take_along_axis(total, best_idx[:, :, np.newaxis], axis=2).squeeze(axis=2)
        policy_C = np.take_along_axis(C_grid, best_idx[:, :, np.newaxis], axis=2).squeeze(axis=2)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < config['VFI_TOL']:
            print(f"    VFI converged at iteration {it+1}, max-diff = {diff:.2e}")
            break

    def policy_fn(K, A):
        i_A = int(np.argmin(np.abs(A_grid - A)))
        i_K = int(np.clip(np.searchsorted(K_grid, K), 1, nK - 1))
        K_lo, K_hi = K_grid[i_K - 1], K_grid[i_K]
        w = (K - K_lo) / max(K_hi - K_lo, 1e-12)
        w = float(np.clip(w, 0.0, 1.0))
        c_lo = policy_C[i_A, i_K - 1]
        c_hi = policy_C[i_A, i_K]
        return float((1 - w) * c_lo + w * c_hi)

    return {
        'K_grid': K_grid, 'A_grid': A_grid, 'P_A': P_A,
        'policy_C': policy_C, 'V': V,
        'policy_fn': policy_fn,
    }

# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def consumption_from_action(a, W):
    # a in unbounded R -> savings fraction in (0, 1) via sigmoid -> c = (1 - sav) * W
    sav = 1.0 / (1.0 + np.exp(-a))
    return float(np.clip((1.0 - sav) * W, 1e-4, W - 1e-4))


def deterministic_eval(policy_fn, params, eval_subset, T):
    """Evaluate a policy on a fixed list of (K0, A0, shocks) tuples; return mean return."""
    env = RBCEnv(params)
    rets = []
    for (K0, A0, shocks) in eval_subset:
        obs = env.reset(K0, A0)
        ep_ret = 0.0
        T_ep = min(T, len(shocks))
        for t in range(T_ep):
            c = policy_fn(float(obs[0]), float(obs[1]))
            Y = env.A * env.K ** env.alpha
            W = Y + (1.0 - env.delta) * env.K
            c_eff = float(np.clip(c, 1e-4, W - 1e-4))
            K_next = W - c_eff
            A_next = float(np.exp(env.rho * np.log(env.A) + shocks[t]))
            r = float(np.log(c_eff))
            env.K, env.A = K_next, A_next
            obs = np.array([env.K, env.A], dtype=np.float32)
            ep_ret += r
        rets.append(ep_ret)
    return float(np.mean(rets))


def ppo_train_one_seed(seed, config, params, env_factory, eval_subset=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = env_factory()
    obs_dim = 2
    actor = GaussianActor(obs_dim, config['HIDDEN_DIM'])
    critic = Critic(obs_dim, config['HIDDEN_DIM'])
    opt_a = optim.Adam(actor.parameters(), lr=config['LR'])
    opt_c = optim.Adam(critic.parameters(), lr=config['LR'])

    learning_curve = []
    eval_curve = []

    def current_policy_fn(K, A):
        obs_t = torch.tensor([K, A], dtype=torch.float32)
        with torch.no_grad():
            dist = actor(obs_t)
            a = dist.mean.item()
        W = wealth(K, A, params)
        return consumption_from_action(a, W)

    def reset_random():
        K0 = rng.uniform(config['K_LOW'], config['K_HIGH'])
        A0 = rng.uniform(config['A_LOW'], config['A_HIGH'])
        return env.reset(K0, A0)

    obs = reset_random()
    for update in range(config['N_UPDATES']):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        ep_returns = []
        ep_ret = 0.0
        for _ in range(config['N_STEPS_PER_UPDATE']):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            dist = actor(obs_t)
            a = dist.sample()
            logp = dist.log_prob(a).sum()
            v = critic(obs_t)
            K, A = float(obs[0]), float(obs[1])
            W = wealth(K, A, params)
            c = consumption_from_action(float(a.item()), W)
            obs_next, r, done = env.step(c, rng)
            obs_buf.append(obs); act_buf.append(float(a.item())); logp_buf.append(float(logp.item()))
            rew_buf.append(r); val_buf.append(float(v.item())); done_buf.append(done)
            ep_ret += r
            obs = obs_next
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs = reset_random()

        # GAE
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.float32)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)
        with torch.no_grad():
            v_last = float(critic(torch.tensor(obs, dtype=torch.float32)).item())
        adv = np.zeros_like(rew_arr)
        last_gae = 0.0
        next_v = v_last
        for t in reversed(range(len(rew_arr))):
            nonterminal = 1.0 - done_arr[t]
            delta = rew_arr[t] + config['beta'] * next_v * nonterminal - val_arr[t]
            last_gae = delta + config['beta'] * config['GAE_LAMBDA'] * nonterminal * last_gae
            adv[t] = last_gae
            next_v = val_arr[t]
        ret_arr = adv + val_arr
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        obs_t = torch.tensor(obs_arr); act_t = torch.tensor(act_arr).unsqueeze(-1)
        logp_t = torch.tensor(logp_arr); adv_t = torch.tensor(adv); ret_t = torch.tensor(ret_arr)
        n = len(rew_arr)
        idx = np.arange(n)
        for _ in range(config['N_PPO_EPOCHS']):
            rng.shuffle(idx)
            for start in range(0, n, config['MINIBATCH_SIZE']):
                mb = idx[start:start + config['MINIBATCH_SIZE']]
                dist = actor(obs_t[mb])
                new_logp = dist.log_prob(act_t[mb]).sum(-1)
                ratio = torch.exp(new_logp - logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - config['CLIP_EPS'], 1 + config['CLIP_EPS']) * adv_t[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                v_pred = critic(obs_t[mb])
                loss_v = ((v_pred - ret_t[mb]) ** 2).mean()
                opt_a.zero_grad(); loss_pi.backward(); opt_a.step()
                opt_c.zero_grad(); loss_v.backward(); opt_c.step()

        learning_curve.append(np.mean(ep_returns) if ep_returns else 0.0)
        if eval_subset is not None:
            eval_curve.append(deterministic_eval(current_policy_fn, params, eval_subset, config['EVAL_T']))

    return {
        'policy_fn': current_policy_fn,
        'learning_curve': np.array(learning_curve),
        'eval_curve': np.array(eval_curve),
    }

# ---------------------------------------------------------------------------
# DDPG
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action.unsqueeze(-1) if action.dim() == 1 else action], dim=-1)
        return self.net(x).squeeze(-1)


def ddpg_train_one_seed(seed, config, params, env_factory, eval_subset=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = env_factory()
    obs_dim = 2
    actor = DeterministicActor(obs_dim, config['HIDDEN_DIM'])
    critic = QNet(obs_dim, config['HIDDEN_DIM'])
    actor_t = DeterministicActor(obs_dim, config['HIDDEN_DIM'])
    critic_t = QNet(obs_dim, config['HIDDEN_DIM'])
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_a = optim.Adam(actor.parameters(), lr=config['LR_ACTOR'])
    opt_c = optim.Adam(critic.parameters(), lr=config['LR_CRITIC'])

    buf_obs = np.zeros((config['BUFFER_SIZE'], obs_dim), dtype=np.float32)
    buf_act = np.zeros(config['BUFFER_SIZE'], dtype=np.float32)
    buf_rew = np.zeros(config['BUFFER_SIZE'], dtype=np.float32)
    buf_next = np.zeros((config['BUFFER_SIZE'], obs_dim), dtype=np.float32)
    buf_done = np.zeros(config['BUFFER_SIZE'], dtype=np.float32)
    buf_size = 0
    buf_pos = 0

    learning_curve = []
    eval_curve = []
    ep_returns_window = []
    ep_ret = 0.0

    def current_policy_fn(K, A):
        with torch.no_grad():
            a = float(actor(torch.tensor([K, A], dtype=torch.float32)).item())
        W = wealth(K, A, params)
        return consumption_from_action(a, W)

    def reset_random():
        K0 = rng.uniform(config['K_LOW'], config['K_HIGH'])
        A0 = rng.uniform(config['A_LOW'], config['A_HIGH'])
        return env.reset(K0, A0)

    obs = reset_random()
    eval_every = max(config['N_STEPS'] // 200, 1)
    for step_ in range(config['N_STEPS']):
        if step_ < config['WARMUP_STEPS']:
            a = rng.normal(0.0, 1.0)
        else:
            with torch.no_grad():
                a_det = float(actor(torch.tensor(obs, dtype=torch.float32)).item())
            a = a_det + rng.normal(0.0, config['NOISE_STD'])
        K, A = float(obs[0]), float(obs[1])
        W = wealth(K, A, params)
        c = consumption_from_action(a, W)
        obs_next, r, done = env.step(c, rng)
        ep_ret += r

        # store
        buf_obs[buf_pos] = obs
        buf_act[buf_pos] = a
        buf_rew[buf_pos] = r
        buf_next[buf_pos] = obs_next
        buf_done[buf_pos] = float(done)
        buf_pos = (buf_pos + 1) % config['BUFFER_SIZE']
        buf_size = min(buf_size + 1, config['BUFFER_SIZE'])

        obs = obs_next
        if done:
            ep_returns_window.append(ep_ret)
            if len(ep_returns_window) > 20:
                ep_returns_window.pop(0)
            ep_ret = 0.0
            obs = reset_random()

        if buf_size >= config['BATCH_SIZE'] and step_ >= config['WARMUP_STEPS'] and step_ % config['UPDATE_EVERY'] == 0:
            idx = rng.integers(0, buf_size, size=config['BATCH_SIZE'])
            o = torch.tensor(buf_obs[idx])
            ac = torch.tensor(buf_act[idx])
            rw = torch.tensor(buf_rew[idx])
            on = torch.tensor(buf_next[idx])
            dn = torch.tensor(buf_done[idx])
            with torch.no_grad():
                a_next = actor_t(on)
                q_next = critic_t(on, a_next)
                y = rw + config['beta'] * (1.0 - dn) * q_next
            q = critic(o, ac)
            loss_c = ((q - y) ** 2).mean()
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
            a_pred = actor(o)
            loss_a = -critic(o, a_pred).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.data.mul_(1 - config['TAU']); pt.data.add_(config['TAU'] * p.data)
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.data.mul_(1 - config['TAU']); pt.data.add_(config['TAU'] * p.data)

        if (step_ + 1) % eval_every == 0:
            learning_curve.append(np.mean(ep_returns_window) if ep_returns_window else 0.0)
            if eval_subset is not None:
                eval_curve.append(deterministic_eval(current_policy_fn, params, eval_subset, config['EVAL_T']))

    policy_fn = current_policy_fn

    return {
        'policy_fn': policy_fn,
        'learning_curve': np.array(learning_curve),
        'eval_curve': np.array(eval_curve),
    }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(policy_fn, params, n_eps, T, K_low, K_high, A_low, A_high, seed, eval_set=None):
    """Evaluate policy over a set of episodes with fixed (K0, A0, shock sequence) tuples.
    If eval_set is provided, replay those episodes; otherwise sample n_eps episodes.
    """
    env = RBCEnv(params)
    returns = []
    capital_traj = []
    if eval_set is None:
        rng = np.random.default_rng(seed + 99991)
        eval_set = []
        for ep in range(n_eps):
            K0 = rng.uniform(K_low, K_high)
            A0 = rng.uniform(A_low, A_high)
            shocks = rng.normal(0.0, params['sigma'], size=T)
            eval_set.append((K0, A0, shocks))
    for (K0, A0, shocks) in eval_set:
        obs = env.reset(K0, A0)
        ep_ret = 0.0
        K_path = [float(obs[0])]
        T_ep = min(T, len(shocks))
        for t in range(T_ep):
            c = policy_fn(float(obs[0]), float(obs[1]))
            # Deterministic step using pre-drawn shock
            Y = env.A * env.K ** env.alpha
            W = Y + (1.0 - env.delta) * env.K
            c_eff = float(np.clip(c, 1e-4, W - 1e-4))
            K_next = W - c_eff
            A_next = float(np.exp(env.rho * np.log(env.A) + shocks[t]))
            r = float(np.log(c_eff))
            env.K, env.A = K_next, A_next
            obs = np.array([env.K, env.A], dtype=np.float32)
            ep_ret += r
            K_path.append(float(obs[0]))
        returns.append(ep_ret)
        capital_traj.append(np.array(K_path))
    return np.array(returns), capital_traj


def policy_mse_against_reference(policy_fn, ref_fn, test_states):
    diffs = []
    for K, A in test_states:
        c1 = policy_fn(K, A)
        c2 = ref_fn(K, A)
        diffs.append((c1 - c2) ** 2)
    return float(np.mean(diffs))

# ---------------------------------------------------------------------------
# Per-component compute functions
# ---------------------------------------------------------------------------

def compute_shared(config):
    params = {k: config[k] for k in ENV_PARAMS}
    env = RBCEnv(params)
    rng = np.random.default_rng(0)
    test_states = [(rng.uniform(config['K_LOW'], config['K_HIGH']),
                    rng.uniform(config['A_LOW'], config['A_HIGH'])) for _ in range(1000)]

    # Shared evaluation set: same initial conditions and shock paths for every algorithm.
    eval_rng = np.random.default_rng(99991)
    eval_set = []
    for _ in range(config['N_EVAL_EPISODES']):
        K0 = eval_rng.uniform(config['K_LOW'], config['K_HIGH'])
        A0 = eval_rng.uniform(config['A_LOW'], config['A_HIGH'])
        shocks = eval_rng.normal(0.0, params['sigma'], size=config['EVAL_T'])
        eval_set.append((K0, A0, shocks))

    print(f"    Steady state: K* = {env.K_ss:.3f}, C* = {env.C_ss:.3f}")
    return {
        'params': params,
        'K_ss': env.K_ss, 'C_ss': env.C_ss,
        'test_states': test_states,
        'eval_set': eval_set,
    }


def compute_kpr(shared, config):
    params = shared['params']
    sol = kpr_solution(params)
    rets, _ = evaluate_policy(
        sol['policy_fn'], params, n_eps=config['N_EVAL_EPISODES'], T=config['EVAL_T'],
        K_low=config['K_LOW'], K_high=config['K_HIGH'],
        A_low=config['A_LOW'], A_high=config['A_HIGH'], seed=42, eval_set=shared['eval_set'],
    )
    print(f"    KPR coefficients: eta_k = {sol['eta_k']:.4f}, eta_a = {sol['eta_a']:.4f}")
    return {
        'eta_k': float(sol['eta_k']),
        'eta_a': float(sol['eta_a']),
        'mean_return': float(np.mean(rets)),
        'se_return': float(np.std(rets) / np.sqrt(len(rets))),
        'eval_returns': rets,
    }


def compute_vfi(shared, config):
    t0 = time.time()
    sol = vfi_solve(config)
    vfi_time = time.time() - t0
    rets, traj = evaluate_policy(
        sol['policy_fn'], shared['params'],
        n_eps=config['N_EVAL_EPISODES'], T=config['EVAL_T'],
        K_low=config['K_LOW'], K_high=config['K_HIGH'],
        A_low=config['A_LOW'], A_high=config['A_HIGH'], seed=42, eval_set=shared['eval_set'],
    )
    return {
        'K_grid': sol['K_grid'], 'A_grid': sol['A_grid'],
        'policy_C': sol['policy_C'], 'V': sol['V'],
        'mean_return': float(np.mean(rets)),
        'se_return': float(np.std(rets) / np.sqrt(len(rets))),
        'policy_C_grid': sol['policy_C'],  # consumption policy on (A, K) grid; retained for parity with cache schema
        'eval_returns': rets,
        'eval_traj': traj,
        'vfi_time_sec': vfi_time,
    }


def compute_ppo(shared, config):
    params = shared['params']
    seeds = list(range(config['N_SEEDS']))
    eval_subset = shared['eval_set'][:5]  # smaller eval set for in-training eval
    learning_curves = []
    eval_curves = []
    mean_returns = []
    se_returns = []
    policies = []
    train_times = []
    for seed in tqdm(seeds, desc='PPO seeds'):
        t0 = time.time()
        result = ppo_train_one_seed(seed, config, params, lambda: RBCEnv(params), eval_subset=eval_subset)
        train_times.append(time.time() - t0)
        rets, _ = evaluate_policy(
            result['policy_fn'], params,
            n_eps=config['N_EVAL_EPISODES'], T=config['EVAL_T'],
            K_low=config['K_LOW'], K_high=config['K_HIGH'],
            A_low=config['A_LOW'], A_high=config['A_HIGH'], seed=seed + 1000,
            eval_set=shared['eval_set'],
        )
        learning_curves.append(result['learning_curve'])
        eval_curves.append(result['eval_curve'])
        mean_returns.append(float(np.mean(rets)))
        se_returns.append(float(np.std(rets) / np.sqrt(len(rets))))
        polvals = np.array([result['policy_fn'](K, A) for (K, A) in shared['test_states']])
        policies.append(polvals)
    return {
        'learning_curves': np.array(learning_curves),
        'eval_curves': np.array(eval_curves),
        'mean_returns': np.array(mean_returns),
        'se_returns': np.array(se_returns),
        'policy_on_test_states': np.array(policies),
        'train_time_sec_mean': float(np.mean(train_times)),
        'train_times_sec': np.array(train_times),
    }


def compute_ddpg(shared, config):
    params = shared['params']
    seeds = list(range(config['N_SEEDS']))
    eval_subset = shared['eval_set'][:5]
    learning_curves = []
    eval_curves = []
    mean_returns = []
    se_returns = []
    policies = []
    train_times = []
    for seed in tqdm(seeds, desc='DDPG seeds'):
        t0 = time.time()
        result = ddpg_train_one_seed(seed, config, params, lambda: RBCEnv(params), eval_subset=eval_subset)
        train_times.append(time.time() - t0)
        rets, _ = evaluate_policy(
            result['policy_fn'], params,
            n_eps=config['N_EVAL_EPISODES'], T=config['EVAL_T'],
            K_low=config['K_LOW'], K_high=config['K_HIGH'],
            A_low=config['A_LOW'], A_high=config['A_HIGH'], seed=seed + 2000,
            eval_set=shared['eval_set'],
        )
        learning_curves.append(result['learning_curve'])
        eval_curves.append(result['eval_curve'])
        mean_returns.append(float(np.mean(rets)))
        se_returns.append(float(np.std(rets) / np.sqrt(len(rets))))
        polvals = np.array([result['policy_fn'](K, A) for (K, A) in shared['test_states']])
        policies.append(polvals)
    return {
        'learning_curves': np.array(learning_curves),
        'eval_curves': np.array(eval_curves),
        'mean_returns': np.array(mean_returns),
        'se_returns': np.array(se_returns),
        'policy_on_test_states': np.array(policies),
        'train_time_sec_mean': float(np.mean(train_times)),
        'train_times_sec': np.array(train_times),
    }


def compute_data(force=None):
    force = force or set()
    config = SHARED_CONFIG
    shared = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'shared', config, compute_shared,
                              config, force=('shared' in force))
    kpr = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'KPR', KPR_CONFIG, compute_kpr,
                          shared, KPR_CONFIG, force=('KPR' in force or 'shared' in force))
    vfi = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'VFI', VFI_CONFIG, compute_vfi,
                          shared, VFI_CONFIG, force=('VFI' in force or 'shared' in force))
    ppo = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'PPO', PPO_CONFIG, compute_ppo,
                          shared, PPO_CONFIG, force=('PPO' in force or 'shared' in force))
    ddpg = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'DDPG', DDPG_CONFIG, compute_ddpg,
                            shared, DDPG_CONFIG, force=('DDPG' in force or 'shared' in force))
    return {'shared': shared, 'KPR': kpr, 'VFI': vfi, 'PPO': ppo, 'DDPG': ddpg}

# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_outputs(data):
    shared = data['shared']; params = shared['params']
    kpr = data['KPR']; vfi = data['VFI']; ppo = data['PPO']; ddpg = data['DDPG']

    # Reference: VFI policy on test states
    vfi_test_C = np.zeros(len(shared['test_states']))
    K_grid, A_grid = vfi['K_grid'], vfi['A_grid']
    policy_C = vfi['policy_C']
    def vfi_policy_fn(K, A):
        i_A = int(np.argmin(np.abs(A_grid - A)))
        i_K = int(np.clip(np.searchsorted(K_grid, K), 1, len(K_grid) - 1))
        K_lo, K_hi = K_grid[i_K - 1], K_grid[i_K]
        w = (K - K_lo) / max(K_hi - K_lo, 1e-12)
        w = float(np.clip(w, 0.0, 1.0))
        return float((1 - w) * policy_C[i_A, i_K - 1] + w * policy_C[i_A, i_K])
    for i, (K, A) in enumerate(shared['test_states']):
        vfi_test_C[i] = vfi_policy_fn(K, A)

    # Policy MSE vs VFI
    ppo_mse = float(np.mean((ppo['policy_on_test_states'] - vfi_test_C[np.newaxis, :]) ** 2))
    ddpg_mse = float(np.mean((ddpg['policy_on_test_states'] - vfi_test_C[np.newaxis, :]) ** 2))
    kpr_pol, _, _ = kpr_policy_factory(params)
    kpr_C = np.array([kpr_pol(K, A) for (K, A) in shared['test_states']])
    kpr_mse = float(np.mean((kpr_C - vfi_test_C) ** 2))

    # ---- Learning-curve figure (deterministic eval during training) ----
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    # PPO eval curve (one point per training update)
    pc = ppo.get('eval_curves', ppo['learning_curves'])
    n_x = pc.shape[1]
    pc_x = np.linspace(0.0, 1.0, n_x)
    pc_mean = pc.mean(axis=0)
    pc_se = pc.std(axis=0) / np.sqrt(pc.shape[0])
    ax.plot(pc_x, pc_mean, label='PPO', color=ALGO_COLORS['PPO'])
    ax.fill_between(pc_x, pc_mean - pc_se, pc_mean + pc_se, color=ALGO_COLORS['PPO'], alpha=0.2)
    # DDPG eval curve (normalized to same [0, 1] training-progress axis)
    dc = ddpg.get('eval_curves', ddpg['learning_curves'])
    dc_x = np.linspace(0.0, 1.0, dc.shape[1])
    dc_mean = dc.mean(axis=0)
    dc_se = dc.std(axis=0) / np.sqrt(dc.shape[0])
    ax.plot(dc_x, dc_mean, label='DDPG', color=COLORS['cyan'])
    ax.fill_between(dc_x, dc_mean - dc_se, dc_mean + dc_se, color=COLORS['cyan'], alpha=0.2)
    # Reference lines
    ax.axhline(vfi['mean_return'], label='VFI', **BENCH_STYLE)
    ax.axhline(kpr['mean_return'], label='KPR', color=COLORS['gray'], linestyle=':', linewidth=1.0, zorder=1)
    ax.set_xlabel('Training progress (normalised)')
    ax.set_ylabel('Mean episode return (deterministic eval)')
    ax.set_title('Learning curves on representative-agent RBC')
    ax.set_ylim(-100, 55)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'rbc_dp_vs_drl_learning_curves.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    # ---- Results table ----
    def fmt(x, n=3):
        return f"{x:.{n}f}"
    ppo_wc = ppo.get('train_time_sec_mean')
    ddpg_wc = ddpg.get('train_time_sec_mean')
    rows = [
        ('KPR',  kpr['mean_return'], kpr['se_return'], kpr_mse, '$-$'),
        ('VFI',  vfi['mean_return'], vfi['se_return'], 0.0, fmt(vfi.get('vfi_time_sec', 0.0), 1) + ' s'),
        ('PPO',  float(np.mean(ppo['mean_returns'])), float(np.std(ppo['mean_returns']) / np.sqrt(len(ppo['mean_returns']))), ppo_mse,
            (fmt(ppo_wc, 1) + ' s') if ppo_wc is not None else '$-$'),
        ('DDPG', float(np.mean(ddpg['mean_returns'])), float(np.std(ddpg['mean_returns']) / np.sqrt(len(ddpg['mean_returns']))), ddpg_mse,
            (fmt(ddpg_wc, 1) + ' s') if ddpg_wc is not None else '$-$'),
    ]

    tex = []
    tex.append(r"\begin{tabular}{lrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Method & Mean return & SE & Policy MSE vs VFI & Wall clock \\")
    tex.append(r"\midrule")
    for name, mret, mse_s, mse, wc in rows:
        tex.append(f"{name} & {fmt(mret, 2)} & {fmt(mse_s, 3)} & {fmt(mse, 4)} & {wc} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex_path = os.path.join(OUTPUT_DIR, 'rbc_dp_vs_drl_results.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex) + "\n")
    print(f"  Saved table: {tex_path}")

    # ---- Stdout report ----
    print()
    print("=" * 60)
    print("Results summary")
    print("=" * 60)
    print(f"  Steady state: K* = {shared['K_ss']:.3f}, C* = {shared['C_ss']:.3f}")
    print(f"  Seeds per algorithm: {SHARED_CONFIG['N_SEEDS']}")
    print(f"  Evaluation episodes per seed: {SHARED_CONFIG['N_EVAL_EPISODES']}")
    print(f"  Evaluation horizon: {SHARED_CONFIG['EVAL_T']}")
    print()
    print(f"  {'Method':<8}{'Mean return':>14}{'SE':>10}{'Policy MSE vs VFI':>22}{'Wall clock':>14}")
    for name, mret, mse_s, mse, wc in rows:
        print(f"  {name:<8}{mret:>14.3f}{mse_s:>10.3f}{mse:>22.4f}{wc:>14}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    if args.plots_only:
        print("Loading cached data...")
        data = compute_data(force=set())
        generate_outputs(data)
        return

    print("Computing data...")
    data = compute_data(force=force)
    if args.data_only:
        print("Data-only mode: skipping output generation.")
        return
    generate_outputs(data)


if __name__ == '__main__':
    main()
