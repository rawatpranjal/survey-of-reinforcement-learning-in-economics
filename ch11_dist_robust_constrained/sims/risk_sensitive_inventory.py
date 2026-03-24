"""
Risk-Sensitive Inventory Management via IQN
Chapter 11: Distributional, Robust and Constrained RL

One IQN network trained on a newsvendor MDP produces four risk-sensitive
policies by changing the tau sampling distribution at decision time:
risk-neutral, CVaR-95, CVaR-99, and CPT.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_SINGLE, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args

apply_style()
import matplotlib.pyplot as plt

SCRIPT_NAME = 'risk_sensitive_inventory'
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.dirname(__file__)

# ── Configuration ────────────────────────────────────────────────────────────

ENV_PARAMS = {
    'S_MAX': 15,
    'A_MAX': 5,
    'HORIZON': 10,
    'GAMMA': 0.99,
    'REVENUE': 5.0,
    'ORDER_COST': 2.0,
    'HOLDING_COST': 1.0,
    'STOCKOUT_PENALTY': 8.0,
    'DEMAND_PROBS': [0.8, 0.2],  # mixture weights
    'DEMAND_LAMBDAS': [3.0, 10.0],  # Poisson rates
}

IQN_CONFIG = {
    'LAYER_SIZE': 64,
    'N_COS': 64,
    'N_TAU': 8,
    'LR': 1e-3,
    'BUFFER_SIZE': 100000,
    'BATCH_SIZE': 64,
    'GAMMA': ENV_PARAMS['GAMMA'],
    'TAU_TARGET': 0.01,
    'EPS_START': 1.0,
    'EPS_END': 0.02,
    'EPS_DECAY_STEPS': 10000,
    'N_TRAIN_EPISODES': 50000,
    'UPDATE_EVERY': 2,
    'KAPPA': 1.0,
}

EVAL_CONFIG = {
    'N_EVAL_EPISODES': 50000,
    'N_SEEDS': 1,
    'N_TAU_EVAL': 256,
}

CONFIG = {**ENV_PARAMS, **IQN_CONFIG, **EVAL_CONFIG, 'version': 4}

# ── Environment ──────────────────────────────────────────────────────────────

class NewsvendorMDP:
    def __init__(self, params):
        self.s_max = params['S_MAX']
        self.a_max = params['A_MAX']
        self.horizon = params['HORIZON']
        self.gamma = params['GAMMA']
        self.revenue = params['REVENUE']
        self.order_cost = params['ORDER_COST']
        self.holding_cost = params['HOLDING_COST']
        self.stockout_penalty = params['STOCKOUT_PENALTY']
        self.demand_probs = params['DEMAND_PROBS']
        self.demand_lambdas = params['DEMAND_LAMBDAS']
        self.n_states = self.s_max + 1
        self.n_actions = self.a_max + 1
        self._precompute_demand()

    def _precompute_demand(self):
        max_d = 40
        self.demand_pmf = np.zeros(max_d + 1)
        for prob, lam in zip(self.demand_probs, self.demand_lambdas):
            for d in range(max_d + 1):
                from math import factorial
                self.demand_pmf[d] += prob * (lam**d * np.exp(-lam)
                                              / factorial(d))
        self.demand_pmf /= self.demand_pmf.sum()
        self.max_demand = max_d

    def sample_demand(self, rng):
        return rng.choice(len(self.demand_pmf), p=self.demand_pmf)

    def step(self, state, action, demand):
        inventory_after_order = min(state + action, self.s_max)
        sold = min(inventory_after_order, demand)
        leftover = inventory_after_order - demand
        reward = (self.revenue * sold
                  - self.order_cost * action
                  - self.holding_cost * max(leftover, 0)
                  - self.stockout_penalty * max(-leftover, 0))
        next_state = max(min(leftover, self.s_max), 0)
        return next_state, reward

    def reset(self, rng):
        return rng.randint(0, self.n_states)

    def dp_solve(self):
        V = np.zeros((self.horizon + 1, self.n_states))
        policy = np.zeros((self.horizon, self.n_states), dtype=int)
        for t in range(self.horizon - 1, -1, -1):
            for s in range(self.n_states):
                best_val = -np.inf
                best_a = 0
                for a in range(self.n_actions):
                    val = 0.0
                    for d in range(len(self.demand_pmf)):
                        ns, r = self.step(s, a, d)
                        val += self.demand_pmf[d] * (r + self.gamma * V[t+1, ns])
                    if val > best_val:
                        best_val = val
                        best_a = a
                V[t, s] = best_val
                policy[t, s] = best_a
        return V, policy


# ── IQN Network ──────────────────────────────────────────────────────────────

class IQNNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, layer_size=128, n_cos=64):
        super().__init__()
        self.n_actions = n_actions
        self.n_cos = n_cos
        self.layer_size = layer_size
        self.pis = torch.FloatTensor(
            [np.pi * i for i in range(1, n_cos + 1)]
        ).view(1, 1, n_cos)
        self.state_head = nn.Linear(state_dim, layer_size)
        self.cos_embedding = nn.Linear(n_cos, layer_size)
        self.ff1 = nn.Linear(layer_size, layer_size)
        self.ff2 = nn.Linear(layer_size, n_actions)

    def forward(self, state, n_tau=32, taus=None):
        batch_size = state.shape[0]
        x = torch.relu(self.state_head(state))
        if taus is None:
            taus = torch.rand(batch_size, n_tau, 1)
        else:
            n_tau = taus.shape[1]
        cos_input = torch.cos(taus * self.pis.to(state.device))
        tau_embed = torch.relu(
            self.cos_embedding(cos_input.view(batch_size * n_tau, self.n_cos))
        ).view(batch_size, n_tau, self.layer_size)
        combined = (x.unsqueeze(1) * tau_embed).view(
            batch_size * n_tau, self.layer_size
        )
        combined = torch.relu(self.ff1(combined))
        out = self.ff2(combined)
        return out.view(batch_size, n_tau, self.n_actions), taus


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rng=None):
        if rng is not None:
            indices = rng.choice(len(self.buffer), batch_size, replace=False)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── Training ─────────────────────────────────────────────────────────────────

def train_iqn(env, config, seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    n_tau = config['N_TAU']
    kappa = config['KAPPA']

    net = IQNNetwork(1, env.n_actions, config['LAYER_SIZE'], config['N_COS'])
    target_net = IQNNetwork(1, env.n_actions, config['LAYER_SIZE'], config['N_COS'])
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=config['LR'])
    buffer = ReplayBuffer(config['BUFFER_SIZE'])

    eps = config['EPS_START']
    eps_decay = (config['EPS_START'] - config['EPS_END']) / config['EPS_DECAY_STEPS']
    total_steps = 0

    for episode in range(config['N_TRAIN_EPISODES']):
        state = env.reset(rng)
        episode_return = 0.0
        for t in range(env.horizon):
            state_t = torch.FloatTensor([[state / env.s_max]])
            if rng.random() < eps:
                action = rng.randint(0, env.n_actions)
            else:
                with torch.no_grad():
                    q_vals, _ = net(state_t, n_tau)
                    action = q_vals.mean(dim=1).argmax(dim=1).item()
            demand = env.sample_demand(rng)
            next_state, reward = env.step(state, action, demand)
            done = 1.0 if t == env.horizon - 1 else 0.0
            buffer.add(state / env.s_max, action, reward,
                       next_state / env.s_max, done)
            episode_return += reward * (env.gamma ** t)
            state = next_state
            total_steps += 1
            eps = max(config['EPS_END'], eps - eps_decay)

            if (total_steps % config['UPDATE_EVERY'] == 0
                    and len(buffer) >= config['BATCH_SIZE']):
                states, actions, rewards_b, next_states, dones = \
                    buffer.sample(config['BATCH_SIZE'])
                s_t = torch.FloatTensor(states).unsqueeze(1)
                a_t = torch.LongTensor(actions)
                r_t = torch.FloatTensor(rewards_b)
                ns_t = torch.FloatTensor(next_states).unsqueeze(1)
                d_t = torch.FloatTensor(dones)

                with torch.no_grad():
                    q_next, _ = target_net(ns_t, n_tau)
                    best_actions = q_next.mean(dim=1).argmax(dim=1)
                    q_next_best = q_next[
                        torch.arange(config['BATCH_SIZE']),
                        :, best_actions
                    ]
                    targets = r_t.unsqueeze(1) + config['GAMMA'] * (
                        1 - d_t.unsqueeze(1)) * q_next_best

                q_current, taus = net(s_t, n_tau)
                q_current_a = q_current[
                    torch.arange(config['BATCH_SIZE']), :, a_t
                ]

                td_error = targets.unsqueeze(1) - q_current_a.unsqueeze(2)
                huber = torch.where(
                    td_error.abs() <= kappa,
                    0.5 * td_error.pow(2),
                    kappa * (td_error.abs() - 0.5 * kappa)
                )
                tau_vals = taus.squeeze(-1).unsqueeze(2)
                quantile_loss = (
                    (tau_vals - (td_error.detach() < 0).float()).abs() * huber
                )
                loss = quantile_loss.sum(dim=2).mean(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                for tp, lp in zip(target_net.parameters(), net.parameters()):
                    tp.data.copy_(
                        config['TAU_TARGET'] * lp.data
                        + (1 - config['TAU_TARGET']) * tp.data
                    )

    return net


# ── Risk-Sensitive Action Selection ──────────────────────────────────────────

def cpt_distortion(tau, gamma_param=0.71):
    t = tau ** gamma_param
    omt = (1.0 - tau) ** gamma_param
    return t / (t + omt) ** (1.0 / gamma_param)


def sample_cpt_taus(n, rng):
    u = rng.uniform(0, 1, size=n).astype(np.float32)
    return cpt_distortion(u)


def iqn_act(net, state_scalar, s_max, tau_mode='uniform', alpha=1.0,
            n_tau_eval=64, rng=None):
    state_t = torch.FloatTensor([[state_scalar / s_max]])
    if tau_mode == 'uniform':
        taus = torch.rand(1, n_tau_eval, 1)
    elif tau_mode == 'cvar':
        taus = torch.rand(1, n_tau_eval, 1) * alpha
    elif tau_mode == 'cpt':
        raw = rng.uniform(0, 1, size=n_tau_eval).astype(np.float32)
        taus = torch.FloatTensor(cpt_distortion(raw)).view(1, n_tau_eval, 1)
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")
    with torch.no_grad():
        q_vals, _ = net(state_t, taus=taus)
    return q_vals.mean(dim=1).argmax(dim=1).item()


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_policy(policy_fn, env, n_episodes, seed):
    rng = np.random.RandomState(seed)
    returns = np.zeros(n_episodes)
    actions_by_state = np.zeros((n_episodes, env.horizon), dtype=int)
    states_visited = np.zeros((n_episodes, env.horizon), dtype=int)
    for ep in range(n_episodes):
        state = env.reset(rng)
        ep_return = 0.0
        for t in range(env.horizon):
            states_visited[ep, t] = state
            action = policy_fn(state, t)
            actions_by_state[ep, t] = action
            demand = env.sample_demand(rng)
            next_state, reward = env.step(state, action, demand)
            ep_return += reward * (env.gamma ** t)
            state = next_state
        returns[ep] = ep_return
    return returns, states_visited, actions_by_state


# ── Compute ──────────────────────────────────────────────────────────────────

def compute_data(force=False):
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached and not force:
        return cached

    env = NewsvendorMDP(ENV_PARAMS)
    print("Solving DP oracle...")
    V_dp, policy_dp = env.dp_solve()
    dp_init_val = np.mean([V_dp[0, s] for s in range(env.n_states)])
    print(f"  DP average initial value: {dp_init_val:.2f}")

    n_seeds = EVAL_CONFIG['N_SEEDS']
    n_eval = EVAL_CONFIG['N_EVAL_EPISODES']
    n_tau_eval = EVAL_CONFIG['N_TAU_EVAL']

    policy_configs = {
        'DP Oracle': {'type': 'dp'},
        'IQN-Neutral': {'type': 'iqn', 'tau_mode': 'uniform'},
        'IQN-CVaR95': {'type': 'iqn', 'tau_mode': 'cvar', 'alpha': 0.05},
        'IQN-CVaR99': {'type': 'iqn', 'tau_mode': 'cvar', 'alpha': 0.01},
        'IQN-CPT': {'type': 'iqn', 'tau_mode': 'cpt'},
    }

    all_returns = {name: [] for name in policy_configs}
    all_avg_orders = {name: [] for name in policy_configs}

    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds}")
        print("  Training IQN...")
        net = train_iqn(env, IQN_CONFIG, seed)

        for name, cfg in policy_configs.items():
            eval_rng = np.random.RandomState(seed + 10000)
            if cfg['type'] == 'dp':
                def dp_policy(s, t, _pol=policy_dp):
                    return _pol[t, s]
                returns, states, actions = evaluate_policy(
                    dp_policy, env, n_eval, seed + 10000)
            else:
                act_rng = np.random.RandomState(seed + 20000)
                def iqn_policy(s, t, _net=net, _cfg=cfg, _rng=act_rng):
                    return iqn_act(_net, s, env.s_max,
                                  tau_mode=_cfg['tau_mode'],
                                  alpha=_cfg.get('alpha', 1.0),
                                  n_tau_eval=n_tau_eval, rng=_rng)
                returns, states, actions = evaluate_policy(
                    iqn_policy, env, n_eval, seed + 10000)

            all_returns[name].append(returns)
            avg_order_by_inv = np.zeros(env.n_states)
            counts = np.zeros(env.n_states)
            for ep in range(n_eval):
                for t in range(env.horizon):
                    s = states[ep, t]
                    avg_order_by_inv[s] += actions[ep, t]
                    counts[s] += 1
            mask = counts > 0
            avg_order_by_inv[mask] /= counts[mask]
            all_avg_orders[name].append(avg_order_by_inv)
            mean_ret = returns.mean()
            cvar95 = np.mean(np.sort(returns)[:int(0.05 * n_eval)])
            print(f"  {name}: mean={mean_ret:.1f}, CVaR95={cvar95:.1f}")

    data = {
        'returns': {k: np.array(v) for k, v in all_returns.items()},
        'avg_orders': {k: np.array(v) for k, v in all_avg_orders.items()},
        'dp_init_val': dp_init_val,
        'demand_pmf': env.demand_pmf,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ── Outputs ──────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    'DP Oracle': COLORS['black'],
    'IQN-Neutral': COLORS['blue'],
    'IQN-CVaR95': COLORS['orange'],
    'IQN-CVaR99': COLORS['red'],
    'IQN-CPT': COLORS['purple'],
}

POLICY_STYLES = {
    'DP Oracle': '--',
    'IQN-Neutral': '-',
    'IQN-CVaR95': '-',
    'IQN-CVaR99': '-',
    'IQN-CPT': '-',
}


def generate_outputs(data):
    returns = data['returns']
    avg_orders = data['avg_orders']
    n_seeds = len(list(returns.values())[0])
    n_eval = list(returns.values())[0].shape[1]
    policy_names = ['DP Oracle', 'IQN-Neutral', 'IQN-CVaR95']

    # ── Table ────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("RESULTS")
    print("="*72)
    header = f"{'Policy':<16} {'Mean':>8} {'SE':>6} {'CVaR95':>8} {'CVaR99':>8} {'Avg Ord':>8}"
    print(header)
    print("-" * 72)

    tex_lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Policy & Mean Return & SE & CVaR$_{95}$ & CVaR$_{99}$ & Avg.\ Order \\",
        r"\hline",
    ]

    for name in policy_names:
        seed_means = np.array([r.mean() for r in returns[name]])
        seed_cvar95 = np.array([
            np.mean(np.sort(r)[:int(0.05 * len(r))]) for r in returns[name]
        ])
        seed_cvar99 = np.array([
            np.mean(np.sort(r)[:int(0.01 * len(r))]) for r in returns[name]
        ])
        seed_avg_ord = np.array([o.mean() for o in avg_orders[name]])

        mean_val = seed_means.mean()
        se_val = seed_means.std() / np.sqrt(n_seeds)
        cvar95_val = seed_cvar95.mean()
        cvar99_val = seed_cvar99.mean()
        avg_ord_val = seed_avg_ord.mean()

        print(f"{name:<16} {mean_val:>8.1f} {se_val:>6.1f} "
              f"{cvar95_val:>8.1f} {cvar99_val:>8.1f} {avg_ord_val:>8.2f}")
        tex_lines.append(
            f"{name} & {mean_val:.1f} & {se_val:.1f} & "
            f"{cvar95_val:.1f} & {cvar99_val:.1f} & {avg_ord_val:.2f} \\\\"
        )

    tex_lines += [r"\hline", r"\end{tabular}"]
    table_path = os.path.join(OUTPUT_DIR, 'risk_sensitive_inventory_table.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f"\nTable: {table_path}")

    # ── Figure 1: Return CDFs ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name in policy_names:
        all_rets = np.concatenate(returns[name])
        sorted_rets = np.sort(all_rets)
        cdf = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets)
        ax.plot(sorted_rets, cdf, label=name,
                color=POLICY_COLORS[name], linestyle=POLICY_STYLES[name],
                linewidth=1.5)
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('CDF')
    ax.legend(loc='lower right', frameon=True)
    ax.set_title('')
    fig1_path = os.path.join(OUTPUT_DIR, 'risk_sensitive_inventory_returns.png')
    fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 1: {fig1_path}")

    # ── Figure 2: Ordering policy ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    inv_levels = np.arange(ENV_PARAMS['S_MAX'] + 1)
    for name in policy_names:
        mean_orders = np.mean(avg_orders[name], axis=0)
        se_orders = np.std(avg_orders[name], axis=0) / np.sqrt(n_seeds)
        ax.plot(inv_levels, mean_orders, label=name,
                color=POLICY_COLORS[name], linestyle=POLICY_STYLES[name],
                linewidth=1.5)
        ax.fill_between(inv_levels, mean_orders - se_orders,
                        mean_orders + se_orders,
                        color=POLICY_COLORS[name], alpha=0.15)
    ax.set_xlabel('Inventory Level')
    ax.set_ylabel('Average Order Quantity')
    ax.legend(loc='upper right', frameon=True)
    ax.set_title('')
    fig2_path = os.path.join(OUTPUT_DIR, 'risk_sensitive_inventory_policy.png')
    fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 2: {fig2_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if data is None:
            print("No cached data found. Run without --plots-only first.")
            sys.exit(1)
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
