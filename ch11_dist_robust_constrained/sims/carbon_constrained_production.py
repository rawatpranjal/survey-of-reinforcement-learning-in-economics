"""
Carbon-Constrained Production via Lagrangian Dual Ascent
Chapter 13: Quantile, Robust and Constrained RL

Factory maximizes profit subject to carbon budget. The Lagrange multiplier
(carbon tax) converges to the analytical shadow price from the LP dual.
"""

import sys
import os
import argparse
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, BENCH_STYLE, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args

apply_style()
import matplotlib.pyplot as plt

SCRIPT_NAME = 'carbon_constrained_production'
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.dirname(__file__)

# ── Configuration ────────────────────────────────────────────────────────────

ENV_PARAMS = {
    'I_MAX': 8,
    'P_MAX': 3,
    'GAMMA': 0.95,
    'PRICE': 10.0,
    'HOLD_COST': 1.0,
    'DIRTY_COST': 2.0,
    'CLEAN_COST': 5.0,
    'DIRTY_EMISSION': 3.0,
    'CLEAN_EMISSION': 0.5,
    'DEMAND_LOW': [0.30, 0.30, 0.20, 0.15, 0.05],
    'DEMAND_HIGH': [0.05, 0.10, 0.20, 0.30, 0.35],
    'REGIME_TRANS': [[0.8, 0.2], [0.3, 0.7]],
}

QL_CONFIG = {
    'N_EPISODES': 30_000,
    'HORIZON': 100,
    'LR': 0.15,
    'LR_DECAY': 0.99999,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 15_000,
    'DUAL_LR': 0.003,
    'DUAL_UPDATE_FREQ': 1000,
    'EVAL_FREQ': 500,
    'N_EVAL_EPISODES': 300,
}

EVAL_CONFIG = {
    'N_FINAL_EVAL': 5000,
}

CONFIG = {**ENV_PARAMS, **QL_CONFIG, **EVAL_CONFIG, 'version': 12}

# ── Environment ──────────────────────────────────────────────────────────────

class CarbonFactory:
    def __init__(self, params):
        self.I_MAX = params['I_MAX']
        self.P_MAX = params['P_MAX']
        self.gamma = params['GAMMA']
        self.price = params['PRICE']
        self.hold_cost = params['HOLD_COST']
        self.dirty_cost = params['DIRTY_COST']
        self.clean_cost = params['CLEAN_COST']
        self.dirty_emission = params['DIRTY_EMISSION']
        self.clean_emission = params['CLEAN_EMISSION']
        self.demand_pmfs = [
            np.array(params['DEMAND_LOW']),
            np.array(params['DEMAND_HIGH']),
        ]
        self.regime_trans = np.array(params['REGIME_TRANS'])
        self.n_demand = len(params['DEMAND_LOW'])
        self.n_inv = self.I_MAX + 1
        self.n_regime = 2
        self.n_states = self.n_inv * self.n_regime
        self.n_prod = self.P_MAX + 1
        self.n_energy = 2
        self.n_actions = self.n_prod * self.n_energy
        self.state = None

    def state_to_index(self, inv, regime):
        return inv * self.n_regime + regime

    def index_to_state(self, idx):
        return idx // self.n_regime, idx % self.n_regime

    def action_to_tuple(self, a_idx):
        return a_idx // self.n_energy, a_idx % self.n_energy

    def reset(self, rng):
        self.state = (rng.randint(0, self.n_inv), rng.randint(0, self.n_regime))
        return self.state_to_index(*self.state)

    def step(self, a_idx, rng):
        inv, regime = self.state
        prod, energy = self.action_to_tuple(a_idx)
        inv_after = min(inv + prod, self.I_MAX)
        demand = rng.choice(self.n_demand, p=self.demand_pmfs[regime])
        sales = min(inv_after, demand)
        inv_next = inv_after - sales
        prod_cost = (self.dirty_cost if energy == 0 else self.clean_cost) * prod
        reward = self.price * sales - prod_cost - self.hold_cost * inv_next
        emission = (self.dirty_emission if energy == 0 else self.clean_emission) * prod
        regime_next = rng.choice(self.n_regime, p=self.regime_trans[regime])
        self.state = (inv_next, regime_next)
        return self.state_to_index(*self.state), reward, emission

    def build_matrices(self):
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))
        C = np.zeros((nS, nA))
        for s in range(nS):
            inv, regime = self.index_to_state(s)
            for a in range(nA):
                prod, energy = self.action_to_tuple(a)
                inv_after = min(inv + prod, self.I_MAX)
                prod_cost = (self.dirty_cost if energy == 0 else self.clean_cost) * prod
                C[s, a] = (self.dirty_emission if energy == 0 else self.clean_emission) * prod
                exp_reward = 0.0
                for d in range(self.n_demand):
                    p_d = self.demand_pmfs[regime][d]
                    sales = min(inv_after, d)
                    inv_next = inv_after - sales
                    r = self.price * sales - prod_cost - self.hold_cost * inv_next
                    exp_reward += p_d * r
                    for rg_next in range(self.n_regime):
                        p_rg = self.regime_trans[regime][rg_next]
                        s_next = self.state_to_index(inv_next, rg_next)
                        P[s, a, s_next] += p_d * p_rg
                R[s, a] = exp_reward
        return P, R, C


# ── Exact methods ────────────────────────────────────────────────────────────

def value_iteration(P, R, gamma, max_iter=5000, tol=1e-10):
    nS = R.shape[0]
    V = np.zeros(nS)
    for _ in range(max_iter):
        Q = R + gamma * np.einsum('sab,b->sa', P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V, Q, Q.argmax(axis=1)


def eval_policy_exact(P, R, C, pi, gamma, alpha):
    nS = R.shape[0]
    R_pi = np.array([R[s, pi[s]] for s in range(nS)])
    C_pi = np.array([C[s, pi[s]] for s in range(nS)])
    P_pi = np.array([P[s, pi[s]] for s in range(nS)])
    V_r = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    V_c = np.linalg.solve(np.eye(nS) - gamma * P_pi, C_pi)
    return alpha @ V_r, alpha @ V_c


def eval_stochastic_exact(P, R, C, policy_probs, gamma, alpha):
    nS, nA = R.shape
    R_pi = np.sum(policy_probs * R, axis=1)
    C_pi = np.sum(policy_probs * C, axis=1)
    P_pi = np.einsum('sa,sab->sb', policy_probs, P)
    V_r = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    V_c = np.linalg.solve(np.eye(nS) - gamma * P_pi, C_pi)
    return alpha @ V_r, alpha @ V_c


# ── LP Oracle ────────────────────────────────────────────────────────────────

def solve_cmdp_lp(env, carbon_budget):
    P, R, C = env.build_matrices()
    nS, nA = env.n_states, env.n_actions
    gamma = env.gamma
    alpha = np.ones(nS) / nS

    c_obj = -R.ravel()
    A_eq = np.zeros((nS, nS * nA))
    b_eq = alpha.copy()
    for s in range(nS):
        for a in range(nA):
            A_eq[s, s * nA + a] += 1.0
        for sp in range(nS):
            for ap in range(nA):
                A_eq[s, sp * nA + ap] -= gamma * P[sp, ap, s]

    A_ub = C.ravel().reshape(1, -1)
    b_ub = np.array([carbon_budget])
    bounds = [(0, None)] * (nS * nA)

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    assert result.success, f"LP failed: {result.message}"

    nu = result.x.reshape(nS, nA)
    nu_s = nu.sum(axis=1)
    policy_probs = np.zeros((nS, nA))
    for s in range(nS):
        if nu_s[s] > 1e-12:
            policy_probs[s] = nu[s] / nu_s[s]
        else:
            policy_probs[s, 0] = 1.0

    lambda_star = 0.0
    if hasattr(result, 'ineqlin') and result.ineqlin is not None:
        lambda_star = abs(float(result.ineqlin.marginals[0]))

    return {
        'policy_probs': policy_probs,
        'opt_return': -result.fun,
        'opt_cost': np.sum(C * nu),
        'lambda_star': lambda_star,
    }


# ── Q-Learning ───────────────────────────────────────────────────────────────

def run_q_learning(env, config, seed, carbon_budget=None):
    """Tabular Q-learning. If carbon_budget given, uses Lagrangian dual ascent
    with a single Q-table for the Lagrangian reward r - lambda*c."""
    rng = np.random.RandomState(seed)
    nS, nA = env.n_states, env.n_actions
    gamma = env.gamma
    horizon = config['HORIZON']
    constrained = carbon_budget is not None

    Q = np.zeros((nS, nA))  # Q for r (unconstrained) or r-lam*c (constrained)
    lr = config['LR']
    lam = 0.0

    eval_eps_list = []
    eval_returns = []
    eval_costs = []
    lambda_traj = [] if constrained else None
    cost_buffer = []

    for ep in range(config['N_EPISODES']):
        frac = min(1.0, ep / config['EPS_DECAY'])
        eps = config['EPS_START'] + frac * (config['EPS_END'] - config['EPS_START'])

        s = env.reset(rng)
        ep_cost = 0.0
        disc = 1.0

        for t in range(horizon):
            if rng.random() < eps:
                a = rng.randint(nA)
            else:
                a = np.argmax(Q[s])

            s_next, r, c = env.step(a, rng)

            # Lagrangian reward
            r_lag = r - lam * c if constrained else r
            Q[s, a] += lr * (r_lag + gamma * Q[s_next].max() - Q[s, a])

            ep_cost += disc * c
            disc *= gamma
            s = s_next

        cost_buffer.append(ep_cost)
        lr *= config['LR_DECAY']

        # Dual update (every K episodes)
        if constrained and (ep + 1) % config['DUAL_UPDATE_FREQ'] == 0 and ep > 0:
            avg_cost = np.mean(cost_buffer[-config['DUAL_UPDATE_FREQ']:])
            trunc = 1.0 - gamma ** horizon
            avg_cost_inf = avg_cost / trunc
            old_lam = lam
            lam = max(0.0, lam + config['DUAL_LR'] * (avg_cost_inf - carbon_budget))
            # Q adapts gradually through continued learning

        # Eval
        if (ep + 1) % config['EVAL_FREQ'] == 0:
            pi = np.argmax(Q, axis=1)
            eval_rng = np.random.RandomState(42)
            ret, cost = _eval_det(env, pi, config['N_EVAL_EPISODES'], horizon, eval_rng)
            eval_eps_list.append(ep + 1)
            eval_returns.append(ret)
            eval_costs.append(cost)
            if constrained:
                lambda_traj.append(lam)

        if (ep + 1) % 10_000 == 0:
            pi = np.argmax(Q, axis=1)
            eval_rng = np.random.RandomState(42)
            ret, cost = _eval_det(env, pi, 500, horizon, eval_rng)
            tag = "QL-Lag" if constrained else "QL"
            lam_s = f", lam={lam:.3f}" if constrained else ""
            print(f"    {tag} ep {ep+1:>6d}: ret={ret:.1f}, cost={cost:.2f}"
                  f"{lam_s}", flush=True)

    # Final
    pi_final = np.argmax(Q, axis=1)
    eval_rng = np.random.RandomState(99)
    final_ret, final_cost = _eval_det(env, pi_final, EVAL_CONFIG['N_FINAL_EVAL'], horizon, eval_rng)

    result = {
        'eval_episodes': eval_eps_list,
        'eval_returns': eval_returns,
        'eval_costs': eval_costs,
        'final_return': final_ret,
        'final_cost': final_cost,
        'pi': pi_final,
    }
    if constrained:
        result['lambda_trajectory'] = lambda_traj
        result['final_lambda'] = lam
    return result


def _eval_det(env, pi, n_episodes, horizon, rng):
    gamma = env.gamma
    rets, costs = [], []
    for _ in range(n_episodes):
        s = env.reset(rng)
        ep_r, ep_c, disc = 0.0, 0.0, 1.0
        for _ in range(horizon):
            s, r, c = env.step(pi[s], rng)
            ep_r += disc * r
            ep_c += disc * c
            disc *= gamma
        rets.append(ep_r)
        costs.append(ep_c)
    return np.mean(rets), np.mean(costs)


# ── compute_data / generate_outputs ──────────────────────────────────────────

def compute_data(force=False):
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached and not force:
        print("Loaded from cache.")
        return cached

    env = CarbonFactory(ENV_PARAMS)
    P, R, C = env.build_matrices()
    nS = env.n_states
    gamma = env.gamma
    alpha = np.ones(nS) / nS
    print(f"States: {nS}, Actions: {env.n_actions}")

    # Unconstrained DP
    print("\n--- Unconstrained Value Iteration ---")
    _, _, pi_unc = value_iteration(P, R, gamma)
    ret_unc, cost_unc = eval_policy_exact(P, R, C, pi_unc, gamma, alpha)
    print(f"  Return: {ret_unc:.2f}, Cost: {cost_unc:.2f}")

    # Budget = 30% of unconstrained cost
    carbon_budget = 0.3 * cost_unc
    print(f"  Carbon budget d = {carbon_budget:.2f}")

    # Constrained LP
    print("\n--- Constrained LP Oracle ---")
    oracle = solve_cmdp_lp(env, carbon_budget)
    ret_con, cost_con = eval_stochastic_exact(P, R, C, oracle['policy_probs'], gamma, alpha)
    print(f"  LP return: {ret_con:.2f}, LP cost: {cost_con:.2f}")
    print(f"  lambda*:   {oracle['lambda_star']:.4f}")

    # Unconstrained Q-learning
    print("\n--- Unconstrained Q-Learning ---")
    ql = run_q_learning(env, QL_CONFIG, seed=0, carbon_budget=None)
    print(f"  Final return: {ql['final_return']:.2f}, cost: {ql['final_cost']:.2f}")

    # Lagrangian Q-learning
    print("\n--- Lagrangian Q-Learning ---")
    ql_lag = run_q_learning(env, QL_CONFIG, seed=0, carbon_budget=carbon_budget)
    print(f"  Final return: {ql_lag['final_return']:.2f}, cost: {ql_lag['final_cost']:.2f}")
    print(f"  Final lambda: {ql_lag['final_lambda']:.4f}")

    data = {
        'carbon_budget': carbon_budget,
        'ret_unc': ret_unc, 'cost_unc': cost_unc,
        'oracle': oracle,
        'oracle_return': ret_con, 'oracle_cost': cost_con,
        'ql': ql, 'ql_lag': ql_lag,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    budget = data['carbon_budget']
    oracle = data['oracle']
    ql = data['ql']
    lag = data['ql_lag']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    eps = lag['eval_episodes']

    # (a) Lambda trajectory
    ax1.plot(eps, lag['lambda_trajectory'], color=COLORS['green'], label='Lagrangian Q-learning')
    ax1.axhline(oracle['lambda_star'], **BENCH_STYLE,
                label=f"$\\lambda^* = {oracle['lambda_star']:.2f}$")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('$\\lambda$ (carbon tax)')
    ax1.set_title('(a)  Lagrange multiplier convergence')
    ax1.legend()

    # (b) Return over training
    ax2.plot(ql['eval_episodes'], ql['eval_returns'],
             color=COLORS['orange'], label='Unconstrained Q-learning')
    ax2.plot(eps, lag['eval_returns'], color=COLORS['green'], label='Lagrangian Q-learning')
    ax2.axhline(data['oracle_return'], **BENCH_STYLE,
                label=f"Constrained LP = {data['oracle_return']:.0f}")
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean discounted return')
    ax2.set_title('(b)  Return over training')
    ax2.legend()

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'{SCRIPT_NAME}_convergence.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")

    # Table
    sat_ql = "Y" if ql['final_cost'] <= budget else "N"
    sat_lag = "Y" if lag['final_cost'] <= budget else "N"
    lines = [
        r"\begin{tabular}{lrrcc}",
        r"\hline",
        r"Method & Return & Cost & Budget & $\lambda$ \\",
        r"\hline",
        f"LP Oracle & {data['oracle_return']:.1f} & {data['oracle_cost']:.2f} "
        f"& Y & {oracle['lambda_star']:.2f} \\\\",
        f"Unconstrained Q-learning & {ql['final_return']:.1f} & {ql['final_cost']:.2f} "
        f"& {sat_ql} & -- \\\\",
        f"Lagrangian Q-learning & {lag['final_return']:.1f} & {lag['final_cost']:.2f} "
        f"& {sat_lag} & {lag['final_lambda']:.2f} \\\\",
        r"\hline",
        r"\end{tabular}",
    ]
    table_path = os.path.join(OUTPUT_DIR, f'{SCRIPT_NAME}_table.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Table saved: {table_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Carbon budget d = {budget:.2f}")
    print(f"{'Method':<26s} {'Return':>8s} {'Cost':>8s} {'Budget':>7s} {'lambda':>8s}")
    print(f"{'-'*60}")
    print(f"{'LP Oracle':<26s} {data['oracle_return']:>8.1f} {data['oracle_cost']:>8.2f} "
          f"{'Y':>7s} {oracle['lambda_star']:>8.2f}")
    print(f"{'Unconstrained Q-learning':<26s} {ql['final_return']:>8.1f} {ql['final_cost']:>8.2f} "
          f"{sat_ql:>7s} {'--':>8s}")
    print(f"{'Lagrangian Q-learning':<26s} {lag['final_return']:>8.1f} {lag['final_cost']:>8.2f} "
          f"{sat_lag:>7s} {lag['final_lambda']:>8.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carbon-Constrained Production')
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cached data."
    else:
        data = compute_data(force=False)
    if not args.data_only:
        generate_outputs(data)
