"""
Consumption-Savings Under Model Mismatch
Chapter 13: Quantile, Robust and Constrained RL

Standard DP vs robust DP (Hansen-Sargent exponential tilting) vs oracle DP,
plus model-free Q-learning and robust Q-learning counterparts.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, FIG_SINGLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()
import matplotlib.pyplot as plt

SCRIPT_NAME = 'robust_consumption_savings'
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.dirname(__file__)

# -- Environment Configuration ------------------------------------------------

W_MAX = 30
Y_VALUES = [1, 2, 3, 4, 5]
R = 1.02
GAMMA = 0.95
SIGMA = 2.0

NOMINAL_INCOME = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
PERTURBED_INCOME = np.array([0.30, 0.30, 0.20, 0.10, 0.10])
THETA_VALUES = [5.0, 2.0]

N_EVAL = 5000
EVAL_LEN = 100

# -- Q-Learning Configuration ------------------------------------------------

QL_EPISODES = 100_000
QL_HORIZON = 100
QL_ALPHA_C = 100.0       # visit-count LR: alpha = C / (C + N(s,a))
QL_EPS_START = 1.0
QL_EPS_END = 0.05
QL_EPS_DECAY = 0.99998
N_SEEDS = 1

# -- Per-Component Configs ----------------------------------------------------

ENV_PARAMS = {
    'W_MAX': W_MAX, 'Y_VALUES': Y_VALUES, 'R': R, 'GAMMA': GAMMA,
    'SIGMA': SIGMA, 'nominal': NOMINAL_INCOME.tolist(),
    'perturbed': PERTURBED_INCOME.tolist(), 'version': 5,
}

SHARED_CONFIG = {
    **ENV_PARAMS,
    'thetas': THETA_VALUES,
    'n_eval': N_EVAL, 'eval_len': EVAL_LEN,
}

QL_CONFIG = {
    **SHARED_CONFIG,
    'ql_episodes': QL_EPISODES, 'ql_horizon': QL_HORIZON,
    'ql_alpha_c': QL_ALPHA_C, 'ql_eps_start': QL_EPS_START,
    'ql_eps_end': QL_EPS_END, 'ql_eps_decay': QL_EPS_DECAY,
    'n_seeds': N_SEEDS,
}

ROBUST_QL5_CONFIG = {**QL_CONFIG, 'robust_theta': 5.0}
ROBUST_QL2_CONFIG = {**QL_CONFIG, 'robust_theta': 2.0}


def crra(c, sigma=SIGMA):
    c = max(c, 1e-10)
    return c ** (1.0 - sigma) / (1.0 - sigma) if sigma != 1 else np.log(c)


# -- Value Iteration ----------------------------------------------------------

def standard_vi(income_probs, tol=1e-10, max_iter=5000):
    V = np.zeros(W_MAX + 1)
    pi = np.zeros(W_MAX + 1, dtype=int)
    for it in range(max_iter):
        V_old = V.copy()
        for w in range(W_MAX + 1):
            best = -np.inf
            for c in range(w + 1):
                s = w - c
                ev = sum(income_probs[i] * V_old[min(int(round(R * s)) + y, W_MAX)]
                         for i, y in enumerate(Y_VALUES))
                val = crra(c) + GAMMA * ev
                if val > best:
                    best, pi[w] = val, c
            V[w] = best
        if np.max(np.abs(V - V_old)) < tol:
            return V, pi, it + 1
    return V, pi, max_iter


def robust_vi(income_probs, theta, tol=1e-10, max_iter=5000):
    V = np.zeros(W_MAX + 1)
    pi = np.zeros(W_MAX + 1, dtype=int)
    for it in range(max_iter):
        V_old = V.copy()
        for w in range(W_MAX + 1):
            best = -np.inf
            for c in range(w + 1):
                s = w - c
                cont = np.array([V_old[min(int(round(R * s)) + y, W_MAX)]
                                 for y in Y_VALUES])
                lw = -GAMMA * cont / theta
                lw -= lw.max()
                wts = income_probs * np.exp(lw)
                q = wts / wts.sum()
                val = crra(c) + GAMMA * q.dot(cont)
                if val > best:
                    best, pi[w] = val, c
            V[w] = best
        if np.max(np.abs(V - V_old)) < tol:
            return V, pi, it + 1
    return V, pi, max_iter


# -- Evaluate -----------------------------------------------------------------

def evaluate(pi, income_probs, n_ep=N_EVAL, ep_len=EVAL_LEN, seed=42):
    rng = np.random.RandomState(seed)
    total = 0.0
    for _ in range(n_ep):
        w, disc, g = W_MAX // 2, 1.0, 0.0
        for _ in range(ep_len):
            c = min(pi[w], w)
            g += disc * crra(c)
            disc *= GAMMA
            w = min(int(round(R * (w - c))) + rng.choice(Y_VALUES, p=income_probs), W_MAX)
        total += g
    return total / n_ep


# -- Q-Learning ---------------------------------------------------------------

def train_q_learning(seed):
    rng = np.random.RandomState(seed)
    Q = np.zeros((W_MAX + 1, W_MAX + 1))
    counts = np.zeros((W_MAX + 1, W_MAX + 1))
    epsilon = QL_EPS_START

    for ep in range(QL_EPISODES):
        w = rng.randint(0, W_MAX + 1)
        for t in range(QL_HORIZON):
            # epsilon-greedy with action masking
            if rng.random() < epsilon:
                c = rng.randint(0, w + 1)
            else:
                c = int(np.argmax(Q[w, :w + 1]))

            reward = crra(c)
            y = rng.choice(Y_VALUES, p=NOMINAL_INCOME)
            w_next = min(int(round(R * (w - c))) + y, W_MAX)

            counts[w, c] += 1
            alpha = QL_ALPHA_C / (QL_ALPHA_C + counts[w, c])
            best_next = np.max(Q[w_next, :w_next + 1]) if w_next > 0 else Q[0, 0]
            td_target = reward + GAMMA * best_next
            Q[w, c] += alpha * (td_target - Q[w, c])

            w = w_next
        epsilon = max(QL_EPS_END, epsilon * QL_EPS_DECAY)

    pi = np.zeros(W_MAX + 1, dtype=int)
    for w in range(W_MAX + 1):
        pi[w] = int(np.argmax(Q[w, :w + 1]))
    return pi


def train_robust_q_learning(seed, theta):
    rng = np.random.RandomState(seed)
    Q = np.zeros((W_MAX + 1, W_MAX + 1))
    counts = np.zeros((W_MAX + 1, W_MAX + 1))
    epsilon = QL_EPS_START

    for ep in range(QL_EPISODES):
        w = rng.randint(0, W_MAX + 1)
        for t in range(QL_HORIZON):
            if rng.random() < epsilon:
                c = rng.randint(0, w + 1)
            else:
                c = int(np.argmax(Q[w, :w + 1]))

            reward = crra(c)
            y = rng.choice(Y_VALUES, p=NOMINAL_INCOME)
            w_next = min(int(round(R * (w - c))) + y, W_MAX)

            # robust TD target: worst-case over KL ball via exponential tilting
            s = w - c
            cont = np.array([
                np.max(Q[min(int(round(R * s)) + yy, W_MAX),
                         :min(int(round(R * s)) + yy, W_MAX) + 1])
                for yy in Y_VALUES
            ])
            lw = -GAMMA * cont / theta
            lw -= lw.max()
            wts = NOMINAL_INCOME * np.exp(lw)
            q_worst = wts / wts.sum()
            robust_ev = q_worst.dot(cont)

            counts[w, c] += 1
            alpha = QL_ALPHA_C / (QL_ALPHA_C + counts[w, c])
            td_target = reward + GAMMA * robust_ev
            Q[w, c] += alpha * (td_target - Q[w, c])

            w = w_next
        epsilon = max(QL_EPS_END, epsilon * QL_EPS_DECAY)

    pi = np.zeros(W_MAX + 1, dtype=int)
    for w in range(W_MAX + 1):
        pi[w] = int(np.argmax(Q[w, :w + 1]))
    return pi


# -- Compute Components -------------------------------------------------------

def compute_shared():
    print("=" * 60)
    print("Consumption-Savings Under Model Mismatch")
    print("=" * 60)
    print(f"\nW_MAX={W_MAX}, R={R}, gamma={GAMMA}, sigma={SIGMA}")
    print(f"Nominal:   {NOMINAL_INCOME}")
    print(f"Perturbed: {PERTURBED_INCOME}")
    print(f"Eval: {N_EVAL} episodes x {EVAL_LEN} steps\n")

    print("--- DP Policies ---")
    V_nom, pi_nom, n1 = standard_vi(NOMINAL_INCOME)
    print(f"Standard DP:       {n1:>4} iters, V(15)={V_nom[15]:.4f}")

    robust = {}
    for th in THETA_VALUES:
        V_r, pi_r, nr = robust_vi(NOMINAL_INCOME, th)
        robust[th] = pi_r
        print(f"Robust (th={th:<4}):   {nr:>4} iters, V(15)={V_r[15]:.4f}")

    V_orc, pi_orc, no = standard_vi(PERTURBED_INCOME)
    print(f"Oracle DP:         {no:>4} iters, V(15)={V_orc[15]:.4f}")

    print(f"\n{'w':>3} {'Std':>5} {'th=5':>5} {'th=2':>5} {'Orc':>5}")
    for w in range(0, W_MAX + 1, 5):
        print(f"{w:>3} {pi_nom[w]:>5} {robust[5.0][w]:>5} {robust[2.0][w]:>5} {pi_orc[w]:>5}")

    dp_methods = {'Standard DP': pi_nom}
    for th in THETA_VALUES:
        dp_methods[f'Robust DP (th={th})'] = robust[th]
    dp_methods['Oracle'] = pi_orc

    print(f"\n--- DP Evaluation ---")
    dp_results = {}
    for name, pi in dp_methods.items():
        nom = evaluate(pi, NOMINAL_INCOME)
        pert = evaluate(pi, PERTURBED_INCOME)
        pct = 100 * (pert - nom) / abs(nom)
        dp_results[name] = {'nom': nom, 'pert': pert, 'pct': pct}
        print(f"  {name:<25} nom={nom:.3f}  pert={pert:.3f}  delta={pct:.1f}%")

    return {
        'dp_results': dp_results,
        'dp_policies': {n: pi.tolist() for n, pi in dp_methods.items()},
    }


def compute_q_learning(shared):
    print(f"\n--- Q-Learning ({N_SEEDS} seeds, {QL_EPISODES} episodes) ---")
    all_results = []
    for seed in range(N_SEEDS):
        pi = train_q_learning(seed)
        nom = evaluate(pi, NOMINAL_INCOME, seed=seed + 1000)
        pert = evaluate(pi, PERTURBED_INCOME, seed=seed + 1000)
        pct = 100 * (pert - nom) / abs(nom)
        all_results.append({'policy': pi.tolist(), 'nom': nom, 'pert': pert, 'pct': pct})
        print(f"  Seed {seed}: nom={nom:.3f}, pert={pert:.3f}, delta={pct:.1f}%")

    dp_pi = np.array(shared['dp_policies']['Standard DP'])
    mean_pi = np.mean([np.array(r['policy']) for r in all_results], axis=0)
    modal_pi = np.round(mean_pi).astype(int)
    max_dev = int(np.max(np.abs(modal_pi - dp_pi)))
    print(f"  Max policy deviation from Standard DP: {max_dev}")
    return {'seed_results': all_results}


def compute_robust_ql_5(shared):
    print(f"\n--- Robust Q-Learning theta=5 ({N_SEEDS} seeds, {QL_EPISODES} episodes) ---")
    all_results = []
    for seed in range(N_SEEDS):
        pi = train_robust_q_learning(seed, theta=5.0)
        nom = evaluate(pi, NOMINAL_INCOME, seed=seed + 1000)
        pert = evaluate(pi, PERTURBED_INCOME, seed=seed + 1000)
        pct = 100 * (pert - nom) / abs(nom)
        all_results.append({'policy': pi.tolist(), 'nom': nom, 'pert': pert, 'pct': pct})
        print(f"  Seed {seed}: nom={nom:.3f}, pert={pert:.3f}, delta={pct:.1f}%")

    dp_pi = np.array(shared['dp_policies']['Robust DP (th=5.0)'])
    mean_pi = np.mean([np.array(r['policy']) for r in all_results], axis=0)
    modal_pi = np.round(mean_pi).astype(int)
    max_dev = int(np.max(np.abs(modal_pi - dp_pi)))
    print(f"  Max policy deviation from Robust DP (th=5): {max_dev}")
    return {'seed_results': all_results}


def compute_robust_ql_2(shared):
    print(f"\n--- Robust Q-Learning theta=2 ({N_SEEDS} seeds, {QL_EPISODES} episodes) ---")
    all_results = []
    for seed in range(N_SEEDS):
        pi = train_robust_q_learning(seed, theta=2.0)
        nom = evaluate(pi, NOMINAL_INCOME, seed=seed + 1000)
        pert = evaluate(pi, PERTURBED_INCOME, seed=seed + 1000)
        pct = 100 * (pert - nom) / abs(nom)
        all_results.append({'policy': pi.tolist(), 'nom': nom, 'pert': pert, 'pct': pct})
        print(f"  Seed {seed}: nom={nom:.3f}, pert={pert:.3f}, delta={pct:.1f}%")

    dp_pi = np.array(shared['dp_policies']['Robust DP (th=2.0)'])
    mean_pi = np.mean([np.array(r['policy']) for r in all_results], axis=0)
    modal_pi = np.round(mean_pi).astype(int)
    max_dev = int(np.max(np.abs(modal_pi - dp_pi)))
    print(f"  Max policy deviation from Robust DP (th=2): {max_dev}")
    return {'seed_results': all_results}


# -- Main Compute -------------------------------------------------------------

def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'shared', SHARED_CONFIG,
                              compute_shared, force=('shared' in force))
    ql = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'Q-learning', QL_CONFIG,
                          compute_q_learning, shared,
                          force=('Q-learning' in force or 'shared' in force))
    rql5 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'Robust-QL-5', ROBUST_QL5_CONFIG,
                            compute_robust_ql_5, shared,
                            force=('Robust-QL-5' in force or 'shared' in force))
    rql2 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'Robust-QL-2', ROBUST_QL2_CONFIG,
                            compute_robust_ql_2, shared,
                            force=('Robust-QL-2' in force or 'shared' in force))
    return {'shared': shared, 'Q-learning': ql, 'Robust-QL-5': rql5, 'Robust-QL-2': rql2}


# -- Outputs ------------------------------------------------------------------

METHOD_ORDER = [
    'Standard DP', 'Q-learning',
    'Robust DP (th=5.0)', 'Robust Q-learning (th=5)',
    'Robust DP (th=2.0)', 'Robust Q-learning (th=2)',
    'Oracle',
]

MCOLORS = {
    'Standard DP':              COLORS['blue'],
    'Q-learning':               ALGO_COLORS['Q-Learning'],
    'Robust DP (th=5.0)':       COLORS['orange'],
    'Robust Q-learning (th=5)': COLORS['purple'],
    'Robust DP (th=2.0)':       COLORS['red'],
    'Robust Q-learning (th=2)': COLORS['green'],
    'Oracle':                   COLORS['black'],
}

MSTYLES = {
    'Standard DP': '-', 'Q-learning': '--',
    'Robust DP (th=5.0)': '-', 'Robust Q-learning (th=5)': '--',
    'Robust DP (th=2.0)': '-', 'Robust Q-learning (th=2)': '--',
    'Oracle': ':',
}


def _aggregate_ql(seed_results):
    policies = np.array([r['policy'] for r in seed_results])
    noms = np.array([r['nom'] for r in seed_results])
    perts = np.array([r['pert'] for r in seed_results])
    pcts = np.array([r['pct'] for r in seed_results])
    n = len(seed_results)
    if n > 1:
        se_pol = np.std(policies, axis=0, ddof=1) / np.sqrt(n)
        se_nom = np.std(noms, ddof=1) / np.sqrt(n)
        se_pert = np.std(perts, ddof=1) / np.sqrt(n)
        se_pct = np.std(pcts, ddof=1) / np.sqrt(n)
    else:
        se_pol = np.zeros(policies.shape[1])
        se_nom = se_pert = se_pct = 0.0
    return {
        'mean_policy': np.mean(policies, axis=0), 'se_policy': se_pol,
        'nom': np.mean(noms), 'nom_se': se_nom,
        'pert': np.mean(perts), 'pert_se': se_pert,
        'pct': np.mean(pcts), 'pct_se': se_pct,
    }


def generate_outputs(data):
    shared = data['shared']
    dp_results = shared['dp_results']
    dp_policies = {n: np.array(p) for n, p in shared['dp_policies'].items()}

    ql_agg = _aggregate_ql(data['Q-learning']['seed_results'])
    rql5_agg = _aggregate_ql(data['Robust-QL-5']['seed_results'])
    rql2_agg = _aggregate_ql(data['Robust-QL-2']['seed_results'])

    # consolidated results for table
    all_results = {}
    all_policies = {}
    all_se_policies = {}

    for name in ['Standard DP', 'Robust DP (th=5.0)', 'Robust DP (th=2.0)', 'Oracle']:
        r = dp_results[name]
        all_results[name] = {'nom': r['nom'], 'pert': r['pert'], 'pct': r['pct'],
                             'nom_se': None, 'pert_se': None, 'pct_se': None}
        all_policies[name] = dp_policies[name]
        all_se_policies[name] = None

    for label, agg in [('Q-learning', ql_agg),
                       ('Robust Q-learning (th=5)', rql5_agg),
                       ('Robust Q-learning (th=2)', rql2_agg)]:
        all_results[label] = agg
        all_policies[label] = agg['mean_policy']
        all_se_policies[label] = agg['se_policy']

    # -- Figure ---------------------------------------------------------------
    ws = np.arange(W_MAX + 1)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name in METHOD_ORDER:
        pi = all_policies[name]
        se = all_se_policies[name]
        color = MCOLORS[name]
        ls = MSTYLES[name]
        lbl = name.replace('th=', r'$\theta$=')
        ax.plot(ws, pi, label=lbl, color=color, linestyle=ls, linewidth=1.5)
        if se is not None:
            ax.fill_between(ws, pi - se, pi + se, color=color, alpha=0.15)
    ax.set_xlabel('Wealth $w$')
    ax.set_ylabel('Consumption $c(w)$')
    ax.legend(loc='upper left', frameon=True, fontsize=7)
    fig_path = os.path.join(OUTPUT_DIR, 'robust_consumption_savings_policy.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure: {fig_path}")

    # -- Table ----------------------------------------------------------------
    tex = [r"\begin{tabular}{lrrr}", r"\hline",
           r"Method & Nominal & Perturbed & Degradation (\%) \\", r"\hline"]
    for name in METHOD_ORDER:
        r = all_results[name]
        if r['nom_se'] is not None and r['nom_se'] > 0:
            nom_str = f"${r['nom']:.2f} \\pm {r['nom_se']:.2f}$"
            pert_str = f"${r['pert']:.2f} \\pm {r['pert_se']:.2f}$"
            pct_str = f"${r['pct']:.1f}$"
        else:
            nom_str = f"${r['nom']:.2f}$"
            pert_str = f"${r['pert']:.2f}$"
            pct_str = f"${r['pct']:.1f}$"
        label = name.replace('th=', r'$\theta$=')
        tex.append(f"{label} & {nom_str} & {pert_str} & {pct_str} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    tab_path = os.path.join(OUTPUT_DIR, 'robust_consumption_savings_table.tex')
    with open(tab_path, 'w') as f:
        f.write('\n'.join(tex))
    print(f"Table: {tab_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    if args.plots_only:
        data = compute_data()
    else:
        data = compute_data(force=force)

    if not args.data_only:
        generate_outputs(data)
