"""
brock_mirman_bellman.py
Chapter: The Empirics of Deep RL
Compares Bellman Residual Minimization (BRM) vs Fitted Q-Evaluation (FQE)
on an offline Brock-Mirman dataset to demonstrate Fujimoto (2022) Table 1:
BRM achieves lower Bellman error on the dataset but higher value error on
the full MDP than FQE. An OLS analogue shows tight coupling as a baseline.
"""

import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# ============================================================================
# Parameters
# ============================================================================

ALPHA    = 0.36
BETA     = 0.96
Z_VALS   = np.array([0.9, 1.1])
PI_TRANS = np.array([[0.8, 0.2],
                     [0.2, 0.8]])

N_K = 50
N_Z = 2
N_S = N_K * N_Z   # 100 states
N_A = N_K         # 50 actions (choice of k')

GAMMA = BETA

# Offline dataset
T_OFFLINE = 2_000

# Neural network training
BATCH_SIZE    = 64
LR            = 5e-4
HIDDEN        = 64
TARGET_UPDATE = 500   # FQE target network update (gradient steps)
TRAIN_STEPS   = 50_000
LOG_EVERY     = 500

SEEDS = [42, 123, 777]

# OLS
T_SIM     = 2000
T_TEST    = 500
NOISE_STD = 0.30   # larger noise so R^2 has room to grow

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR  = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'brock_mirman_bellman'
CONFIG = {
    'alpha': ALPHA,
    'beta': BETA,
    'z_vals': Z_VALS.tolist(),
    'pi_trans': PI_TRANS.tolist(),
    'n_k': N_K,
    'n_z': N_Z,
    't_offline': T_OFFLINE,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'hidden': HIDDEN,
    'target_update': TARGET_UPDATE,
    'train_steps': TRAIN_STEPS,
    'log_every': LOG_EVERY,
    'seeds': SEEDS,
    't_sim': T_SIM,
    't_test': T_TEST,
    'noise_std': NOISE_STD,
    'version': 1,
}

# ============================================================================
# MDP setup
# ============================================================================

def build_capital_grid():
    k_ss_high = (ALPHA * BETA * Z_VALS.max()) ** (1.0 / (1.0 - ALPHA))
    return np.logspace(np.log10(0.01), np.log10(1.5 * k_ss_high), N_K)


def build_mdp(k_grid):
    """Return R[s,a] (-inf for infeasible) and P[s,a,s']."""
    R = np.full((N_S, N_A), -np.inf)
    P = np.zeros((N_S, N_A, N_S))
    for ik in range(N_K):
        for iz in range(N_Z):
            s = ik * N_Z + iz
            output = Z_VALS[iz] * k_grid[ik] ** ALPHA
            for ia in range(N_A):
                c = output - k_grid[ia]
                if c > 1e-10:
                    R[s, ia] = np.log(c)
                    for iz2 in range(N_Z):
                        s2 = ia * N_Z + iz2
                        P[s, ia, s2] = PI_TRANS[iz, iz2]
    return R, P


def compute_q_star(R, P, tol=1e-10, max_iter=5000):
    """Value iteration; returns Q*(s,a) and V*(s)."""
    V = np.zeros(N_S)
    feasible = R > -1e30
    for _ in range(max_iter):
        Q = R + GAMMA * np.einsum('san,n->sa', P, V)
        Q_masked = np.where(feasible, Q, -np.inf)
        V_new = np.max(Q_masked, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q_star = R + GAMMA * np.einsum('san,n->sa', P, V)
    Q_star[~feasible] = -np.inf
    return Q_star, V


def build_feature_matrix(k_grid):
    """(N_S, 2) normalized features: [k/k_max, z/z_max]."""
    feats = np.zeros((N_S, 2), dtype=np.float32)
    for ik in range(N_K):
        for iz in range(N_Z):
            s = ik * N_Z + iz
            feats[s] = [k_grid[ik] / k_grid[-1], Z_VALS[iz] / Z_VALS[-1]]
    return feats


def optimal_policy(k_grid):
    """Closed-form k'* = alpha*beta*z*k^alpha, rounded to grid."""
    pi = np.zeros(N_S, dtype=int)
    for ik in range(N_K):
        for iz in range(N_Z):
            s = ik * N_Z + iz
            k_opt = ALPHA * BETA * Z_VALS[iz] * k_grid[ik] ** ALPHA
            pi[s] = int(np.argmin(np.abs(k_grid - k_opt)))
    return pi


# ============================================================================
# Offline dataset from optimal policy
# ============================================================================

def generate_offline_dataset(k_grid, R_shifted, feats, pi_star, seed=0):
    """
    Simulate T_OFFLINE transitions from the optimal policy.
    Returns (sf, a, r, sfn, sn): features at s, action, shifted reward,
    features at s', index of s'.
    """
    np.random.seed(seed)
    ik, iz = N_K // 4, 0   # start near lower quarter of capital grid
    sf_list, a_list, r_list, sfn_list, sn_list = [], [], [], [], []

    for _ in range(T_OFFLINE):
        s = ik * N_Z + iz
        a = pi_star[s]
        r = R_shifted[s, a]
        iz_next = int(np.random.choice(N_Z, p=PI_TRANS[iz]))
        ik_next = a
        s_next  = ik_next * N_Z + iz_next

        sf_list.append(feats[s])
        a_list.append(a)
        r_list.append(float(r))
        sfn_list.append(feats[s_next])
        sn_list.append(s_next)

        ik, iz = ik_next, iz_next

    return (np.array(sf_list,  dtype=np.float32),
            np.array(a_list,   dtype=np.int64),
            np.array(r_list,   dtype=np.float32),
            np.array(sfn_list, dtype=np.float32),
            np.array(sn_list,  dtype=np.int64))


# ============================================================================
# Neural network and utilities
# ============================================================================

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, N_A),
        )
    def forward(self, x):
        return self.net(x)


def make_feas_penalty_tensor(feasible):
    """(N_S, N_A) tensor: 0 for feasible, -1e9 for infeasible."""
    arr = np.where(feasible, 0.0, -1e9).astype(np.float32)
    return torch.tensor(arr)


def bellman_error_on_D(net, D_sf, D_a, D_r, D_sfn, D_sn, feas_penalty):
    """
    Mean squared Bellman error on dataset D using current network on both sides.
    BE = mean( (Q(s,a) - (r + gamma * max_a' Q(s',a')))^2 )
    """
    with torch.no_grad():
        sf_t  = torch.tensor(D_sf)
        a_t   = torch.tensor(D_a)
        r_t   = torch.tensor(D_r)
        sfn_t = torch.tensor(D_sfn)
        sn_t  = D_sn  # numpy array of indices

        Q_sa  = net(sf_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        Q_nxt = net(sfn_t) + feas_penalty[sn_t]
        y     = r_t + GAMMA * Q_nxt.max(dim=1)[0]
        be    = float(((Q_sa - y) ** 2).mean().item())
    return be


def value_error_full_MDP(net, feats_t, Q_star_shifted, feasible):
    """
    Mean absolute value error over all feasible (s,a).
    VE = mean( |Q(s,a) - Q*(s,a)| ) over feasible pairs.
    """
    with torch.no_grad():
        q_theta = net(feats_t).numpy()
    return float(np.abs(q_theta[feasible] - Q_star_shifted[feasible]).mean())


def policy_agreement(net, feats_t, pi_star, feasible):
    """Fraction of states where greedy(Q_theta) == pi_star."""
    with torch.no_grad():
        q_theta = net(feats_t).numpy()
    pi_theta = np.argmax(np.where(feasible, q_theta, -np.inf), axis=1)
    return float(np.mean(pi_theta == pi_star))


# ============================================================================
# BRM training: gradients flow through both Q(s,a) and Q(s',a')
# ============================================================================

def run_brm(seed, D_sf, D_a, D_r, D_sfn, D_sn,
            feats_t, Q_star_shifted, feasible, pi_star, feas_penalty):
    """
    Bellman Residual Minimization.
    Loss: mean( (Q(s,a) - (r + gamma * max_a' Q(s',a')))^2 )
    Both Q(s,a) and Q(s',a') use the current network; no target network.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = QNet()
    opt = optim.Adam(net.parameters(), lr=LR)
    N_D = len(D_sf)

    sf_all  = torch.tensor(D_sf)
    a_all   = torch.tensor(D_a)
    r_all   = torch.tensor(D_r)
    sfn_all = torch.tensor(D_sfn)

    steps_log, be_log, ve_log, pa_log = [], [], [], []

    for step in range(1, TRAIN_STEPS + 1):
        idx   = np.random.randint(0, N_D, BATCH_SIZE)
        sf_t  = sf_all[idx]
        a_t   = a_all[idx]
        r_t   = r_all[idx]
        sfn_t = sfn_all[idx]
        sni_b = D_sn[idx]

        # BRM: current network used on both sides — no torch.no_grad() on target
        Q_pred    = net(sf_t)
        Q_pred_sa = Q_pred.gather(1, a_t.unsqueeze(1)).squeeze(1)

        Q_next         = net(sfn_t)                      # gradients flow here
        Q_next_masked  = Q_next + feas_penalty[sni_b]    # -1e9 for infeasible
        Q_next_max     = Q_next_masked.max(dim=1)[0]

        y    = r_t + GAMMA * Q_next_max   # target depends on current network
        loss = ((Q_pred_sa - y) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            be = bellman_error_on_D(net, D_sf, D_a, D_r, D_sfn, D_sn, feas_penalty)
            ve = value_error_full_MDP(net, feats_t, Q_star_shifted, feasible)
            pa = policy_agreement(net, feats_t, pi_star, feasible)
            steps_log.append(step)
            be_log.append(be)
            ve_log.append(ve)
            pa_log.append(pa)

    return (np.array(steps_log), np.array(be_log),
            np.array(ve_log),    np.array(pa_log))


# ============================================================================
# FQE training: frozen target network for Q(s',a')
# ============================================================================

def run_fqe(seed, D_sf, D_a, D_r, D_sfn, D_sn,
            feats_t, Q_star_shifted, feasible, pi_star, feas_penalty):
    """
    Fitted Q-Evaluation.
    Loss: mean( (Q(s,a) - (r + gamma * max_a' Q_target(s',a')))^2 )
    Target network is a frozen copy of the current network, updated every
    TARGET_UPDATE gradient steps. Gradients flow only through Q(s,a).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    net        = QNet()
    target_net = QNet()
    target_net.load_state_dict(net.state_dict())
    opt  = optim.Adam(net.parameters(), lr=LR)
    N_D  = len(D_sf)

    sf_all  = torch.tensor(D_sf)
    a_all   = torch.tensor(D_a)
    r_all   = torch.tensor(D_r)
    sfn_all = torch.tensor(D_sfn)

    steps_log, be_log, ve_log, pa_log = [], [], [], []

    for step in range(1, TRAIN_STEPS + 1):
        idx   = np.random.randint(0, N_D, BATCH_SIZE)
        sf_t  = sf_all[idx]
        a_t   = a_all[idx]
        r_t   = r_all[idx]
        sfn_t = sfn_all[idx]
        sni_b = D_sn[idx]

        # FQE: frozen target network provides Q(s',a')
        Q_pred    = net(sf_t)
        Q_pred_sa = Q_pred.gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            Q_next        = target_net(sfn_t)
            Q_next_masked = Q_next + feas_penalty[sni_b]
            Q_next_max    = Q_next_masked.max(dim=1)[0]
            y             = r_t + GAMMA * Q_next_max

        loss = ((Q_pred_sa - y) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(net.state_dict())

        if step % LOG_EVERY == 0:
            be = bellman_error_on_D(net, D_sf, D_a, D_r, D_sfn, D_sn, feas_penalty)
            ve = value_error_full_MDP(net, feats_t, Q_star_shifted, feasible)
            pa = policy_agreement(net, feats_t, pi_star, feasible)
            steps_log.append(step)
            be_log.append(be)
            ve_log.append(ve)
            pa_log.append(pa)

    return (np.array(steps_log), np.array(be_log),
            np.array(ve_log),    np.array(pa_log))


# ============================================================================
# OLS expanding window
# ============================================================================

def run_ols(k_grid, seed=42):
    """
    Simulate from optimal policy with log-space noise. Fit OLS:
      log c = b0 + b1*log k + b2*log z + epsilon
    on expanding windows. Log out-of-sample MSE and out-of-sample R^2.
    Both metrics are evaluated on the same held-out test set and track
    each other with Pearson r near -1 (tight coupling in supervised learning).
    """
    np.random.seed(seed)
    T_TOTAL = T_SIM + T_TEST

    ik, iz = N_K // 2, 0
    ks, zs, lcs = [], [], []
    for _ in range(T_TOTAL):
        k = k_grid[ik]
        z = Z_VALS[iz]
        c = (1 - ALPHA * BETA) * z * k ** ALPHA
        lcs.append(np.log(c) + np.random.normal(0, NOISE_STD))
        ks.append(k); zs.append(z)
        k_next = ALPHA * BETA * z * k ** ALPHA
        ik = int(np.argmin(np.abs(k_grid - k_next)))
        iz = int(np.random.choice(N_Z, p=PI_TRANS[iz]))

    log_k = np.log(np.array(ks))
    log_z = np.log(np.array(zs))
    log_c = np.array(lcs)

    X_te = np.column_stack([np.ones(T_TEST), log_k[-T_TEST:], log_z[-T_TEST:]])
    y_te = log_c[-T_TEST:]
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)

    n_list, mse_oos_list, r2_oos_list = [], [], []
    for n in range(10, T_SIM + 1, 20):
        X_tr = np.column_stack([np.ones(n), log_k[:n], log_z[:n]])
        y_tr = log_c[:n]
        beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]

        y_hat_oos = X_te @ beta
        ss_res    = np.sum((y_hat_oos - y_te) ** 2)
        mse_oos   = float(np.mean((y_hat_oos - y_te) ** 2))
        r2_oos    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        n_list.append(n); mse_oos_list.append(mse_oos); r2_oos_list.append(r2_oos)

    return np.array(n_list), np.array(mse_oos_list), np.array(r2_oos_list)


# ============================================================================
# Figure
# ============================================================================

def make_figure(brm_results, fqe_results, ols_n, ols_mse_in, ols_r2_oos):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # ---- Left: OLS tight-coupling baseline ----
    c_mse = COLORS['red']
    c_r2  = COLORS['blue']

    ax_l.plot(ols_n, ols_mse_in, color=c_mse, linewidth=2, label='Out-of-sample MSE (left)')
    ax_l.set_xlabel('Training observations')
    ax_l.set_ylabel('Out-of-sample MSE', color=c_mse)
    ax_l.tick_params(axis='y', labelcolor=c_mse)
    ax_l.set_title('OLS: Euler Equation Regression')

    ax_l2 = ax_l.twinx()
    ax_l2.plot(ols_n, ols_r2_oos, color=c_r2, linewidth=2, linestyle='--',
               label='Out-of-sample R² (right)')
    ax_l2.set_ylabel('Out-of-sample R²', color=c_r2)
    ax_l2.tick_params(axis='y', labelcolor=c_r2)
    ax_l2.spines['right'].set_visible(True)

    lines1, lbl1 = ax_l.get_legend_handles_labels()
    lines2, lbl2 = ax_l2.get_legend_handles_labels()
    ax_l.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc='center right')

    r_ols, _ = pearsonr(ols_mse_in, ols_r2_oos)
    ax_l.text(0.05, 0.08, f'r(MSE, R²) = {r_ols:.3f}',
              transform=ax_l.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

    # ---- Right: BRM vs FQE on offline Brock-Mirman data ----
    c_brm = COLORS['orange']
    c_fqe = COLORS['blue']

    steps = brm_results[SEEDS[0]][0]

    for alg_name, results, color in [('BRM', brm_results, c_brm),
                                      ('FQE', fqe_results, c_fqe)]:
        be_all = np.array([results[s][1] for s in SEEDS])
        ve_all = np.array([results[s][2] for s in SEEDS])
        be_mean = be_all.mean(0); be_se = be_all.std(0) / np.sqrt(len(SEEDS))
        ve_mean = ve_all.mean(0); ve_se = ve_all.std(0) / np.sqrt(len(SEEDS))

        ax_r.plot(steps, be_mean, color=color, linewidth=2,
                  label=f'{alg_name}: Bellman error on D')
        ax_r.fill_between(steps, be_mean - be_se, be_mean + be_se,
                          color=color, alpha=0.15)
        ax_r.plot(steps, ve_mean, color=color, linewidth=2, linestyle='--',
                  label=f'{alg_name}: Value error (all states)')
        ax_r.fill_between(steps, ve_mean - ve_se, ve_mean + ve_se,
                          color=color, alpha=0.15)

    ax_r.set_xlabel('Gradient steps')
    ax_r.set_ylabel('Error (log scale)')
    ax_r.set_yscale('log')
    ax_r.set_title('BRM vs FQE: Offline Brock–Mirman Data')
    ax_r.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'brock_mirman_bellman.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")
    return r_ols


# ============================================================================
# compute_data
# ============================================================================

def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("=" * 70)
    print("Brock-Mirman: BRM vs FQE on Offline Data")
    print("Demonstrates Fujimoto (2022): lower Bellman error != lower value error")
    print(f"alpha={ALPHA}, beta={BETA}, N_K={N_K}, N_S={N_S}, N_A={N_A}, gamma={GAMMA}")
    print(f"Offline dataset: T={T_OFFLINE} | Training: {TRAIN_STEPS:,} steps x {len(SEEDS)} seeds")
    print("=" * 70)

    k_grid = build_capital_grid()
    print(f"\nCapital grid: [{k_grid[0]:.4f}, {k_grid[-1]:.4f}], log-spaced, N={N_K}")

    print("\n[1] Building MDP (R, P)...")
    R, P = build_mdp(k_grid)
    feasible = R > -1e30
    print(f"  Feasible (s,a) pairs: {feasible.sum()} / {N_S * N_A}")

    print("\n[2] Computing Q* via value iteration...")
    Q_star, V_star = compute_q_star(R, P)

    # Reward normalization: shift by -r_mean so Q* is centered near 0
    r_mean    = R[feasible].mean()
    R_shifted = R.copy()
    R_shifted[feasible] -= r_mean
    Q_star_shifted, _ = compute_q_star(R_shifted, P)

    pi_star = np.argmax(np.where(feasible, Q_star, -np.inf), axis=1)

    print(f"  r_mean (shift):     {r_mean:.4f}")
    print(f"  Q* original:  [{Q_star[feasible].min():.3f}, {Q_star[feasible].max():.3f}]")
    print(f"  Q* shifted:   [{Q_star_shifted[feasible].min():.3f}, "
          f"{Q_star_shifted[feasible].max():.3f}]")

    # Verify optimal policy against closed form
    pi_cf = optimal_policy(k_grid)
    cf_agree = float(np.mean(pi_star == pi_cf))
    print(f"  VI vs closed-form policy agreement: {cf_agree*100:.1f}%")

    feats = build_feature_matrix(k_grid)
    feats_t = torch.tensor(feats)
    feas_penalty = make_feas_penalty_tensor(feasible)

    print("\n[3] Generating offline dataset D from optimal policy (seed=0)...")
    D_sf, D_a, D_r, D_sfn, D_sn = generate_offline_dataset(
        k_grid, R_shifted, feats, pi_star, seed=0)

    # Unique (s,a) pairs in D — proxy for coverage
    unique_sa = len(set(zip(D_sn.tolist(), D_a.tolist())))
    print(f"  Dataset size: {len(D_sf)} transitions")
    print(f"  Unique (s',a) pairs in D: {unique_sa} / {feasible.sum()} feasible")
    print(f"  Coverage: {unique_sa / feasible.sum() * 100:.1f}%")
    print(f"  Reward stats (shifted): mean={D_r.mean():.4f}, "
          f"min={D_r.min():.4f}, max={D_r.max():.4f}")

    print(f"\n[4] Training BRM ({len(SEEDS)} seeds x {TRAIN_STEPS:,} steps)...")
    brm_results = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        steps, be, ve, pa = run_brm(
            seed, D_sf, D_a, D_r, D_sfn, D_sn,
            feats_t, Q_star_shifted, feasible, pi_star, feas_penalty)
        brm_results[seed] = (steps, be, ve, pa)
        ratio = ve[-1] / (be[-1] + 1e-10)
        r_s, _ = pearsonr(be, ve)
        print(f"    Final BE on D:       {be[-1]:.6f}")
        print(f"    Final VE (all s,a):  {ve[-1]:.6f}")
        print(f"    VE/BE ratio:         {ratio:.2f}")
        print(f"    Final pol. agree:    {pa[-1]*100:.1f}%")
        print(f"    Pearson r(BE, VE):   {r_s:.3f}")

    print(f"\n[5] Training FQE ({len(SEEDS)} seeds x {TRAIN_STEPS:,} steps)...")
    fqe_results = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        steps, be, ve, pa = run_fqe(
            seed, D_sf, D_a, D_r, D_sfn, D_sn,
            feats_t, Q_star_shifted, feasible, pi_star, feas_penalty)
        fqe_results[seed] = (steps, be, ve, pa)
        ratio = ve[-1] / (be[-1] + 1e-10)
        r_s, _ = pearsonr(be, ve)
        print(f"    Final BE on D:       {be[-1]:.6f}")
        print(f"    Final VE (all s,a):  {ve[-1]:.6f}")
        print(f"    VE/BE ratio:         {ratio:.2f}")
        print(f"    Final pol. agree:    {pa[-1]*100:.1f}%")
        print(f"    Pearson r(BE, VE):   {r_s:.3f}")

    print("\n[6] Running OLS expanding window...")
    ols_n, ols_mse_in, ols_r2_oos = run_ols(k_grid)
    r_ols, _ = pearsonr(ols_mse_in, ols_r2_oos)
    print(f"  Window: {ols_n[0]}–{ols_n[-1]} obs | Noise std: {NOISE_STD}")
    print(f"  Initial OOS MSE: {ols_mse_in[0]:.6f}, Final OOS MSE: {ols_mse_in[-1]:.6f}")
    print(f"  Initial OOS R²:  {ols_r2_oos[0]:.4f},  Final OOS R²:  {ols_r2_oos[-1]:.4f}")
    print(f"  Pearson r(OOS MSE, OOS R²): {r_ols:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("Note: BE = mean squared Bellman error on D (current net both sides);")
    print("      VE = mean absolute value error over all feasible (s,a) pairs")
    print("=" * 80)
    print(f"{'Method':<8} {'Seed':>6} {'Final BE':>12} {'Final VE':>12} "
          f"{'VE/BE':>9} {'Pol.Agree':>10} {'r(BE,VE)':>10}")
    print("-" * 80)

    brm_be_finals, brm_ve_finals, brm_ratios = [], [], []
    for seed in SEEDS:
        _, be, ve, pa = brm_results[seed]
        ratio = ve[-1] / (be[-1] + 1e-10)
        r_s, _ = pearsonr(be, ve)
        brm_be_finals.append(be[-1]); brm_ve_finals.append(ve[-1])
        brm_ratios.append(ratio)
        print(f"{'BRM':<8} {seed:>6} {be[-1]:>12.6f} {ve[-1]:>12.6f} "
              f"{ratio:>9.2f} {pa[-1]*100:>9.1f}% {r_s:>10.3f}")
    print(f"{'BRM mean':<8} {'---':>6} {np.mean(brm_be_finals):>12.6f} "
          f"{np.mean(brm_ve_finals):>12.6f} {np.mean(brm_ratios):>9.2f} "
          f"{'---':>10} {'---':>10}")
    print()

    fqe_be_finals, fqe_ve_finals, fqe_ratios = [], [], []
    for seed in SEEDS:
        _, be, ve, pa = fqe_results[seed]
        ratio = ve[-1] / (be[-1] + 1e-10)
        r_s, _ = pearsonr(be, ve)
        fqe_be_finals.append(be[-1]); fqe_ve_finals.append(ve[-1])
        fqe_ratios.append(ratio)
        print(f"{'FQE':<8} {seed:>6} {be[-1]:>12.6f} {ve[-1]:>12.6f} "
              f"{ratio:>9.2f} {pa[-1]*100:>9.1f}% {r_s:>10.3f}")
    print(f"{'FQE mean':<8} {'---':>6} {np.mean(fqe_be_finals):>12.6f} "
          f"{np.mean(fqe_ve_finals):>12.6f} {np.mean(fqe_ratios):>9.2f} "
          f"{'---':>10} {'---':>10}")
    print()
    print(f"{'OLS':<8} {'---':>6} OOS MSE={ols_mse_in[-1]:.6f}  OOS R²={ols_r2_oos[-1]:.4f}  r={r_ols:.3f}")

    # ------------------------------------------------------------------
    # Verification checks
    # ------------------------------------------------------------------
    brm_be_mean = np.mean(brm_be_finals)
    brm_ve_mean = np.mean(brm_ve_finals)
    brm_ratio_mean = np.mean(brm_ratios)
    fqe_be_mean = np.mean(fqe_be_finals)
    fqe_ve_mean = np.mean(fqe_ve_finals)
    fqe_ratio_mean = np.mean(fqe_ratios)

    print("\n" + "=" * 80)
    print("VERIFICATION (Fujimoto 2022 Table 1 pattern)")
    print("=" * 80)
    chk1 = brm_be_mean < fqe_be_mean
    chk2 = brm_ve_mean > fqe_ve_mean
    chk3 = brm_ratio_mean > fqe_ratio_mean
    chk4 = r_ols < -0.90
    print(f"  [{'PASS' if chk1 else 'FAIL'}] BRM BE < FQE BE: "
          f"{brm_be_mean:.6f} vs {fqe_be_mean:.6f}")
    print(f"  [{'PASS' if chk2 else 'FAIL'}] BRM VE > FQE VE: "
          f"{brm_ve_mean:.6f} vs {fqe_ve_mean:.6f}")
    print(f"  [{'PASS' if chk3 else 'FAIL'}] BRM VE/BE > FQE VE/BE: "
          f"{brm_ratio_mean:.2f} vs {fqe_ratio_mean:.2f}")
    print(f"  [{'PASS' if chk4 else 'FAIL'}] OLS r(MSE,R²) < -0.90: {r_ols:.3f}")

    # ------------------------------------------------------------------
    # Pack data for caching
    # ------------------------------------------------------------------
    # Convert brm_results and fqe_results dicts with tuple values
    # into serializable form
    brm_serializable = {}
    for seed in SEEDS:
        steps, be, ve, pa = brm_results[seed]
        brm_serializable[seed] = (steps.tolist(), be.tolist(), ve.tolist(), pa.tolist())

    fqe_serializable = {}
    for seed in SEEDS:
        steps, be, ve, pa = fqe_results[seed]
        fqe_serializable[seed] = (steps.tolist(), be.tolist(), ve.tolist(), pa.tolist())

    data = {
        'brm_results': brm_serializable,
        'fqe_results': fqe_serializable,
        'ols_n': ols_n.tolist(),
        'ols_mse_in': ols_mse_in.tolist(),
        'ols_r2_oos': ols_r2_oos.tolist(),
    }

    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ============================================================================
# generate_outputs
# ============================================================================

def _unpack_results(data):
    """Convert cached data back to the format make_figure expects."""
    brm_results = {}
    for seed_str, (steps, be, ve, pa) in data['brm_results'].items():
        seed = int(seed_str) if isinstance(seed_str, str) else seed_str
        brm_results[seed] = (np.array(steps), np.array(be), np.array(ve), np.array(pa))

    fqe_results = {}
    for seed_str, (steps, be, ve, pa) in data['fqe_results'].items():
        seed = int(seed_str) if isinstance(seed_str, str) else seed_str
        fqe_results[seed] = (np.array(steps), np.array(be), np.array(ve), np.array(pa))

    ols_n = np.array(data['ols_n'])
    ols_mse_in = np.array(data['ols_mse_in'])
    ols_r2_oos = np.array(data['ols_r2_oos'])

    return brm_results, fqe_results, ols_n, ols_mse_in, ols_r2_oos


def generate_outputs(data):
    brm_results, fqe_results, ols_n, ols_mse_in, ols_r2_oos = _unpack_results(data)

    print("\n[7] Generating figure...")
    r_ols_fig = make_figure(brm_results, fqe_results, ols_n, ols_mse_in, ols_r2_oos)

    print("\nOutput files:")
    print(f"  {os.path.join(OUTPUT_DIR, 'brock_mirman_bellman.png')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'brock_mirman_bellman_stdout.txt')}")


# ============================================================================
# Main
# ============================================================================

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
