"""
brock_mirman_dqn.py
Chapter: The Empirics of Deep RL
Demonstrates the TD loss vs. true value error disconnect using the
Brock-Mirman stochastic growth model where Q* is known analytically via VI.
"""

import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
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

# DQN
BUFFER_SIZE   = 10_000
BATCH_SIZE    = 128
LR            = 5e-4
HIDDEN        = 64
TARGET_UPDATE = 200
TRAIN_STEPS   = 80_000
LOG_EVERY     = 500
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 40_000   # linear decay

SEEDS = [42, 123, 777]

# OLS
T_SIM    = 2000
T_TEST   = 500
NOISE_STD = 0.05   # log-consumption noise std

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR  = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'brock_mirman_dqn'
CONFIG = {
    'alpha': ALPHA,
    'beta': BETA,
    'z_vals': Z_VALS.tolist(),
    'pi_trans': PI_TRANS.tolist(),
    'n_k': N_K,
    'n_z': N_Z,
    'buffer_size': BUFFER_SIZE,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'hidden': HIDDEN,
    'target_update': TARGET_UPDATE,
    'train_steps': TRAIN_STEPS,
    'log_every': LOG_EVERY,
    'eps_start': EPS_START,
    'eps_end': EPS_END,
    'eps_decay': EPS_DECAY,
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
    k_ss_high = (ALPHA * BETA * Z_VALS.max()) ** (1 / (1 - ALPHA))
    return np.logspace(np.log10(0.01), np.log10(1.5 * k_ss_high), N_K)


def build_mdp(k_grid):
    """Build R[s,a] and P[s,a,s'] tensors."""
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
    """Value iteration → Q*(s,a) = R(s,a) + γ Σ P(s,a,s') V*(s')."""
    V = np.zeros(N_S)
    for _ in range(max_iter):
        Q = R + GAMMA * np.einsum('san,n->sa', P, V)
        Q_masked = np.where(R > -1e30, Q, -np.inf)
        V_new = np.max(Q_masked, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q_star = R + GAMMA * np.einsum('san,n->sa', P, V)
    Q_star[R <= -1e30] = -np.inf
    return Q_star, V


def build_feature_matrix(k_grid):
    """(N_S, 2) normalized feature matrix: [k/k_max, z/z_max]."""
    feats = np.zeros((N_S, 2), dtype=np.float32)
    for ik in range(N_K):
        for iz in range(N_Z):
            s = ik * N_Z + iz
            feats[s] = [k_grid[ik] / k_grid[-1], Z_VALS[iz] / Z_VALS[-1]]
    return feats

# ============================================================================
# DQN
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


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, s_feat, a, r, s_next_feat, s_next_idx):
        self.buf.append((s_feat, a, r, s_next_feat, s_next_idx))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        sf, a, r, sfn, sni = zip(*batch)
        return (np.array(sf, np.float32), np.array(a, np.int64),
                np.array(r, np.float32), np.array(sfn, np.float32),
                np.array(sni, np.int64))
    def __len__(self):
        return len(self.buf)


def run_dqn(seed, k_grid, R, P, Q_star, feats, feasible):
    """Train DQN; return (steps, td_losses, val_errors, pol_agrees)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    online = QNet()
    target_net = QNet()
    target_net.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=LR)

    buf = ReplayBuffer(BUFFER_SIZE)
    pi_star = np.argmax(np.where(feasible, Q_star, -np.inf), axis=1)

    # Start from a random state
    ik = np.random.randint(N_K)
    iz = np.random.randint(N_Z)

    steps_log, td_log, ve_log, pa_log = [], [], [], []

    for step in range(1, TRAIN_STEPS + 1):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
        s = ik * N_Z + iz
        sf = feats[s]
        fa = np.where(feasible[s])[0]

        # Epsilon-greedy (mask infeasible)
        if np.random.rand() < eps or len(fa) == 0:
            a = int(np.random.choice(fa)) if len(fa) > 0 else 0
        else:
            with torch.no_grad():
                qv = online(torch.tensor(sf).unsqueeze(0)).numpy()[0]
                qv_masked = np.where(feasible[s], qv, -np.inf)
                a = int(np.argmax(qv_masked))

        r = R[s, a]
        iz_next = int(np.random.choice(N_Z, p=PI_TRANS[iz]))
        ik_next = a
        s_next = ik_next * N_Z + iz_next

        buf.push(sf, a, float(r), feats[s_next], s_next)
        ik, iz = ik_next, iz_next

        if len(buf) < BATCH_SIZE:
            continue

        sf_b, a_b, r_b, sfn_b, sni_b = buf.sample(BATCH_SIZE)
        sf_t  = torch.tensor(sf_b)
        a_t   = torch.tensor(a_b)
        r_t   = torch.tensor(r_b)
        sfn_t = torch.tensor(sfn_b)

        with torch.no_grad():
            qn = target_net(sfn_t).numpy()  # (B, N_A)
            # Mask infeasible actions in next states
            for i, sni in enumerate(sni_b):
                qn[i, ~feasible[sni]] = -np.inf
            q_next_max = np.max(qn, axis=1)
            q_next_max = np.where(np.isfinite(q_next_max), q_next_max, 0.0)
            y = r_t.numpy() + GAMMA * q_next_max
            y_t = torch.tensor(y, dtype=torch.float32)

        q_pred = online(sf_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = ((q_pred - y_t) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        opt.step()

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(online.state_dict())

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                q_theta = online(torch.tensor(feats)).numpy()  # (N_S, N_A)
            # True value error over all feasible (s,a) pairs
            ve = float(np.abs(q_theta[feasible] - Q_star[feasible]).mean())
            # Policy agreement
            pi_theta = np.argmax(np.where(feasible, q_theta, -np.inf), axis=1)
            pa = float(np.mean(pi_theta == pi_star))

            steps_log.append(step)
            td_log.append(float(loss.item()))
            ve_log.append(ve)
            pa_log.append(pa)

    return (np.array(steps_log), np.array(td_log),
            np.array(ve_log), np.array(pa_log))

# ============================================================================
# OLS expanding window
# ============================================================================

def run_ols(k_grid, seed=42):
    """
    Simulate from true policy; fit OLS on expanding windows;
    log out-of-sample MSE and R² on held-out test set.
    """
    np.random.seed(seed)
    T_TOTAL = T_SIM + T_TEST

    ik, iz = N_K // 2, 0
    ks, zs, lcs = [], [], []
    for _ in range(T_TOTAL):
        k = k_grid[ik]
        z = Z_VALS[iz]
        c = (1 - ALPHA * BETA) * z * k ** ALPHA
        log_c_obs = np.log(c) + np.random.normal(0, NOISE_STD)
        ks.append(k); zs.append(z); lcs.append(log_c_obs)
        k_next = ALPHA * BETA * z * k ** ALPHA
        ik = int(np.argmin(np.abs(k_grid - k_next)))
        iz = int(np.random.choice(N_Z, p=PI_TRANS[iz]))

    log_k = np.log(np.array(ks))
    log_z = np.log(np.array(zs))
    log_c = np.array(lcs)

    # Held-out test set: last T_TEST obs
    X_te = np.column_stack([np.ones(T_TEST), log_k[-T_TEST:], log_z[-T_TEST:]])
    y_te = log_c[-T_TEST:]
    y_te_bar = y_te.mean()

    n_list, mse_list, r2_list = [], [], []
    for n in range(30, T_SIM + 1, 10):
        X_tr = np.column_stack([np.ones(n), log_k[:n], log_z[:n]])
        y_tr = log_c[:n]
        beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        y_hat = X_te @ beta
        mse = float(np.mean((y_hat - y_te) ** 2))
        ss_res = np.sum((y_hat - y_te) ** 2)
        ss_tot = np.sum((y_te - y_te_bar) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        n_list.append(n); mse_list.append(mse); r2_list.append(r2)

    return np.array(n_list), np.array(mse_list), np.array(r2_list)

# ============================================================================
# Figure
# ============================================================================

def make_figure(dqn_results, ols_n, ols_mse, ols_r2):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # ---- Left: OLS ----
    color_mse = COLORS['red']
    color_r2  = COLORS['blue']

    ax_l.plot(ols_n, ols_mse * 1e3, color=color_mse, linewidth=2,
              label='MSE ×10⁻³ (left axis)')
    ax_l.set_xlabel('Training observations')
    ax_l.set_ylabel('Out-of-sample MSE (×10⁻³)', color=color_mse)
    ax_l.tick_params(axis='y', labelcolor=color_mse)
    ax_l.set_title('OLS: Euler Equation Regression')

    ax_l2 = ax_l.twinx()
    ax_l2.plot(ols_n, ols_r2, color=color_r2, linewidth=2, linestyle='--',
               label='R² (right axis)')
    ax_l2.set_ylabel('Out-of-sample R²', color=color_r2)
    ax_l2.tick_params(axis='y', labelcolor=color_r2)
    ax_l2.set_ylim(bottom=0)
    ax_l2.spines['right'].set_visible(True)

    lines1, lbl1 = ax_l.get_legend_handles_labels()
    lines2, lbl2 = ax_l2.get_legend_handles_labels()
    ax_l.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc='center right')

    r_ols, _ = pearsonr(ols_mse, ols_r2)
    ax_l.text(0.05, 0.92, f'r(MSE, R²) = {r_ols:.3f}',
              transform=ax_l.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray'], alpha=0.6))

    # ---- Right: DQN ----
    td_all = np.array([dqn_results[s][1] for s in SEEDS])
    ve_all = np.array([dqn_results[s][2] for s in SEEDS])
    common_steps = dqn_results[SEEDS[0]][0]

    td_mean = td_all.mean(0); td_se = td_all.std(0) / np.sqrt(len(SEEDS))
    ve_mean = ve_all.mean(0); ve_se = ve_all.std(0) / np.sqrt(len(SEEDS))

    # Normalize both series to [0,1] for visual comparison on shared axis
    def norm01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    td_n = norm01(td_mean)
    ve_n = norm01(ve_mean)
    td_se_n = td_se / (td_mean.max() - td_mean.min() + 1e-12)
    ve_se_n = ve_se / (ve_mean.max() - ve_mean.min() + 1e-12)

    ax_r.plot(common_steps, td_n, color=COLORS['orange'], linewidth=2,
              label='TD loss (normalized)')
    ax_r.fill_between(common_steps, td_n - td_se_n, td_n + td_se_n,
                      color=COLORS['orange'], alpha=0.15)

    ax_r.plot(common_steps, ve_n, color=COLORS['blue'], linewidth=2,
              linestyle='--', label='True value error (normalized)')
    ax_r.fill_between(common_steps, ve_n - ve_se_n, ve_n + ve_se_n,
                      color=COLORS['blue'], alpha=0.15)

    ax_r.set_xlabel('Environment steps')
    ax_r.set_ylabel('Normalized metric (0 = min, 1 = max)')
    ax_r.set_title('DQN: Brock–Mirman Growth Model')
    ax_r.legend(fontsize=8, loc='upper right')

    r_dqn, _ = pearsonr(td_mean, ve_mean)
    ax_r.text(0.05, 0.92, f'r(TD loss, value error) = {r_dqn:.3f}',
              transform=ax_r.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray'], alpha=0.6))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'brock_mirman_dqn.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")
    return r_ols, r_dqn

# ============================================================================
# compute_data
# ============================================================================

def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("=" * 65)
    print("Brock-Mirman DQN: TD Loss vs. True Value Error")
    print(f"alpha={ALPHA}, beta={BETA}, N_K={N_K}, N_S={N_S}, N_A={N_A}")
    print(f"DQN: {TRAIN_STEPS:,} steps x {len(SEEDS)} seeds")
    print("=" * 65)

    k_grid = build_capital_grid()
    print(f"\nCapital grid: [{k_grid[0]:.4f}, {k_grid[-1]:.4f}], log-spaced, N={N_K}")

    print("\n[1] Building MDP (R, P)...")
    R, P = build_mdp(k_grid)
    feasible = R > -1e30
    print(f"  Feasible (s,a) pairs: {feasible.sum()} / {N_S * N_A}")

    print("\n[2] Computing Q* via value iteration...")
    Q_star, V_star = compute_q_star(R, P)
    pi_star = np.argmax(np.where(feasible, Q_star, -np.inf), axis=1)
    print(f"  Q* over feasible pairs: min={Q_star[feasible].min():.3f}, "
          f"max={Q_star[feasible].max():.3f}")
    print(f"  V*: min={V_star.min():.3f}, max={V_star.max():.3f}")

    # Verify VI policy against closed-form k'* = αβ z k^α
    cf_agree = sum(
        pi_star[ik * N_Z + iz] == int(np.argmin(np.abs(
            k_grid - ALPHA * BETA * Z_VALS[iz] * k_grid[ik] ** ALPHA)))
        for ik in range(N_K) for iz in range(N_Z)
    )
    print(f"  VI policy vs. closed-form: {cf_agree / N_S * 100:.1f}%")

    feats = build_feature_matrix(k_grid)

    print(f"\n[3] Running DQN ({len(SEEDS)} seeds x {TRAIN_STEPS:,} steps)...")
    dqn_results = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        steps, td, ve, pa = run_dqn(seed, k_grid, R, P, Q_star, feats, feasible)
        dqn_results[seed] = (steps, td, ve, pa)
        r_s, _ = pearsonr(td, ve)
        print(f"    Log interval: every {LOG_EVERY} steps, {len(steps)} records")
        print(f"    Final TD loss:     {td[-1]:.4f}")
        print(f"    Final value error: {ve[-1]:.4f}")
        print(f"    Final pol. agree:  {pa[-1]*100:.1f}%")
        print(f"    Pearson r(TD loss, value error): {r_s:.3f}")

    print("\n[4] Running OLS expanding window...")
    ols_n, ols_mse, ols_r2 = run_ols(k_grid)
    r_ols, _ = pearsonr(ols_mse, ols_r2)
    print(f"  Window sizes: {ols_n[0]} to {ols_n[-1]} obs")
    print(f"  Initial MSE: {ols_mse[0]:.6f}, Final MSE: {ols_mse[-1]:.6f}")
    print(f"  Initial R²:  {ols_r2[0]:.4f}, Final R²:  {ols_r2[-1]:.4f}")
    print(f"  Pearson r(MSE, R²): {r_ols:.3f}")

    # Mean DQN statistics
    td_all = np.array([dqn_results[s][1] for s in SEEDS])
    ve_all = np.array([dqn_results[s][2] for s in SEEDS])
    pa_all = np.array([dqn_results[s][3] for s in SEEDS])
    td_mean = td_all.mean(0)
    ve_mean = ve_all.mean(0)
    r_dqn_mean, _ = pearsonr(td_mean, ve_mean)

    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Method':<38} {'Final loss':>10} {'Final err':>10} {'Pearson r':>10}")
    print("-" * 70)
    print(f"{'OLS (Euler eq.)':<38} {ols_mse[-1]*1e3:>9.4f}m {ols_r2[-1]:>10.4f} {r_ols:>10.3f}")
    for seed in SEEDS:
        steps, td, ve, pa = dqn_results[seed]
        r_s, _ = pearsonr(td, ve)
        print(f"{'DQN (seed '+str(seed)+')':<38} {td[-1]:>10.4f} {ve[-1]:>10.4f} {r_s:>10.3f}")
    print(f"{'DQN (mean, 3 seeds)':<38} {td_mean[-1]:>10.4f} {ve_mean[-1]:>10.4f} {r_dqn_mean:>10.3f}")

    # Pack data for caching
    dqn_serializable = {}
    for seed in SEEDS:
        steps, td, ve, pa = dqn_results[seed]
        dqn_serializable[seed] = (steps.tolist(), td.tolist(), ve.tolist(), pa.tolist())

    data = {
        'dqn_results': dqn_serializable,
        'ols_n': ols_n.tolist(),
        'ols_mse': ols_mse.tolist(),
        'ols_r2': ols_r2.tolist(),
    }

    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ============================================================================
# generate_outputs
# ============================================================================

def _unpack_results(data):
    """Convert cached data back to the format make_figure expects."""
    dqn_results = {}
    for seed_str, (steps, td, ve, pa) in data['dqn_results'].items():
        seed = int(seed_str) if isinstance(seed_str, str) else seed_str
        dqn_results[seed] = (np.array(steps), np.array(td), np.array(ve), np.array(pa))

    ols_n = np.array(data['ols_n'])
    ols_mse = np.array(data['ols_mse'])
    ols_r2 = np.array(data['ols_r2'])

    return dqn_results, ols_n, ols_mse, ols_r2


def generate_outputs(data):
    dqn_results, ols_n, ols_mse, ols_r2 = _unpack_results(data)

    print("\n[5] Generating figure...")
    make_figure(dqn_results, ols_n, ols_mse, ols_r2)

    print("\nOutput files:")
    print(f"  {os.path.join(OUTPUT_DIR, 'brock_mirman_dqn.png')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'brock_mirman_dqn_stdout.txt')}")


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
