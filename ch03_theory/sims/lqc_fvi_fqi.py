"""Linear-Quadratic Control: Fitted Value Iteration vs Fitted Q-Iteration vs DQN.

Chapter 3, Theory -- demonstrates near-zero approximation error when Q* lies in span(Phi),
and that neural function approximation (DQN) also converges since Q* is smooth and quadratic.

Model: x' = a*x + b*u, r(x,u) = -(x^2 + u^2), discount gamma.
Parameters a=0.5, b=1.0: x' = 0.5*x + u, so x' in [-4,4] whenever x in [-4,4], u in [-2,2].
Grid is exactly invariant; no boundary clipping needed.

Riccati: V*(x) = -P*x^2, Q*(x,u) = c_xx*x^2 + c_xu*xu + c_uu*u^2 with
  P solves gamma*b^2*P^2 + P*(1 - gamma*(a^2+b^2)) - 1 = 0  =>  P ~ 1.129
Both V* in span{x,x^2} and Q* in span{x,x^2,u,u^2,xu}, so both FVI and FQI converge.
Features exclude the constant: V*(0)=Q*(0,0)=0 by symmetry, no intercept needed.

Four cached components:
  exact_VI  — tabular value iteration on the discretized LQC
  FVI       — fitted value iteration with polynomial features [x, x^2]
  FQI       — fitted Q-iteration with polynomial features [x, x^2, u, u^2, xu]
  DQN       — deep Q-network with 2x64 ReLU
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, CMAP_SEQ
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)

# ── Parameters ─────────────────────────────────────────────────────────────────
# a=0.5, b=1.0: x' = 0.5x + u. For x in [-4,4], u in [-2,2]:
#   min(x') = 0.5*(-4)+(-2) = -4, max(x') = 0.5*4+2 = 4 -- exactly invariant.
a     = 0.5
b     = 1.0
gamma = 0.95
N_X   = 301     # state grid on [-4, 4], step ~ 0.0267
N_U   = 201     # action grid on [-2, 2], step = 0.02
X     = np.linspace(-4.0, 4.0, N_X)
U     = np.linspace(-2.0, 2.0, N_U)
OUTDIR   = os.path.dirname(os.path.abspath(__file__))

h_X = X[1] - X[0]
h_U = U[1] - U[0]

# ── DQN hyperparameters ──────────────────────────────────────────────────────
DQN_HIDDEN    = 64
DQN_LR        = 3e-4
DQN_BUFFER    = 50_000
DQN_BATCH     = 256
DQN_TARGET_UP = 500       # hard target-net update every 500 steps (stable bootstrap)
DQN_STEPS     = 100_000
DQN_EPS_START = 1.0
DQN_EPS_END   = 0.05
DQN_EPS_DECAY = 40_000
DQN_EVAL_INT  = 1_000     # evaluate error every N steps
REWARD_SCALE  = 20.0      # scale rewards to [-1, 0] range; Q-targets in [-1.85, 0]
DQN_N_SEEDS   = 10        # number of seeds to average DQN results over

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'lqc_fvi_fqi'

# ── Per-component config dicts ─────────────────────────────────────────────────
ENV_PARAMS = {
    'a': a, 'b': b, 'gamma': gamma,
    'N_X': N_X, 'N_U': N_U, 'version': 2,
}
EXACT_VI_CONFIG = {**ENV_PARAMS, 'MAX_ITER_VI': 20000, 'TOL_VI': 1e-10}
FVI_CONFIG      = {**EXACT_VI_CONFIG, 'MAX_ITER_FVI': 500, 'TOL_FVI': 1e-9}
FQI_CONFIG      = {**EXACT_VI_CONFIG, 'MAX_ITER_FQI': 500, 'TOL_FQI': 1e-9}
DQN_CONFIG      = {
    **ENV_PARAMS,
    'DQN_HIDDEN': DQN_HIDDEN, 'DQN_LR': DQN_LR,
    'DQN_BUFFER': DQN_BUFFER, 'DQN_BATCH': DQN_BATCH,
    'DQN_TARGET_UP': DQN_TARGET_UP, 'DQN_STEPS': DQN_STEPS,
    'DQN_EPS_START': DQN_EPS_START, 'DQN_EPS_END': DQN_EPS_END,
    'DQN_EPS_DECAY': DQN_EPS_DECAY, 'DQN_EVAL_INT': DQN_EVAL_INT,
    'REWARD_SCALE': REWARD_SCALE, 'DQN_N_SEEDS': DQN_N_SEEDS,
}

# ── Riccati solution (analytical, deterministic — stays at module level) ──────
disc = (1.0 - gamma * (a**2 + b**2))**2 + 4.0 * gamma * b**2
P    = (gamma * (a**2 + b**2) - 1.0 + np.sqrt(disc)) / (2.0 * gamma * b**2)

P_fp = 0.0
for _ in range(100000):
    P_new = 1.0 + gamma * a**2 * P_fp / (1.0 + gamma * P_fp * b**2)
    if abs(P_new - P_fp) < 1e-12:
        break
    P_fp = P_new
assert abs(P - P_fp) < 1e-6

c_xx = -(1.0 + gamma * P * a**2)
c_xu = -2.0 * gamma * P * a * b
c_uu = -(1.0 + gamma * P * b**2)
K_opt = -gamma * P * a * b / (1.0 + gamma * P * b**2)
V_star = -P * X**2

# ── Grids (deterministic, needed by both compute and output) ─────────────────
XX, UU  = np.meshgrid(X, U, indexing='ij')   # (N_X, N_U)
R        = -(XX**2 + UU**2)                   # reward
Xnext    = a * XX + b * UU                    # next state (no clipping needed)
Xnext_idx = np.clip(
    np.round((Xnext - X[0]) / h_X).astype(int),
    0, N_X - 1
)   # (N_X, N_U)


# ── DQN network and buffer classes ──────────────────────────────────────────

class QNet(nn.Module):
    """Maps normalized state x -> Q(x, u_j) for all j in U grid."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, DQN_HIDDEN), nn.ReLU(),
            nn.Linear(DQN_HIDDEN, DQN_HIDDEN), nn.ReLU(),
            nn.Linear(DQN_HIDDEN, N_U),
        )
    def forward(self, x):
        return self.net(x)   # (batch, N_U)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, x_idx, u_idx, r, xp_idx):
        self.buf.append((x_idx, u_idx, r, xp_idx))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        xi, ui, r, xpi = zip(*batch)
        return (np.array(xi), np.array(ui),
                np.array(r, dtype=np.float32), np.array(xpi))
    def __len__(self):
        return len(self.buf)


# ── Per-component compute functions ───────────────────────────────────────────

def compute_exact_vi():
    """Tabular value iteration on the discretized LQC."""
    V_exact  = np.zeros(N_X)
    vi_iters = 0
    for _ in range(EXACT_VI_CONFIG['MAX_ITER_VI']):
        Vnext   = np.interp(Xnext, X, V_exact)
        V_new   = (R + gamma * Vnext).max(axis=1)
        delta   = np.max(np.abs(V_new - V_exact))
        V_exact = V_new
        vi_iters += 1
        if delta < EXACT_VI_CONFIG['TOL_VI']:
            break

    vi_vs_analytical = np.max(np.abs(V_exact - V_star))
    print(f"\nExact VI: {vi_iters} iters, vs analytical V*: {vi_vs_analytical:.2e}")

    return {
        'V_exact': V_exact,
        'vi_iters': vi_iters,
        'vi_vs_analytical': vi_vs_analytical,
    }


def compute_fvi(exact_vi_data):
    """Fitted Value Iteration with polynomial features phi_V(x) = [x, x^2]."""
    V_exact = np.array(exact_vi_data['V_exact'])
    MAX_ITER = FVI_CONFIG['MAX_ITER_FVI']
    TOL      = FVI_CONFIG['TOL_FVI']

    # Features: phi_V(x) = [x, x^2] -- no intercept, since V*(0)=0 by symmetry.
    Phi_V    = np.column_stack([X, X**2])   # (N_X, 2)
    theta_V  = np.zeros(2)
    fvi_errs = []

    for k in range(MAX_ITER):
        V_k      = Phi_V @ theta_V
        Vnext_k  = np.interp(Xnext, X, V_k)            # (N_X, N_U) via linear interp
        V_target = (R + gamma * Vnext_k).max(axis=1)    # (N_X,)
        theta_new = np.linalg.lstsq(Phi_V, V_target, rcond=None)[0]
        err = np.max(np.abs(Phi_V @ theta_new - V_exact))
        fvi_errs.append(err)
        if np.max(np.abs(theta_new - theta_V)) < TOL:
            break
        theta_V = theta_new

    V_fvi      = Phi_V @ theta_V
    fvi_error  = np.max(np.abs(V_fvi - V_exact))
    fvi_err_an = np.max(np.abs(V_fvi - V_star))   # vs analytical V*
    fvi_iters  = len(fvi_errs)
    print(f"\nFVI: {fvi_iters} iters")
    print(f"  Error vs exact VI:    {fvi_error:.2e}")
    print(f"  Error vs analytical:  {fvi_err_an:.2e}")
    print(f"  theta: [x={theta_V[0]:.6f}, x^2={theta_V[1]:.6f}]")
    print(f"  Analytical: [0, {-P:.6f}]")
    print(f"  P recovered = {-theta_V[1]:.6f}  (true P = {P:.6f})")

    # Verification
    assert fvi_err_an < 0.001, f"FVI vs analytical: {fvi_err_an:.6f} exceeds 0.001"
    assert abs(-theta_V[1] - P) < 0.001, \
        f"FVI P recovery: {-theta_V[1]:.6f} vs {P:.6f}"

    return {
        'fvi_errs': fvi_errs,
        'fvi_error': fvi_error,
        'fvi_err_an': fvi_err_an,
        'fvi_iters': fvi_iters,
        'theta_V': theta_V,
        'V_fvi': V_fvi,
    }


def compute_fqi(exact_vi_data):
    """Fitted Q-Iteration with polynomial features phi_Q(x,u) = [x, x^2, u, u^2, xu]."""
    V_exact = np.array(exact_vi_data['V_exact'])
    MAX_ITER = FQI_CONFIG['MAX_ITER_FQI']
    TOL      = FQI_CONFIG['TOL_FQI']

    # Features: phi_Q(x,u) = [x, x^2, u, u^2, xu] -- no intercept (Q*(0,0)=0).
    XX_flat  = XX.ravel()
    UU_flat  = UU.ravel()
    Xnext_fl = Xnext.ravel()
    R_flat   = R.ravel()

    Phi_Q = np.column_stack([
        XX_flat,
        XX_flat**2,
        UU_flat,
        UU_flat**2,
        XX_flat * UU_flat,
    ])   # (N_X*N_U, 5)

    theta_Q  = np.zeros(5)
    fqi_errs = []

    for k in range(MAX_ITER):
        # Parametric evaluation of Q_k(x', u') for all u' in U -- no interpolation
        xp     = Xnext_fl[:, np.newaxis]   # (N_X*N_U, 1)
        up     = U[np.newaxis, :]           # (1, N_U)
        Q_next = (theta_Q[0] * xp
                  + theta_Q[1] * xp**2
                  + theta_Q[2] * up
                  + theta_Q[3] * up**2
                  + theta_Q[4] * xp * up)          # (N_X*N_U, N_U)
        max_Q_next = Q_next.max(axis=1)            # (N_X*N_U,)

        y         = R_flat + gamma * max_Q_next
        theta_new = np.linalg.lstsq(Phi_Q, y, rcond=None)[0]

        # Implied V_fqi(x) = max_u Q(x,u) for error tracking
        x_2d  = X[:, np.newaxis]
        u_2d  = U[np.newaxis, :]
        Q_all = (theta_new[0] * x_2d
                 + theta_new[1] * x_2d**2
                 + theta_new[2] * u_2d
                 + theta_new[3] * u_2d**2
                 + theta_new[4] * x_2d * u_2d)    # (N_X, N_U)
        V_fqi_k = Q_all.max(axis=1)               # (N_X,)

        err = np.max(np.abs(V_fqi_k - V_exact))
        fqi_errs.append(err)
        if np.max(np.abs(theta_new - theta_Q)) < TOL:
            break
        theta_Q = theta_new

    V_fqi      = V_fqi_k
    fqi_error  = np.max(np.abs(V_fqi - V_exact))
    # V_fqi(x) = max_u Q_fqi(x,u); compute analytically for comparison vs V*
    # u*(x) = -(theta[4]*x + theta[2]) / (2*theta[3])
    u_opt_x  = -(theta_Q[4] * X + theta_Q[2]) / (2.0 * theta_Q[3])
    u_opt_x  = np.clip(u_opt_x, U[0], U[-1])
    V_fqi_an = (theta_Q[0]*X + theta_Q[1]*X**2
                + theta_Q[2]*u_opt_x + theta_Q[3]*u_opt_x**2
                + theta_Q[4]*X*u_opt_x)
    fqi_err_an = np.max(np.abs(V_fqi_an - V_star))
    fqi_iters  = len(fqi_errs)
    print(f"\nFQI: {fqi_iters} iters")
    print(f"  Error vs exact VI:    {fqi_error:.2e}")
    print(f"  Error vs analytical:  {fqi_err_an:.2e}")
    print(f"  theta: [x={theta_Q[0]:.5f}, x^2={theta_Q[1]:.5f}, "
          f"u={theta_Q[2]:.5f}, u^2={theta_Q[3]:.5f}, xu={theta_Q[4]:.5f}]")
    print(f"  Analytical: [0, {c_xx:.5f}, 0, {c_uu:.5f}, {c_xu:.5f}]")

    # Verification
    assert fqi_err_an < 0.001, f"FQI vs analytical: {fqi_err_an:.6f} exceeds 0.001"
    assert abs(theta_Q[1] - c_xx) < 0.002, \
        f"FQI c_xx recovery: {theta_Q[1]:.5f} vs {c_xx:.5f}"
    assert abs(theta_Q[4] - c_xu) < 0.002, \
        f"FQI c_xu recovery: {theta_Q[4]:.5f} vs {c_xu:.5f}"
    assert abs(theta_Q[3] - c_uu) < 0.002, \
        f"FQI c_uu recovery: {theta_Q[3]:.5f} vs {c_uu:.5f}"

    return {
        'fqi_errs': fqi_errs,
        'fqi_error': fqi_error,
        'fqi_err_an': fqi_err_an,
        'fqi_iters': fqi_iters,
        'theta_Q': theta_Q,
        'V_fqi': V_fqi,
    }


def _run_dqn_single_seed(seed):
    """Run one DQN training trajectory with the given seed.

    Under the repository color and seed conventions: numpy, torch, and
    python-random seeds are all set so that replay sampling, ε-greedy choice,
    weight init, and reset positions are deterministic given `seed`.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    online = QNet()
    target = QNet()
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=DQN_LR)

    buf = ReplayBuffer(DQN_BUFFER)
    step_log = []
    err_log  = []

    x_idx = np.random.randint(N_X)   # start at random state

    for step in range(1, DQN_STEPS + 1):
        eps = max(DQN_EPS_END,
                  DQN_EPS_START - (DQN_EPS_START - DQN_EPS_END) * step / DQN_EPS_DECAY)

        # Epsilon-greedy action
        if np.random.rand() < eps:
            u_idx = np.random.randint(N_U)
        else:
            x_t = torch.tensor([[X[x_idx] / 4.0]], dtype=torch.float32)
            with torch.no_grad():
                u_idx = int(online(x_t).argmax().item())

        r_val  = float(R[x_idx, u_idx]) / REWARD_SCALE   # scaled to ~[-1, 0]
        xp_idx = int(Xnext_idx[x_idx, u_idx])
        buf.push(x_idx, u_idx, r_val, xp_idx)
        x_idx = xp_idx   # step forward

        # Reset to random state every 20 steps so the buffer covers large |x|
        if step % 20 == 0:
            x_idx = np.random.randint(N_X)

        if len(buf) < DQN_BATCH:
            continue

        # Sample minibatch
        xi, ui, r_b, xpi = buf.sample(DQN_BATCH)
        x_t  = torch.tensor(X[xi, np.newaxis] / 4.0, dtype=torch.float32)
        xp_t = torch.tensor(X[xpi, np.newaxis] / 4.0, dtype=torch.float32)
        r_t  = torch.tensor(r_b, dtype=torch.float32)
        ui_t = torch.tensor(ui, dtype=torch.long)

        with torch.no_grad():
            max_q_next = target(xp_t).max(dim=1).values
        y = r_t + gamma * max_q_next

        q_pred = online(x_t).gather(1, ui_t.unsqueeze(1)).squeeze(1)
        loss   = nn.functional.mse_loss(q_pred, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        opt.step()

        if step % DQN_TARGET_UP == 0:
            target.load_state_dict(online.state_dict())

        if step % DQN_EVAL_INT == 0:
            with torch.no_grad():
                x_all = torch.tensor(X[:, np.newaxis] / 4.0, dtype=torch.float32)
                V_dqn_now = online(x_all).max(dim=1).values.numpy() * REWARD_SCALE
            err = np.max(np.abs(V_dqn_now - V_star))
            step_log.append(step)
            err_log.append(err)

    # Final DQN value function (rescale back to original units)
    with torch.no_grad():
        x_all = torch.tensor(X[:, np.newaxis] / 4.0, dtype=torch.float32)
        V_dqn = online(x_all).max(dim=1).values.numpy() * REWARD_SCALE
    final_err = float(np.max(np.abs(V_dqn - V_star)))
    return np.asarray(step_log), np.asarray(err_log), V_dqn, final_err


def compute_dqn():
    """DQN with 2x64 ReLU network. Loops over DQN_N_SEEDS seeds and reports
    mean and standard error across seeds for the final value-function error
    and the learning curve. Reproducibility: numpy + torch + python-random
    seeds are set inside `_run_dqn_single_seed` for each seed independently.
    """
    print(f"\nDQN training ({DQN_STEPS} steps × {DQN_N_SEEDS} seeds)...")
    step_log_ref = None
    err_curves = []          # (N_SEEDS, n_evals)
    final_errs = []
    V_seeds = []

    for s_idx in range(DQN_N_SEEDS):
        seed = 42 + s_idx
        step_log, err_log, V_dqn_s, final_err = _run_dqn_single_seed(seed)
        if step_log_ref is None:
            step_log_ref = step_log
        err_curves.append(err_log)
        final_errs.append(final_err)
        V_seeds.append(V_dqn_s)
        print(f"  seed {seed:3d}: final err vs V* = {final_err:.4f}")

    err_curves = np.asarray(err_curves)                 # (N_SEEDS, n_evals)
    final_errs = np.asarray(final_errs)                 # (N_SEEDS,)
    V_seeds    = np.asarray(V_seeds)                    # (N_SEEDS, N_X)

    # Mean ± SE (SE = std / sqrt(N)) across seeds
    err_mean = err_curves.mean(axis=0)
    err_se   = err_curves.std(axis=0, ddof=1) / np.sqrt(DQN_N_SEEDS)
    dqn_err_an_mean = float(final_errs.mean())
    dqn_err_an_se   = float(final_errs.std(ddof=1) / np.sqrt(DQN_N_SEEDS))
    V_dqn_mean = V_seeds.mean(axis=0)

    print(f"\nDQN: {DQN_STEPS} steps, {DQN_N_SEEDS} seeds")
    print(f"  Final error vs analytical V*: {dqn_err_an_mean:.4f} ± {dqn_err_an_se:.4f} (mean ± SE)")
    print(f"  Min final err across seeds:   {final_errs.min():.4f}")
    print(f"  Max final err across seeds:   {final_errs.max():.4f}")

    # Verification: every seed's final error stays in the documented regime
    assert dqn_err_an_mean < 2.0, \
        f"DQN mean error {dqn_err_an_mean:.4f} exceeds 2.0"

    return {
        'dqn_step_log': step_log_ref,
        'dqn_err_mean': err_mean,
        'dqn_err_se':   err_se,
        'dqn_err_an':   dqn_err_an_mean,
        'dqn_err_an_se': dqn_err_an_se,
        'dqn_final_errs': final_errs,
        'V_dqn': V_dqn_mean,
        'n_seeds': DQN_N_SEEDS,
        # Back-compat name for any external caller that read 'dqn_err_log'
        'dqn_err_log': err_mean,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def compute_data(force=None):
    force = force or set()

    exact_vi = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'exact_VI', EXACT_VI_CONFIG,
                                compute_exact_vi, force=('exact_VI' in force))
    fvi = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'FVI', FVI_CONFIG,
                           compute_fvi, exact_vi,
                           force=('FVI' in force or 'exact_VI' in force))
    fqi = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'FQI', FQI_CONFIG,
                           compute_fqi, exact_vi,
                           force=('FQI' in force or 'exact_VI' in force))
    dqn = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'DQN', DQN_CONFIG,
                           compute_dqn, force=('DQN' in force))

    return {'exact_VI': exact_vi, 'FVI': fvi, 'FQI': fqi, 'DQN': dqn}


# ── generate_outputs ──────────────────────────────────────────────────────────

def generate_outputs(data):
    vi_iters       = data['exact_VI']['vi_iters']
    vi_vs_analytical = data['exact_VI']['vi_vs_analytical']
    fvi_errs       = data['FVI']['fvi_errs']
    fvi_error      = data['FVI']['fvi_error']
    fvi_err_an     = data['FVI']['fvi_err_an']
    fvi_iters      = data['FVI']['fvi_iters']
    theta_V        = np.array(data['FVI']['theta_V'])
    V_fvi          = np.array(data['FVI']['V_fvi'])
    fqi_errs       = data['FQI']['fqi_errs']
    fqi_error      = data['FQI']['fqi_error']
    fqi_err_an     = data['FQI']['fqi_err_an']
    fqi_iters      = data['FQI']['fqi_iters']
    theta_Q        = np.array(data['FQI']['theta_Q'])
    V_fqi          = np.array(data['FQI']['V_fqi'])
    dqn_step_log   = np.array(data['DQN']['dqn_step_log'])
    dqn_err_mean   = np.array(data['DQN'].get('dqn_err_mean',
                                              data['DQN'].get('dqn_err_log')))
    dqn_err_se     = np.array(data['DQN'].get('dqn_err_se',
                                              np.zeros_like(dqn_err_mean)))
    dqn_err_an     = data['DQN']['dqn_err_an']
    dqn_err_an_se  = data['DQN'].get('dqn_err_an_se', 0.0)
    dqn_n_seeds    = data['DQN'].get('n_seeds', 1)
    V_dqn          = np.array(data['DQN']['V_dqn'])

    # ── Summary table (stdout) ────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'Method':<18} {'Iters':>6} {'Err vs VI':>10} {'Err vs V*':>10} "
          f"{'P_recov':>10} {'c_xx':>8} {'c_xu':>8} {'c_uu':>8}")
    print(f"{'-'*78}")
    print(f"{'Exact VI':<18} {vi_iters:>6d} {'---':>10} {vi_vs_analytical:>10.2e} "
          f"{'---':>10} {'---':>8} {'---':>8} {'---':>8}")
    print(f"{'FVI':<18} {fvi_iters:>6d} {fvi_error:>10.2e} {fvi_err_an:>10.2e} "
          f"{-theta_V[1]:>10.4f} {theta_V[1]:>8.4f} {'---':>8} {'---':>8}")
    print(f"{'FQI':<18} {fqi_iters:>6d} {fqi_error:>10.2e} {fqi_err_an:>10.2e} "
          f"{-theta_Q[1]:>10.4f} {theta_Q[1]:>8.4f} {theta_Q[4]:>8.4f} {theta_Q[3]:>8.4f}")
    print(f"{'DQN (2x64 ReLU)':<18} {DQN_STEPS:>6d} {'---':>10} {dqn_err_an:>10.2e} "
          f"{'---':>10} {'---':>8} {'---':>8} {'---':>8}")
    print(f"{'  DQN seeds':<18} {dqn_n_seeds:>6d} {'---':>10} {'± SE':>10} "
          f"{dqn_err_an_se:>10.2e} {'---':>8} {'---':>8} {'---':>8}")
    print(f"{'Analytical':<18} {'---':>6} {'---':>10} {'0':>10} "
          f"{P:>10.4f} {c_xx:>8.4f} {c_xu:>8.4f} {c_uu:>8.4f}")
    print(f"{'='*78}")

    # ── LaTeX table ────────────────────────────────────────────────────────────
    # DQN row reports mean ± SE across DQN_N_SEEDS seeds; FVI/FQI/exact VI are
    # deterministic given the grid, so no SE is reported for them.
    dqn_err_str = rf"{dqn_err_an:.2e} $\pm$ {dqn_err_an_se:.2e}"
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Iterations & Error vs $V^*$ & $P$ (recovered) & Key coefficient \\",
        r"\hline",
        rf"Exact VI (discrete) & {vi_iters} & {vi_vs_analytical:.2e} & --- & --- \\",
        (rf"FVI & {fvi_iters} & {fvi_err_an:.2e} & {-theta_V[1]:.4f}"
         rf" & $\hat\theta_V^{{x^2}} = {theta_V[1]:.4f}$ \\"),
        (rf"FQI & {fqi_iters} & {fqi_err_an:.2e} & {-theta_Q[1]:.4f}"
         rf" & $\hat\theta_Q^{{xu}} = {theta_Q[4]:.4f}$ \\"),
        (rf"DQN ($2 \times 64$ ReLU, {dqn_n_seeds} seeds) & {DQN_STEPS}"
         rf" & {dqn_err_str} & --- & --- \\"),
        (rf"Analytical ($V^* = -Px^2$) & --- & 0 & {P:.4f}"
         rf" & $c_{{xu}} = {c_xu:.4f}$ \\"),
        r"\hline",
        r"\end{tabular}",
    ]
    tab_path = os.path.join(OUTDIR, "lqc_fvi_fqi_weights.tex")
    with open(tab_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Table: {tab_path}")

    # ── Figure (3 panels) ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: FVI/FQI convergence
    ax = axes[0]
    ax.semilogy(range(1, len(fvi_errs) + 1), fvi_errs,
                label='FVI', color=COLORS['blue'], lw=2)
    ax.semilogy(range(1, len(fqi_errs) + 1), fqi_errs,
                label='FQI', color=COLORS['red'], lw=2, ls='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\|V_k - V^*\|_\infty$')
    ax.set_title('FVI and FQI Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: DQN learning curve (mean ± SE across DQN_N_SEEDS seeds)
    ax = axes[1]
    steps_k = np.array(dqn_step_log) / 1000.0   # scale to thousands
    ax.semilogy(steps_k, dqn_err_mean,
                color=ALGO_COLORS['DQN'], lw=2,
                label=f'mean over {dqn_n_seeds} seeds')
    if np.any(dqn_err_se > 0):
        lo = np.clip(dqn_err_mean - dqn_err_se, 1e-12, None)
        hi = dqn_err_mean + dqn_err_se
        ax.fill_between(steps_k, lo, hi, color=ALGO_COLORS['DQN'], alpha=0.25,
                        label='± 1 SE')
    ax.set_xlabel('Gradient steps (thousands)')
    ax.set_ylabel(r'$\|V_{\mathrm{DQN}} - V^*\|_\infty$')
    ax.set_title(f'DQN Learning Curve ({dqn_n_seeds} seeds)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Value function recovery
    ax = axes[2]
    ax.plot(X, V_star, color=COLORS['black'], linestyle='-', lw=2,  label=r'$V^*$ (Riccati, $P={:.3f}$)'.format(P))
    ax.plot(X, V_fvi,  '--',  color=COLORS['blue'], lw=2,
            label=f'FVI  (error {fvi_error:.1e})')
    ax.plot(X, V_fqi,  ':',   color=COLORS['red'], lw=2,
            label=f'FQI  (error {fqi_error:.1e})')
    ax.plot(X, V_dqn,  '-.',  color=ALGO_COLORS['DQN'], lw=2,
            label=f'DQN  (mean error {dqn_err_an:.1e}$\\pm${dqn_err_an_se:.0e}, n={dqn_n_seeds})')
    ax.set_xlabel('State $x$')
    ax.set_ylabel('Value $V(x)$')
    ax.set_title('Value Function Recovery')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(OUTDIR, 'lqc_fvi_fqi.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LQC: Fitted Value Iteration vs Fitted Q-Iteration vs DQN')
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)

    print("=" * 60)
    print("LQC Fitted Value Iteration vs Fitted Q-Iteration vs DQN")
    print(f"  a={a}, b={b}, gamma={gamma}")
    print(f"  State grid: {N_X} pts, step={h_X:.4f}")
    print(f"  Action grid: {N_U} pts, step={h_U:.4f}")
    print(f"  DQN: {DQN_STEPS} steps, hidden={DQN_HIDDEN}, lr={DQN_LR}, reward_scale={REWARD_SCALE}")
    print("=" * 60)

    print(f"\nRiccati P = {P:.6f}")
    print(f"V*(x)   = {-P:.4f}*x^2")
    print(f"Q*(x,u) = {c_xx:.4f}*x^2 + {c_xu:.4f}*xu + {c_uu:.4f}*u^2")
    print(f"Optimal gain K = {K_opt:.4f}  =>  closed-loop x' = {a+b*K_opt:.4f}*x")

    assert Xnext.min() >= X[0] - 1e-10 and Xnext.max() <= X[-1] + 1e-10, \
        f"Grid not invariant: [{Xnext.min():.3f}, {Xnext.max():.3f}]"
    print(f"\nGrid invariant: x' in [{Xnext.min():.4f}, {Xnext.max():.4f}]")

    if force:
        print(f"Force recompute: {sorted(force)}")

    if args.plots_only:
        data = compute_data()  # all cache hits
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
