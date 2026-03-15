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
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, CMAP_SEQ
from sims.sim_cache import load_results, save_results, add_cache_args
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
MAX_ITER = 500
TOL      = 1e-9
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

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'lqc_fvi_fqi'
CONFIG = {
    'version': 1,
    'a': a, 'b': b, 'gamma': gamma,
    'N_X': N_X, 'N_U': N_U,
    'MAX_ITER': MAX_ITER, 'TOL': TOL,
    'DQN_HIDDEN': DQN_HIDDEN, 'DQN_LR': DQN_LR,
    'DQN_BUFFER': DQN_BUFFER, 'DQN_BATCH': DQN_BATCH,
    'DQN_TARGET_UP': DQN_TARGET_UP, 'DQN_STEPS': DQN_STEPS,
    'DQN_EPS_START': DQN_EPS_START, 'DQN_EPS_END': DQN_EPS_END,
    'DQN_EPS_DECAY': DQN_EPS_DECAY, 'DQN_EVAL_INT': DQN_EVAL_INT,
    'REWARD_SCALE': REWARD_SCALE,
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


# ── compute_data ──────────────────────────────────────────────────────────────

def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

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

    # ── Exact VI ───────────────────────────────────────────────────────────────
    V_exact  = np.zeros(N_X)
    vi_iters = 0
    for _ in range(20000):
        Vnext   = np.interp(Xnext, X, V_exact)
        V_new   = (R + gamma * Vnext).max(axis=1)
        delta   = np.max(np.abs(V_new - V_exact))
        V_exact = V_new
        vi_iters += 1
        if delta < TOL / 10.0:
            break

    vi_vs_analytical = np.max(np.abs(V_exact - V_star))
    print(f"\nExact VI: {vi_iters} iters, vs analytical V*: {vi_vs_analytical:.2e}")

    # ── FVI ────────────────────────────────────────────────────────────────────
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

    # ── FQI ────────────────────────────────────────────────────────────────────
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

    # ── DQN ────────────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    random.seed(42)

    online = QNet()
    target = QNet()
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=DQN_LR)

    buf = ReplayBuffer(DQN_BUFFER)
    dqn_step_log = []
    dqn_err_log  = []

    x_idx = np.random.randint(N_X)   # start at random state

    print(f"\nDQN training ({DQN_STEPS} steps)...")
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

        # Reset to random state every 20 steps (optimal policy converges to x≈0 in ~5 steps;
        # frequent resets ensure buffer covers large |x| states throughout training)
        if step % 20 == 0:
            x_idx = np.random.randint(N_X)

        if len(buf) < DQN_BATCH:
            continue

        # Sample minibatch
        xi, ui, r_b, xpi = buf.sample(DQN_BATCH)
        x_t  = torch.tensor(X[xi, np.newaxis] / 4.0, dtype=torch.float32)    # (B, 1)
        xp_t = torch.tensor(X[xpi, np.newaxis] / 4.0, dtype=torch.float32)   # (B, 1)
        r_t  = torch.tensor(r_b, dtype=torch.float32)                         # (B,)
        ui_t = torch.tensor(ui, dtype=torch.long)                             # (B,)

        with torch.no_grad():
            max_q_next = target(xp_t).max(dim=1).values   # (B,)
        y = r_t + gamma * max_q_next

        q_pred = online(x_t).gather(1, ui_t.unsqueeze(1)).squeeze(1)   # (B,)
        loss   = nn.functional.mse_loss(q_pred, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        opt.step()

        if step % DQN_TARGET_UP == 0:
            target.load_state_dict(online.state_dict())

        if step % DQN_EVAL_INT == 0:
            with torch.no_grad():
                x_all = torch.tensor(X[:, np.newaxis] / 4.0, dtype=torch.float32)  # (N_X, 1)
                V_dqn_now = online(x_all).max(dim=1).values.numpy() * REWARD_SCALE  # rescaled
            err = np.max(np.abs(V_dqn_now - V_star))
            dqn_step_log.append(step)
            dqn_err_log.append(err)
            if step % 10_000 == 0:
                print(f"  step {step:6d}, eps={eps:.3f}, err vs V*={err:.4f}")

    # Final DQN value function (rescale back to original units)
    with torch.no_grad():
        x_all = torch.tensor(X[:, np.newaxis] / 4.0, dtype=torch.float32)
        V_dqn = online(x_all).max(dim=1).values.numpy() * REWARD_SCALE
    dqn_err_an = np.max(np.abs(V_dqn - V_star))
    print(f"\nDQN: {DQN_STEPS} steps, error vs analytical V*: {dqn_err_an:.2e}")

    # ── Summary ────────────────────────────────────────────────────────────────
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
    print(f"{'Analytical':<18} {'---':>6} {'---':>10} {'0':>10} "
          f"{P:>10.4f} {c_xx:>8.4f} {c_xu:>8.4f} {c_uu:>8.4f}")
    print(f"{'='*78}")

    # ── Verification ───────────────────────────────────────────────────────────
    # Primary check: error vs analytical V* (measures regression quality, not VI grid error)
    assert fvi_err_an < 0.001, f"FVI vs analytical: {fvi_err_an:.6f} exceeds 0.001"
    assert fqi_err_an < 0.001, f"FQI vs analytical: {fqi_err_an:.6f} exceeds 0.001"
    assert dqn_err_an < 1.0,   f"DQN error {dqn_err_an:.4f} exceeds 1.0"
    # Secondary check: coefficient recovery
    assert abs(-theta_V[1] - P) < 0.001, \
        f"FVI P recovery: {-theta_V[1]:.6f} vs {P:.6f}"
    assert abs(theta_Q[1] - c_xx) < 0.002, \
        f"FQI c_xx recovery: {theta_Q[1]:.5f} vs {c_xx:.5f}"
    assert abs(theta_Q[4] - c_xu) < 0.002, \
        f"FQI c_xu recovery: {theta_Q[4]:.5f} vs {c_xu:.5f}"
    assert abs(theta_Q[3] - c_uu) < 0.002, \
        f"FQI c_uu recovery: {theta_Q[3]:.5f} vs {c_uu:.5f}"
    print("\nAll verification checks passed.")

    data = {
        'vi_iters': vi_iters,
        'vi_vs_analytical': vi_vs_analytical,
        'fvi_errs': fvi_errs,
        'fvi_error': fvi_error,
        'fvi_err_an': fvi_err_an,
        'fvi_iters': fvi_iters,
        'theta_V': theta_V.tolist(),
        'V_fvi': V_fvi.tolist(),
        'fqi_errs': fqi_errs,
        'fqi_error': fqi_error,
        'fqi_err_an': fqi_err_an,
        'fqi_iters': fqi_iters,
        'theta_Q': theta_Q.tolist(),
        'V_fqi': V_fqi.tolist(),
        'dqn_step_log': dqn_step_log,
        'dqn_err_log': dqn_err_log,
        'dqn_err_an': dqn_err_an,
        'V_dqn': V_dqn.tolist(),
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ── generate_outputs ──────────────────────────────────────────────────────────

def generate_outputs(data):
    vi_iters = data['vi_iters']
    vi_vs_analytical = data['vi_vs_analytical']
    fvi_errs = data['fvi_errs']
    fvi_error = data['fvi_error']
    fvi_err_an = data['fvi_err_an']
    fvi_iters = data['fvi_iters']
    theta_V = np.array(data['theta_V'])
    V_fvi = np.array(data['V_fvi'])
    fqi_errs = data['fqi_errs']
    fqi_error = data['fqi_error']
    fqi_err_an = data['fqi_err_an']
    fqi_iters = data['fqi_iters']
    theta_Q = np.array(data['theta_Q'])
    V_fqi = np.array(data['V_fqi'])
    dqn_step_log = data['dqn_step_log']
    dqn_err_log = data['dqn_err_log']
    dqn_err_an = data['dqn_err_an']
    V_dqn = np.array(data['V_dqn'])

    # ── LaTeX table ────────────────────────────────────────────────────────────
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
        (rf"DQN ($2 \times 64$ ReLU) & {DQN_STEPS} & {dqn_err_an:.2e} & --- & --- \\"),
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

    # Panel 2: DQN learning curve
    ax = axes[1]
    steps_k = np.array(dqn_step_log) / 1000.0   # scale to thousands
    ax.semilogy(steps_k, dqn_err_log,
                color=ALGO_COLORS['DQN'], lw=2)
    ax.set_xlabel('Gradient steps (thousands)')
    ax.set_ylabel(r'$\|V_{\mathrm{DQN}} - V^*\|_\infty$')
    ax.set_title('DQN Learning Curve')
    ax.grid(True, alpha=0.3)

    # Panel 3: Value function recovery
    ax = axes[2]
    ax.plot(X, V_star, color=COLORS['black'], linestyle='-', lw=2,  label=r'$V^*$ (Riccati, $P={:.3f}$)'.format(P))
    ax.plot(X, V_fvi,  '--',  color=COLORS['blue'], lw=2,
            label=f'FVI  (error {fvi_error:.1e})')
    ax.plot(X, V_fqi,  ':',   color=COLORS['red'], lw=2,
            label=f'FQI  (error {fqi_error:.1e})')
    ax.plot(X, V_dqn,  '-.',  color=ALGO_COLORS['DQN'], lw=2,
            label=f'DQN  (error {dqn_err_an:.1e})')
    ax.set_xlabel('State $x$')
    ax.set_ylabel('Value $V(x)$')
    ax.set_title('Value Function Recovery')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(OUTDIR, 'lqc_fvi_fqi.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure: {fig_path}")


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
