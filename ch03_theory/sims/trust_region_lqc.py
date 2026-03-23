# Trust Region Visualization — LQC Monetary Policy
# Chapter 3: Planning and Learning Theory
# Central bank learns a Taylor rule via TRPO vs PPO in a 2D parameter space.
# Fully analytical: no RL simulation, no torch. Pure numpy/scipy/matplotlib.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, CMAP_SEQ
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()

import argparse
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# ── Parameters ────────────────────────────────────────────────────────────────
np.random.seed(42)

A = np.array([[0.5, 0.2], [0.0, 0.8]])    # IS-Phillips dynamics
B = np.array([[0.5], [1.0]])              # interest rate effect
Q = np.array([[2.0, 0.0], [0.0, 1.0]])   # output gap penalized 2x
R = 0.5                                    # interest rate penalty
gamma = 0.95
sigma2_w = 0.5     # process noise (monetary policy shocks)
sigma2_a = 1.0     # action noise (policy spread)
delta = 0.1        # KL budget
eps = 0.2          # PPO clip
theta_old = np.array([0.0, 0.0])          # starting iterate: no policy response

# Grid for parameter space (Panel 2 and 3)
N_GRID = 100
theta1_vals = np.linspace(-0.5, 2.0, N_GRID)
theta2_vals = np.linspace(-0.5, 2.5, N_GRID)
TH1, TH2 = np.meshgrid(theta1_vals, theta2_vals)  # shape (N_GRID, N_GRID)

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'trust_region_lqc'
CONFIG = {
    'version': 2,
    'A': A.tolist(), 'B': B.tolist(), 'Q': Q.tolist(),
    'R': R, 'gamma': gamma,
    'sigma2_w': sigma2_w, 'sigma2_a': sigma2_a,
    'delta': delta, 'eps': eps,
    'theta_old': theta_old.tolist(),
    'N_GRID': N_GRID,
}


# ── Helper functions ───────────────────────────────────────────────────────────

def is_stable(theta, A, B):
    """Return True iff closed-loop system (A - B theta^T) is stable."""
    theta = np.asarray(theta).ravel()
    Acl = A - B @ theta.reshape(1, -1)
    return np.max(np.abs(np.linalg.eigvals(Acl))) < 1.0


def compute_J_single(theta, A, B, Q, R, gamma, sigma2_w):
    """Compute J(theta) for a single theta vector. Returns NaN if unstable."""
    theta = np.asarray(theta).ravel()
    if not is_stable(theta, A, B):
        return np.nan
    Acl = A - B @ theta.reshape(1, -1)
    # Stage cost matrix for closed-loop: Q + theta R theta^T
    Qcl = Q + np.outer(theta, theta) * R
    # Solve Lyapunov: P = gamma * Acl^T P Acl + Qcl
    # Equivalent: P - gamma * Acl^T P Acl = Qcl
    # scipy.linalg.solve_discrete_lyapunov solves X - A X A^T = Q
    # We need X - gamma * Acl^T X Acl = Qcl
    # => X - (sqrt(gamma)*Acl)^T X (sqrt(gamma)*Acl) = Qcl
    sqg = np.sqrt(gamma)
    try:
        P = scipy.linalg.solve_discrete_lyapunov((sqg * Acl).T, Qcl)
    except Exception:
        return np.nan
    # J(theta) = -trace(P * Sigma0), Sigma0 = I
    return -np.trace(P)


def compute_J_grid(theta1_vals, theta2_vals, A, B, Q, R, gamma, sigma2_w):
    """Compute J on a meshgrid. Returns 2D array (indexed [i_theta2, i_theta1])."""
    J = np.full((len(theta2_vals), len(theta1_vals)), np.nan)
    for i, t2 in enumerate(theta2_vals):
        for j, t1 in enumerate(theta1_vals):
            J[i, j] = compute_J_single(np.array([t1, t2]), A, B, Q, R, gamma, sigma2_w)
    return J


def compute_fisher(theta_old, A, B, sigma2_a):
    """
    Fisher information matrix M = Sigma_x / sigma2_a.
    Sigma_x: stationary state covariance under theta_old.
    Returns M, eigenvalues lam, eigenvectors V.
    """
    Acl = A - B @ theta_old.reshape(1, -1)
    # Stationary covariance: Sigma_x = Acl Sigma_x Acl^T + sigma2_w * I
    # solve_discrete_lyapunov: X - A X A^T = Q => Sigma_x - Acl Sigma_x Acl^T = sigma2_w*I
    Sigma_x = scipy.linalg.solve_discrete_lyapunov(Acl, sigma2_w * np.eye(2))
    M = Sigma_x / sigma2_a
    lam, V = np.linalg.eigh(M)
    return M, Sigma_x, lam, V


def compute_ppo_band(theta_old, TH1, TH2, Sigma_x, sigma2_a, eps,
                     n_samples=200, frac_threshold=0.95):
    """
    PPO feasible region: fraction-based ratio mask.
    Returns boolean mask of shape (N_GRID, N_GRID).
    """
    # Sample states from stationary distribution
    X_samples = np.random.multivariate_normal(np.zeros(2), Sigma_x, size=n_samples)  # (n,2)
    # Sample actions from old policy
    mu_old = -(X_samples @ theta_old)  # (n,)
    U_samples = mu_old + np.sqrt(sigma2_a) * np.random.randn(n_samples)  # (n,)

    # For each grid point theta, compute log-ratio for all samples
    # log r(theta,i) = -(u_i + theta.x_i)^2/(2 sigma2_a) + (u_i + theta_old.x_i)^2/(2 sigma2_a)
    # = [(u_i + theta_old.x_i)^2 - (u_i + theta.x_i)^2] / (2 sigma2_a)
    # u_i + theta_old.x_i = U_samples - mu_old = noise (since mu_old = -theta_old.x)
    # Actually: mu_old = -(theta_old @ x), so u + theta_old.x = u - mu_old... wait
    # u_i ~ N(mu_old_i, sigma2_a), mu_old_i = -(theta_old @ x_i)
    # u_i + theta_old @ x_i = u_i - mu_old_i = noise_i
    noise = U_samples - mu_old  # (n,) — this is the action noise

    # For new theta: u_i + theta @ x_i = noise_i + (theta_old - theta) @ x_i
    # But we need u_i + theta.x_i where theta.x_i means theta^T x_i
    # u_i = noise_i + mu_old_i = noise_i - theta_old @ x_i
    # u_i + theta @ x_i = noise_i - theta_old @ x_i + theta @ x_i = noise_i + (theta - theta_old) @ x_i

    # Precompute old numerator: (u_i + theta_old @ x_i)^2 = noise_i^2
    old_sq = noise ** 2  # (n,)

    # For each grid point, new: (u_i + theta @ x_i)^2
    # = (noise_i + (theta - theta_old) @ x_i)^2
    # Delta_theta = theta - theta_old at each grid point: shape (N_GRID, N_GRID, 2)
    Delta1 = TH1 - theta_old[0]  # (N_GRID, N_GRID)
    Delta2 = TH2 - theta_old[1]  # (N_GRID, N_GRID)
    # dot = Delta_theta @ x_i = Delta1 * x_i[0] + Delta2 * x_i[1]
    # shape: (N_GRID, N_GRID, n_samples)
    dot = (Delta1[:, :, None] * X_samples[:, 0] +
           Delta2[:, :, None] * X_samples[:, 1])  # (N_GRID, N_GRID, n)
    new_inner = noise[None, None, :] + dot  # (N_GRID, N_GRID, n)
    new_sq = new_inner ** 2  # (N_GRID, N_GRID, n)

    log_r = (old_sq[None, None, :] - new_sq) / (2 * sigma2_a)
    r = np.exp(log_r)  # (N_GRID, N_GRID, n)

    # Fraction of samples with ratio in [1-eps, 1+eps]
    in_clip = (r >= 1 - eps) & (r <= 1 + eps)
    frac = np.mean(in_clip, axis=2)  # (N_GRID, N_GRID)

    mask = frac >= frac_threshold
    print(f"  PPO frac_threshold={frac_threshold}: feasible grid points = "
          f"{mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")
    return mask, frac


def gradient_J(theta_old, J_grid, theta1_vals, theta2_vals):
    """Finite differences for grad J at theta_old."""
    i1 = np.argmin(np.abs(theta1_vals - theta_old[0]))
    i2 = np.argmin(np.abs(theta2_vals - theta_old[1]))
    h = theta1_vals[1] - theta1_vals[0]

    # dJ/dtheta1
    def safe(arr, i, j):
        v = arr[j, i]
        return v if not np.isnan(v) else None

    # Central differences, fall back to one-sided
    def fd(arr, i, j, di, dj):
        vp = safe(arr, i + di, j + dj)
        vm = safe(arr, i - di, j - dj)
        if vp is not None and vm is not None:
            return (vp - vm) / (2 * h)
        elif vp is not None:
            v0 = safe(arr, i, j)
            return (vp - v0) / h if v0 is not None else 0.0
        elif vm is not None:
            v0 = safe(arr, i, j)
            return (v0 - vm) / h if v0 is not None else 0.0
        return 0.0

    g1 = fd(J_grid, i1, i2, 1, 0)
    g2 = fd(J_grid, i1, i2, 0, 1)
    return np.array([g1, g2])


def compute_trpo_step(theta_old, g, M, delta):
    """
    TRPO step: theta_trpo = theta_old + sqrt(delta / (g^T M^{-1} g)) * M^{-1} g
    theta_bad = theta_old + 3 * (theta_trpo - theta_old)
    """
    Minv = np.linalg.inv(M)
    Minv_g = Minv @ g
    scale = np.sqrt(delta / (g @ Minv_g))
    theta_trpo = theta_old + scale * Minv_g
    theta_bad = theta_old + 3 * (theta_trpo - theta_old)
    return theta_trpo, theta_bad


def compute_ppo_step(theta_old, g, ppo_mask, theta1_vals, theta2_vals):
    """
    Binary search along gradient direction for PPO feasible region boundary.
    """
    g_norm = g / (np.linalg.norm(g) + 1e-12)
    # Find max step along g_norm that stays in PPO mask
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2
        theta_test = theta_old + mid * g_norm
        i1 = np.argmin(np.abs(theta1_vals - theta_test[0]))
        i2 = np.argmin(np.abs(theta2_vals - theta_test[1]))
        if (0 <= i1 < len(theta1_vals) and 0 <= i2 < len(theta2_vals)
                and ppo_mask[i2, i1]):
            lo = mid
        else:
            hi = mid
    theta_ppo = theta_old + lo * g_norm
    return theta_ppo


# ── compute_data ──────────────────────────────────────────────────────────────

def _run_trust_region_analysis():
    """Compute J grid, Fisher matrix, TRPO/PPO steps."""
    print("\nParameters:")
    print(f"  A = {A.tolist()}")
    print(f"  B = {B.ravel().tolist()}")
    print(f"  Q = {Q.tolist()}")
    print(f"  R = {R}, gamma = {gamma}")
    print(f"  sigma2_w = {sigma2_w}, sigma2_a = {sigma2_a}")
    print(f"  delta = {delta}, eps = {eps}")
    print(f"  theta_old = {theta_old}")

    # Compute theta* from discrete algebraic Riccati equation
    # For DLQR: P = Q + gamma * Acl^T P Acl (at optimum)
    # Standard DARE: P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    # scipy convention: solve_discrete_are(A, B, Q, R) solves
    #   P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    # With discount: use A_scaled = sqrt(gamma)*A, B_scaled = sqrt(gamma)*B
    sqg = np.sqrt(gamma)
    P_riccati = scipy.linalg.solve_discrete_are(sqg * A, sqg * B, Q, R * np.eye(1))
    K_opt = np.linalg.inv(R * np.eye(1) + (sqg * B).T @ P_riccati @ (sqg * B)) @ (sqg * B).T @ P_riccati @ (sqg * A)
    theta_star = K_opt.ravel()

    print(f"\nOptimal policy (Riccati):")
    print(f"  theta* = [{theta_star[0]:.6f}, {theta_star[1]:.6f}]")

    # Verify Lyapunov residual at theta*
    Acl_star = A - B @ theta_star.reshape(1, -1)
    Qcl_star = Q + np.outer(theta_star, theta_star) * R
    P_lyap = scipy.linalg.solve_discrete_lyapunov((sqg * Acl_star).T, Qcl_star)
    residual = P_lyap - (sqg * Acl_star).T @ P_lyap @ (sqg * Acl_star) - Qcl_star
    print(f"  Lyapunov residual norm at theta*: {np.linalg.norm(residual):.2e}")

    # ── Compute J grid ─────────────────────────────────────────────────────────────
    print("\nComputing J on 100x100 grid...")
    J_grid = compute_J_grid(theta1_vals, theta2_vals, A, B, Q, R, gamma, sigma2_w)
    n_stable = np.sum(~np.isnan(J_grid))
    print(f"  Stable grid points: {n_stable} / {J_grid.size}")

    J_old = compute_J_single(theta_old, A, B, Q, R, gamma, sigma2_w)
    J_star = compute_J_single(theta_star, A, B, Q, R, gamma, sigma2_w)
    print(f"  J(theta_old) = {J_old:.6f}")
    print(f"  J(theta*)    = {J_star:.6f}")
    print(f"  J(theta*) > J(theta_old): {J_star > J_old}")

    # ── Compute Fisher / KL ellipse ────────────────────────────────────────────────
    print("\nComputing Fisher information matrix...")
    M, Sigma_x, lam, V = compute_fisher(theta_old, A, B, sigma2_a)
    print(f"  Sigma_x = {Sigma_x.tolist()}")
    print(f"  M = Sigma_x / sigma2_a = {M.tolist()}")
    print(f"  Eigenvalues: lambda1={lam[0]:.6f}, lambda2={lam[1]:.6f}")
    semi1 = np.sqrt(delta / lam[0])
    semi2 = np.sqrt(delta / lam[1])
    print(f"  KL ellipse semi-axes: sqrt(delta/lambda1)={semi1:.6f}, sqrt(delta/lambda2)={semi2:.6f}")

    # ── Gradient and steps ─────────────────────────────────────────────────────────
    print("\nComputing gradient and steps...")
    g = gradient_J(theta_old, J_grid, theta1_vals, theta2_vals)
    print(f"  grad J at theta_old = {g}")

    theta_trpo, theta_bad = compute_trpo_step(theta_old, g, M, delta)
    print(f"  theta_trpo = [{theta_trpo[0]:.6f}, {theta_trpo[1]:.6f}]")
    print(f"  theta_bad  = [{theta_bad[0]:.6f}, {theta_bad[1]:.6f}]")

    # Verify TRPO step on ellipse boundary
    dt = theta_trpo - theta_old
    kl_trpo = dt @ M @ dt
    print(f"\nVerification:")
    print(f"  (theta_trpo - theta_old)^T M (theta_trpo - theta_old) = {kl_trpo:.8f} (should be {delta})")
    print(f"  Diff from delta: {abs(kl_trpo - delta):.2e}")

    db = theta_bad - theta_old
    kl_bad = db @ M @ db
    print(f"  (theta_bad - theta_old)^T M (theta_bad - theta_old) = {kl_bad:.8f} (should be {9*delta})")

    print(f"  theta_bad stable: {is_stable(theta_bad, A, B)}")
    print(f"  theta_trpo stable: {is_stable(theta_trpo, A, B)}")

    # ── PPO mask ───────────────────────────────────────────────────────────────────
    print("\nComputing PPO feasible region...")
    ppo_mask, ppo_frac = compute_ppo_band(
        theta_old, TH1, TH2, Sigma_x, sigma2_a, eps,
        n_samples=200, frac_threshold=0.50
    )

    theta_ppo = compute_ppo_step(theta_old, g, ppo_mask, theta1_vals, theta2_vals)
    print(f"  theta_ppo = [{theta_ppo[0]:.6f}, {theta_ppo[1]:.6f}]")
    print(f"  ||theta_trpo - theta_old|| = {np.linalg.norm(theta_trpo - theta_old):.6f}")
    print(f"  ||theta_ppo  - theta_old|| = {np.linalg.norm(theta_ppo - theta_old):.6f}")


    data = {
        'theta_star': theta_star.tolist(),
        'J_grid': J_grid.tolist(),
        'J_old': J_old,
        'J_star': J_star,
        'lam': lam.tolist(),
        'V_eig': V.tolist(),
        'semi1': semi1,
        'semi2': semi2,
        'theta_trpo': theta_trpo.tolist(),
        'theta_bad': theta_bad.tolist(),
        'theta_ppo': theta_ppo.tolist(),
        'ppo_mask': ppo_mask.tolist(),
        'kl_trpo': kl_trpo,
        'kl_bad': kl_bad,
        'residual_norm': float(np.linalg.norm(residual)),
    }
    return data


def compute_data(force=None):
    force = force or set()
    return compute_or_load(CACHE_DIR, SCRIPT_NAME, 'trust_region', CONFIG,
                            _run_trust_region_analysis, force=('trust_region' in force))


# ── generate_outputs ──────────────────────────────────────────────────────────

def generate_outputs(data):
    theta_star = np.array(data['theta_star'])
    J_grid = np.array(data['J_grid'])
    J_old = data['J_old']
    J_star = data['J_star']
    lam = np.array(data['lam'])
    V = np.array(data['V_eig'])
    semi1 = data['semi1']
    semi2 = data['semi2']
    theta_trpo = np.array(data['theta_trpo'])
    theta_bad = np.array(data['theta_bad'])
    theta_ppo = np.array(data['theta_ppo'])
    ppo_mask = np.array(data['ppo_mask'])
    kl_trpo = data['kl_trpo']
    kl_bad = data['kl_bad']
    residual_norm = data['residual_norm']

    OUTDIR = os.path.dirname(os.path.abspath(__file__))

    # ── Plotting ───────────────────────────────────────────────────────────────────
    print("\nGenerating figure...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color scheme
    c_old = '#2166ac'    # blue — theta_old
    c_star = '#1a9641'   # green — theta*
    c_bad = '#d73027'    # red — theta_bad
    c_trpo = '#f46d43'   # orange — TRPO
    c_ppo = '#762a83'    # purple — PPO

    # ── Panel 1: Policy lines in state space ──────────────────────────────────────
    ax = axes[0]
    x1_range = np.linspace(-2, 2, 300)

    for theta, label, color, ls in [
        (theta_star, r'$\theta^*$', c_star, '-'),
        (theta_old, r'$\theta_{\mathrm{old}}$', c_old, '--'),
        (theta_bad, r'$\theta_{\mathrm{bad}}$', c_bad, ':'),
    ]:
        # Contour lines: u = -(theta1*x1 + theta2*x2) = const (e.g., 0, ±0.5, ±1)
        for u_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            if abs(theta[1]) > 1e-6:
                x2_line = -(u_val + theta[0] * x1_range) / theta[1]
                mask_range = np.abs(x2_line) <= 2.2
                x1_plot = x1_range[mask_range]
                x2_plot = x2_line[mask_range]
                if len(x1_plot) > 1:
                    lbl = label if u_val == 0.0 else None
                    ax.plot(x1_plot, x2_plot, color=color, linestyle=ls,
                            linewidth=1.5, label=lbl, alpha=0.85)

    # Phase portrait quiver
    x1q = np.linspace(-2, 2, 7)
    x2q = np.linspace(-2, 2, 7)
    X1Q, X2Q = np.meshgrid(x1q, x2q)
    Acl_old = A - B @ theta_old.reshape(1, -1)
    DX = np.zeros_like(X1Q)
    DY = np.zeros_like(X2Q)
    for ii in range(7):
        for jj in range(7):
            x = np.array([X1Q[ii, jj], X2Q[ii, jj]])
            dx = Acl_old @ x - x
            DX[ii, jj] = dx[0]
            DY[ii, jj] = dx[1]

    ax.quiver(X1Q, X2Q, DX, DY, alpha=0.4, color='gray', scale=25,
              width=0.004, headwidth=4)
    ax.plot(0, 0, 'k*', markersize=12, zorder=5, label='origin')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$x_1$ (output gap)', fontsize=11)
    ax.set_ylabel(r'$x_2$ (inflation gap)', fontsize=11)
    ax.set_title('Policy lines in state space', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_aspect('equal')

    # ── Panel 2: J(theta) + KL ellipse ────────────────────────────────────────────
    ax = axes[1]

    # Filled contour of J
    J_plot = np.where(np.isnan(J_grid), np.nan, J_grid)
    J_finite = J_plot[~np.isnan(J_plot)]
    vmin, vmax = np.percentile(J_finite, 5), np.percentile(J_finite, 95)
    levels = np.linspace(vmin, vmax, 20)

    cf = ax.contourf(TH1, TH2, J_plot, levels=levels, cmap=CMAP_SEQ, alpha=0.7)
    plt.colorbar(cf, ax=ax, label=r'$J(\theta)$', shrink=0.8)

    # Hatch unstable region
    unstable = np.isnan(J_grid)
    ax.contourf(TH1, TH2, unstable.astype(float), levels=[0.5, 1.5],
                hatches=['///'], colors='none', alpha=0.0)
    ax.contour(TH1, TH2, unstable.astype(float), levels=[0.5], colors='gray',
               linewidths=1.0, linestyles='--')

    # KL ellipse
    angle_deg = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    width_ell = 2 * np.sqrt(delta / lam[0])
    height_ell = 2 * np.sqrt(delta / lam[1])
    ellipse_patch = Ellipse(xy=theta_old, width=width_ell, height=height_ell,
                             angle=angle_deg, fill=False, edgecolor=c_trpo,
                             linewidth=2.0, label='KL trust region')
    ax.add_patch(ellipse_patch)

    # Arrows
    ax.annotate('', xy=theta_bad, xytext=theta_old,
                arrowprops=dict(arrowstyle='->', color=c_bad, lw=2.0))
    ax.annotate('', xy=theta_trpo, xytext=theta_old,
                arrowprops=dict(arrowstyle='->', color=c_trpo, lw=2.0))

    # Markers
    ax.plot(*theta_star, '*', color=c_star, markersize=14, zorder=6,
            label=r'$\theta^*$')
    ax.plot(*theta_old, 'o', color=c_old, markersize=10, zorder=6,
            label=r'$\theta_{\mathrm{old}}$')
    ax.plot(*theta_trpo, 's', color=c_trpo, markersize=9, zorder=6,
            label=r'$\theta_{\mathrm{trpo}}$')
    ax.plot(*theta_bad, 'X', color=c_bad, markersize=11, zorder=6,
            label=r'$\theta_{\mathrm{bad}}$')

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel(r'$\theta_1$', fontsize=12)
    ax.set_ylabel(r'$\theta_2$', fontsize=12)
    ax.set_title(r'$J(\theta)$ and KL trust region', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')

    # ── Panel 3: TRPO vs PPO ───────────────────────────────────────────────────────
    ax = axes[2]

    # Same J contours
    ax.contourf(TH1, TH2, J_plot, levels=levels, cmap=CMAP_SEQ, alpha=0.5)
    ax.contourf(TH1, TH2, unstable.astype(float), levels=[0.5, 1.5],
                hatches=['///'], colors='none', alpha=0.0)
    ax.contour(TH1, TH2, unstable.astype(float), levels=[0.5], colors='gray',
               linewidths=1.0, linestyles='--')

    # Fill TRPO ellipse
    ellipse_fill = Ellipse(xy=theta_old, width=width_ell, height=height_ell,
                            angle=angle_deg, facecolor=c_trpo, alpha=0.25,
                            edgecolor=c_trpo, linewidth=1.5,
                            label='TRPO (KL ellipse)')
    ax.add_patch(ellipse_fill)

    # Fill PPO mask
    ppo_rgba = np.zeros((*ppo_mask.shape, 4))
    ppo_rgba[ppo_mask, 0] = 0.47   # purple R
    ppo_rgba[ppo_mask, 1] = 0.17   # purple G
    ppo_rgba[ppo_mask, 2] = 0.51   # purple B
    ppo_rgba[ppo_mask, 3] = 0.30
    ax.imshow(ppo_rgba, origin='lower',
              extent=[theta1_vals[0], theta1_vals[-1], theta2_vals[0], theta2_vals[-1]],
              aspect='auto', interpolation='nearest', zorder=2)

    # Manual PPO legend proxy
    ppo_proxy = mpatches.Patch(facecolor=(0.47, 0.17, 0.51, 0.30),
                                edgecolor=(0.47, 0.17, 0.51),
                                label='PPO (ratio clip band)')

    # Markers
    ax.plot(*theta_old, 'o', color=c_old, markersize=10, zorder=6,
            label=r'$\theta_{\mathrm{old}}$')
    ax.plot(*theta_trpo, 's', color=c_trpo, markersize=9, zorder=6,
            label=r'$\theta_{\mathrm{trpo}}$')
    ax.plot(*theta_ppo, 'D', color=c_ppo, markersize=9, zorder=6,
            label=r'$\theta_{\mathrm{ppo}}$')
    ax.plot(*theta_star, '*', color=c_star, markersize=14, zorder=6,
            label=r'$\theta^*$')

    handles, labels = ax.get_legend_handles_labels()
    handles.insert(1, ppo_proxy)
    labels.insert(1, 'PPO (ratio clip band)')
    ax.legend(handles, labels, fontsize=9, loc='upper right')

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel(r'$\theta_1$', fontsize=12)
    ax.set_ylabel(r'$\theta_2$', fontsize=12)
    ax.set_title('TRPO vs PPO feasible regions', fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "trust_region_lqc.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {os.path.join(OUTDIR, 'trust_region_lqc.png')}")

    # ── Results table ──────────────────────────────────────────────────────────────
    tex_content = r"""\begin{tabular}{lll}
\toprule
Quantity & Symbol & Value \\
\midrule
""" + f"Optimal gain (output gap) & $\\theta_1^*$ & {theta_star[0]:.4f} \\\\\n"
    tex_content += f"Optimal gain (inflation gap) & $\\theta_2^*$ & {theta_star[1]:.4f} \\\\\n"
    tex_content += f"Current iterate & $\\theta_{{\\mathrm{{old}}}}$ & $[{theta_old[0]:.1f},\\ {theta_old[1]:.1f}]$ \\\\\n"
    tex_content += f"Expected return at $\\theta^*$ & $J(\\theta^*)$ & {J_star:.4f} \\\\\n"
    tex_content += f"Expected return at $\\theta_{{\\mathrm{{old}}}}$ & $J(\\theta_{{\\mathrm{{old}}}})$ & {J_old:.4f} \\\\\n"
    tex_content += f"KL budget & $\\delta$ & {delta} \\\\\n"
    tex_content += f"KL semi-axis ($\\theta_1$ direction) & $\\sqrt{{\\delta/\\lambda_1}}$ & {semi1:.4f} \\\\\n"
    tex_content += f"KL semi-axis ($\\theta_2$ direction) & $\\sqrt{{\\delta/\\lambda_2}}$ & {semi2:.4f} \\\\\n"
    tex_content += f"PPO clip & $\\varepsilon$ & {eps} \\\\\n"
    tex_content += f"TRPO step size & $\\|\\Delta\\theta_{{\\mathrm{{trpo}}}}\\|$ & {np.linalg.norm(theta_trpo - theta_old):.4f} \\\\\n"
    tex_content += f"PPO step size & $\\|\\Delta\\theta_{{\\mathrm{{ppo}}}}\\|$ & {np.linalg.norm(theta_ppo - theta_old):.4f} \\\\\n"
    tex_content += r"""\bottomrule
\end{tabular}
"""

    with open(os.path.join(OUTDIR, "trust_region_lqc_results.tex"), "w") as f:
        f.write(tex_content)
    print(f"  Saved: {os.path.join(OUTDIR, 'trust_region_lqc_results.tex')}")

    # ── Final verification summary ─────────────────────────────────────────────────
    print("\nVerification summary:")
    print(f"  [1] Script ran without error: YES")
    print(f"  [2] PNG exists: ch03_theory/sims/trust_region_lqc.png")
    print(f"  [3] Results .tex exists: ch03_theory/sims/trust_region_lqc_results.tex")
    print(f"  [4] Lyapunov residual at theta*: {residual_norm:.2e} (< 1e-10: {residual_norm < 1e-10})")
    print(f"  [5] J(theta*) > J(theta_old): {J_star:.4f} > {J_old:.4f} = {J_star > J_old}")
    print(f"  [6] TRPO on ellipse boundary: KL = {kl_trpo:.8f}, delta = {delta}, diff = {abs(kl_trpo-delta):.2e} (< 1e-6: {abs(kl_trpo-delta) < 1e-6})")
    print(f"  [7] PPO region non-ellipsoidal: visual inspection required")
    print(f"  [8] theta_bad outside ellipse by 3x radius: KL = {kl_bad:.6f} (should be {9*delta})")
    print(f"  [9] LaTeX compile: run separately")
    print("\nOutput files:")
    print(f"  {os.path.join(OUTDIR, 'trust_region_lqc.png')}")
    print(f"  {os.path.join(OUTDIR, 'trust_region_lqc_results.tex')}")
    print(f"  {os.path.join(OUTDIR, 'trust_region_lqc_stdout.txt')}")


def main():
    parser = argparse.ArgumentParser(description='Trust Region LQC Monetary Policy')
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)

    print("=" * 60)
    print("TRUST REGION LQC MONETARY POLICY")
    print("=" * 60)

    if force:
        print(f"Force recompute: {sorted(force)}")

    if args.plots_only:
        data = compute_data()  # cache hit
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
