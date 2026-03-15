# Majorization-Minimization Surrogate for TRPO
# Chapter 3: Planning and Learning Theory
# Two-panel figure: single MM step + iterative MM convergence.

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sims'))
from plot_style import COLORS, FIG_DOUBLE, apply_style
apply_style()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Objective function ────────────────────────────────────────────────────────

def J(theta):
    return np.sin(2 * theta) + 0.3 * theta - 0.08 * theta**2

def J_prime(theta):
    return 2 * np.cos(2 * theta) + 0.3 - 0.16 * theta

def J_double_prime(theta):
    return -4 * np.sin(2 * theta) - 0.16


# ── Surrogate construction ────────────────────────────────────────────────────
# L(theta | theta_old) = J(theta_old) + J'(theta_old)(theta - theta_old) - c(theta - theta_old)^2
# Need c large enough that L(theta) <= J(theta) everywhere on visible range.
# This requires: J(theta) - J(theta_old) - J'(theta_old)(theta - theta_old) + c(theta - theta_old)^2 >= 0
# i.e., c >= max over theta of: [J(theta_old) + J'(theta_old)(theta-theta_old) - J(theta)] / (theta-theta_old)^2

def compute_c(theta_old, theta_grid):
    """Find minimum c such that L(theta|theta_old) <= J(theta) for all theta in grid."""
    d = theta_grid - theta_old
    mask = np.abs(d) > 1e-8
    numerator = J(theta_old) + J_prime(theta_old) * d - J(theta_grid)
    ratio = np.full_like(d, -np.inf)
    ratio[mask] = numerator[mask] / (d[mask]**2)
    c = max(np.max(ratio), 0.01)  # ensure positive
    return c * 1.05  # 5% safety margin

def surrogate(theta, theta_old, c):
    d = theta - theta_old
    return J(theta_old) + J_prime(theta_old) * d - c * d**2

def surrogate_argmax(theta_old, c, delta):
    """Argmax of surrogate within trust region [theta_old - delta, theta_old + delta]."""
    # Unconstrained max: theta_old + J'(theta_old) / (2c)
    theta_unc = theta_old + J_prime(theta_old) / (2 * c)
    return np.clip(theta_unc, theta_old - delta, theta_old + delta)


def generate_outputs():
    # ── MM iterations ─────────────────────────────────────────────────────────────

    theta_range = np.linspace(-1, 5, 1000)
    J_vals = J(theta_range)

    # Find global max numerically for reference
    theta_star = theta_range[np.argmax(J_vals)]

    delta_tr = 1.2  # trust region radius
    theta_0_single = 1.0  # for left panel
    theta_0_iter = 0.0    # for right panel
    n_iter = 4

    print("=" * 60)
    print("Majorization-Minimization Surrogate for TRPO")
    print("=" * 60)
    print(f"\nObjective: J(theta) = sin(2*theta) + 0.3*theta - 0.08*theta^2")
    print(f"Domain: [{theta_range[0]:.1f}, {theta_range[-1]:.1f}]")
    print(f"Global max: theta* = {theta_star:.4f}, J(theta*) = {J(theta_star):.4f}")
    print(f"Trust region radius: delta = {delta_tr}")

    # ── Left panel data ──────────────────────────────────────────────────────────
    c_single = compute_c(theta_0_single, theta_range)
    theta_new_single = surrogate_argmax(theta_0_single, c_single, delta_tr)
    L_at_new = surrogate(theta_new_single, theta_0_single, c_single)
    J_at_new = J(theta_new_single)

    print(f"\n--- Single MM Step (left panel) ---")
    print(f"theta_old = {theta_0_single:.4f}")
    print(f"c = {c_single:.4f}")
    print(f"theta_new = {theta_new_single:.4f}")
    print(f"J(theta_old) = {J(theta_0_single):.4f}")
    print(f"J(theta_new) = {J_at_new:.4f}")
    print(f"L(theta_new) = {L_at_new:.4f}")
    print(f"Guaranteed improvement gap: J(theta_new) - L(theta_new) = {J_at_new - L_at_new:.4f}")
    print(f"Actual improvement: J(theta_new) - J(theta_old) = {J_at_new - J(theta_0_single):.4f}")

    # Verify L <= J everywhere
    L_all = surrogate(theta_range, theta_0_single, c_single)
    violations = np.sum(L_all > J_vals + 1e-10)
    print(f"L <= J verification: {violations} violations out of {len(theta_range)} points")

    # ── Right panel data ─────────────────────────────────────────────────────────
    print(f"\n--- Iterative MM Convergence (right panel) ---")
    print(f"{'k':>3s} {'theta_k':>10s} {'J(theta_k)':>12s} {'Delta_J':>10s} {'c_k':>10s}")
    print("-" * 50)

    iterates = [theta_0_iter]
    surrogates_c = []
    print(f"  0   {theta_0_iter:10.4f}   {J(theta_0_iter):10.4f}          --         --")

    for k in range(n_iter):
        theta_k = iterates[-1]
        c_k = compute_c(theta_k, theta_range)
        surrogates_c.append(c_k)
        theta_next = surrogate_argmax(theta_k, c_k, delta_tr)
        delta_J = J(theta_next) - J(theta_k)
        iterates.append(theta_next)
        print(f"  {k+1}   {theta_next:10.4f}   {J(theta_next):10.4f}   {delta_J:10.4f}   {c_k:10.4f}")

        # Verify L <= J
        L_k = surrogate(theta_range, theta_k, c_k)
        viol = np.sum(L_k > J_vals + 1e-10)
        if viol > 0:
            print(f"  WARNING: {viol} violations of L <= J at iteration {k}")

    # Verify monotonic improvement
    J_iterates = [J(th) for th in iterates]
    monotonic = all(J_iterates[i+1] >= J_iterates[i] - 1e-10 for i in range(len(J_iterates)-1))
    print(f"\nMonotonic improvement: {'YES' if monotonic else 'NO'}")
    for i in range(len(iterates)):
        print(f"  J(theta_{i}) = {J_iterates[i]:.6f}")


    # ── Figure ────────────────────────────────────────────────────────────────────

    iter_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: Single MM step ───────────────────────────────────────────────

    ax1.plot(theta_range, J_vals, 'k-', linewidth=2, label=r'$J(\theta)$')

    # Surrogate
    L_single = surrogate(theta_range, theta_0_single, c_single)
    # Only plot surrogate in a reasonable range around theta_old
    surr_mask = np.abs(theta_range - theta_0_single) < 2.5
    ax1.plot(theta_range[surr_mask], L_single[surr_mask], '--', color=COLORS['blue'],
             linewidth=1.8, label=r'$L(\theta \mid \theta_{\mathrm{old}})$')

    # Trust region shading
    tr_mask = (theta_range >= theta_0_single - delta_tr) & (theta_range <= theta_0_single + delta_tr)
    ax1.fill_between(theta_range[tr_mask], ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -2,
                      np.max(J_vals) + 0.5,
                      alpha=0.08, color=COLORS['blue'])
    # Redraw after setting ylim
    ax1.set_ylim(np.min(J_vals) - 0.3, np.max(J_vals) + 0.5)
    # Re-shade
    ax1.fill_between(theta_range[tr_mask], np.min(J_vals) - 0.3, np.max(J_vals) + 0.5,
                      alpha=0.08, color=COLORS['blue'], label='Trust region')

    # Points
    ax1.plot(theta_0_single, J(theta_0_single), 'o', color=COLORS['blue'],
             markersize=9, zorder=5)
    ax1.annotate(r'$\theta_{\mathrm{old}}$',
                 (theta_0_single, J(theta_0_single)),
                 xytext=(-15, -20), textcoords='offset points', fontsize=11,
                 color=COLORS['blue'])

    ax1.plot(theta_new_single, J(theta_new_single), 'o', color=COLORS['green'],
             markersize=9, zorder=5)
    ax1.annotate(r'$\theta_{\mathrm{new}}$',
                 (theta_new_single, J(theta_new_single)),
                 xytext=(8, -15), textcoords='offset points', fontsize=11,
                 color=COLORS['green'])

    # Guaranteed improvement annotation
    mid_y = (L_at_new + J_at_new) / 2
    ax1.annotate('', xy=(theta_new_single + 0.08, J_at_new),
                 xytext=(theta_new_single + 0.08, L_at_new),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.5))
    ax1.text(theta_new_single + 0.2, mid_y, 'gap', fontsize=9, color=COLORS['gray'],
             va='center')

    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$J(\theta)$')
    ax1.set_title('(a) Single MM step')
    ax1.legend(loc='upper right', fontsize=9)

    # ── Right panel: Iterative convergence ────────────────────────────────────────

    ax2.plot(theta_range, J_vals, 'k-', linewidth=2, label=r'$J(\theta)$')

    # Plot surrogates
    for k in range(n_iter):
        theta_k = iterates[k]
        c_k = surrogates_c[k]
        L_k = surrogate(theta_range, theta_k, c_k)
        surr_mask = np.abs(theta_range - theta_k) < 2.0
        ax2.plot(theta_range[surr_mask], L_k[surr_mask], '--', color=iter_colors[k],
                 linewidth=1.2, alpha=0.7)

    # Mark iterates on J curve
    for i, theta_k in enumerate(iterates[:n_iter+1]):
        ax2.plot(theta_k, J(theta_k), 'o', color=iter_colors[min(i, n_iter-1)] if i < n_iter else COLORS['black'],
                 markersize=7, zorder=5)
        # Label with per-iterate offsets to avoid crowding near convergence
        label_offsets = [(-20, -22), (12, 14), (-12, -25), (15, 10), (-15, -20)]
        dx, dy = label_offsets[min(i, len(label_offsets) - 1)]
        ax2.annotate(rf'$\theta_{i}$', (theta_k, J(theta_k)),
                     xytext=(dx, dy), textcoords='offset points',
                     fontsize=10, ha='center', color=COLORS['black'],
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='none', alpha=0.85))

    # Mark theta*
    ax2.plot(theta_star, J(theta_star), '*', color=COLORS['black'], markersize=14, zorder=5)
    ax2.annotate(r'$\theta^*$', (theta_star, J(theta_star)),
                 xytext=(10, 5), textcoords='offset points', fontsize=11)

    # Trajectory arrows
    for i in range(len(iterates)-1):
        th_from, th_to = iterates[i], iterates[i+1]
        j_from, j_to = J(th_from), J(th_to)
        ax2.annotate('', xy=(th_to, j_to), xytext=(th_from, j_from),
                     arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                                     lw=1.2, alpha=0.6))

    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$J(\theta)$')
    ax2.set_title('(b) Iterative MM convergence')
    ax2.set_ylim(np.min(J_vals) - 0.3, np.max(J_vals) + 0.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'mm_surrogate_trpo.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nOutput: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true',
                        help='No computation to cache (diagram-only script)')
    parser.add_argument('--plots-only', action='store_true',
                        help='Runs normally (same as no flags)')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
