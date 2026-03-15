# Information Geometry of Natural Policy Gradient
# Chapter 3: Planning and Learning Theory
# Two-panel figure: policy manifold (conceptual) + tangent plane (analytical).

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sims'))
from plot_style import COLORS, FIG_DOUBLE, apply_style
apply_style()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.path import Path
import matplotlib.patches as mpatches

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_outputs():
    # ── Panel 2 (Right): Tangent plane analytics ─────────────────────────────────

    F = np.array([[2.0, 0.5], [0.5, 0.8]])  # Fisher information matrix
    g = np.array([1.0, 0.3])                 # policy gradient
    delta = 1.0                               # KL budget (for visualization scale)
    eps_euc = 1.0                             # Euclidean ball radius

    F_inv = np.linalg.inv(F)
    eigvals, eigvecs = np.linalg.eigh(F)

    # Euclidean steepest ascent: g/||g|| scaled to ball boundary
    g_norm = np.linalg.norm(g)
    euc_step = eps_euc * g / g_norm

    # Natural gradient steepest ascent: sqrt(delta / (g^T F^{-1} g)) F^{-1} g
    nat_grad = F_inv @ g
    gFg = g @ nat_grad
    nat_step = np.sqrt(delta / gFg) * nat_grad

    # KL of each step
    kl_euc = 0.5 * euc_step @ F @ euc_step
    kl_nat = 0.5 * nat_step @ F @ nat_step

    print("=" * 60)
    print("Information Geometry of Natural Policy Gradient")
    print("=" * 60)
    print(f"\nFisher information matrix F:")
    print(f"  [[{F[0,0]:.1f}, {F[0,1]:.1f}],")
    print(f"   [{F[1,0]:.1f}, {F[1,1]:.1f}]]")
    print(f"Fisher eigenvalues: {eigvals[0]:.4f}, {eigvals[1]:.4f}")
    print(f"Condition number: {eigvals[1]/eigvals[0]:.4f}")
    print(f"\nGradient g = ({g[0]:.1f}, {g[1]:.1f})")
    print(f"Natural gradient F^{{-1}}g = ({nat_grad[0]:.4f}, {nat_grad[1]:.4f})")
    print(f"\nEuclidean step direction: ({euc_step[0]:.4f}, {euc_step[1]:.4f})")
    print(f"  ||step||_2 = {np.linalg.norm(euc_step):.4f}")
    print(f"  KL(step) = {kl_euc:.4f}")
    print(f"\nNatural gradient step: ({nat_step[0]:.4f}, {nat_step[1]:.4f})")
    print(f"  ||step||_2 = {np.linalg.norm(nat_step):.4f}")
    print(f"  KL(step) = {kl_nat:.4f}")
    print(f"\nImprovement (linear approx):")
    print(f"  Euclidean: g^T * euc_step = {g @ euc_step:.4f}")
    print(f"  Natural:   g^T * nat_step = {g @ nat_step:.4f}")
    print(f"  Ratio (natural/euclidean): {(g @ nat_step)/(g @ euc_step):.4f}")


    # ── Figure ────────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: Policy manifold (conceptual) ─────────────────────────────────
    ax1.set_xlim(0.0, 5.3)
    ax1.set_ylim(-0.3, 4.3)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Draw curved manifold boundary using Bezier paths
    verts_outer = [
        (0.5, 0.5), (1.5, 0.0), (3.5, 0.3), (5.0, 1.0),
        (5.2, 2.5), (4.5, 3.8), (3.0, 4.2), (1.5, 3.5),
        (0.3, 2.5), (0.5, 0.5)
    ]
    codes_outer = [Path.MOVETO] + [Path.CURVE3] * 8 + [Path.CLOSEPOLY]
    path_outer = Path(verts_outer, codes_outer)
    patch_outer = mpatches.PathPatch(path_outer, facecolor='#e8e8e8', edgecolor='#555555',
                                      linewidth=1.5, alpha=0.6)
    ax1.add_patch(patch_outer)

    # Internal curved gridlines suggesting curvature
    for y_off in [1.2, 2.2, 3.0]:
        xs = np.linspace(0.8, 4.8, 50)
        ys = y_off + 0.3 * np.sin(np.pi * (xs - 0.8) / 4.0) - 0.05 * (xs - 2.8)**2
        mask = []
        for x, y in zip(xs, ys):
            if path_outer.contains_point((x, y)):
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask)
        xs_m, ys_m = xs[mask], ys[mask]
        if len(xs_m) > 2:
            ax1.plot(xs_m, ys_m, color='#bbbbbb', linewidth=0.6, alpha=0.5)

    for x_off in [1.5, 2.5, 3.5]:
        ys = np.linspace(0.5, 4.0, 50)
        xs = x_off + 0.25 * np.sin(np.pi * (ys - 0.5) / 3.5)
        mask = []
        for x, y in zip(xs, ys):
            if path_outer.contains_point((x, y)):
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask)
        xs_m, ys_m = xs[mask], ys[mask]
        if len(xs_m) > 2:
            ax1.plot(xs_m, ys_m, color='#bbbbbb', linewidth=0.6, alpha=0.5)

    # Points
    pi_old = np.array([1.5, 1.8])
    pi_star = np.array([3.8, 3.2])

    ax1.plot(*pi_old, 'o', color=COLORS['blue'], markersize=10, zorder=5)
    ax1.annotate(r'$\pi_{\theta_{\mathrm{old}}}$', pi_old, xytext=(-35, -20),
                 textcoords='offset points', fontsize=12, color=COLORS['blue'])

    ax1.plot(*pi_star, marker='*', color=COLORS['black'], markersize=16, zorder=5)
    ax1.annotate(r'$\pi^*$', pi_star, xytext=(10, -5),
                 textcoords='offset points', fontsize=12, color=COLORS['black'])

    # Euclidean gradient: straight arrow, pointing somewhat off from pi_star
    euc_dir = np.array([2.0, 0.3])  # nearly horizontal — "wrong" direction
    euc_dir = euc_dir / np.linalg.norm(euc_dir) * 1.8
    ax1.annotate('', xy=pi_old + euc_dir, xytext=pi_old,
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2.2))
    ax1.text(pi_old[0] + euc_dir[0] + 0.1, pi_old[1] + euc_dir[1] - 0.25,
             r'$\nabla_\theta J$', fontsize=11, color=COLORS['red'])

    # Natural gradient: curved arrow following manifold toward pi_star
    t_vals = np.linspace(0, 1, 30)
    curve_x = pi_old[0] + (pi_star[0] - pi_old[0]) * t_vals + 0.4 * np.sin(np.pi * t_vals)
    curve_y = pi_old[1] + (pi_star[1] - pi_old[1]) * t_vals + 0.3 * np.sin(np.pi * t_vals)
    # Only draw first portion (the step, not full path)
    n_draw = 12
    ax1.plot(curve_x[:n_draw], curve_y[:n_draw], color=COLORS['green'], linewidth=2.2,
             linestyle='-', zorder=4)
    # Arrowhead
    dx = curve_x[n_draw-1] - curve_x[n_draw-2]
    dy = curve_y[n_draw-1] - curve_y[n_draw-2]
    ax1.annotate('', xy=(curve_x[n_draw-1], curve_y[n_draw-1]),
                 xytext=(curve_x[n_draw-3], curve_y[n_draw-3]),
                 arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.2))
    ax1.text(curve_x[n_draw-1] + 0.15, curve_y[n_draw-1] + 0.1,
             r'$F^{-1}\nabla_\theta J$', fontsize=11, color=COLORS['green'])

    ax1.set_title('(a) Policy manifold $\\mathcal{M}$')


    # ── Right panel: Tangent plane ────────────────────────────────────────────────

    # KL ellipse: {Δθ : Δθ^T F Δθ ≤ δ}
    # Eigendecomposition for ellipse parameters
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    # Semi-axes: sqrt(delta / eigenvalue)
    width = 2 * np.sqrt(delta / eigvals[0])
    height = 2 * np.sqrt(delta / eigvals[1])

    # Euclidean ball
    euc_circle = plt.Circle((0, 0), eps_euc, color=COLORS['red'], alpha=0.12,
                              linewidth=1.5, linestyle='--', fill=True)
    euc_circle_border = plt.Circle((0, 0), eps_euc, color=COLORS['red'], alpha=0.5,
                                    linewidth=1.5, linestyle='--', fill=False)
    ax2.add_patch(euc_circle)
    ax2.add_patch(euc_circle_border)

    # KL ellipse
    kl_ellipse = Ellipse((0, 0), width, height, angle=angle,
                           color=COLORS['green'], alpha=0.12, linewidth=1.5,
                           linestyle='--')
    kl_ellipse_border = Ellipse((0, 0), width, height, angle=angle,
                                  edgecolor=COLORS['green'], alpha=0.5, linewidth=1.5,
                                  linestyle='--', fill=False)
    ax2.add_patch(kl_ellipse)
    ax2.add_patch(kl_ellipse_border)

    # Origin
    ax2.plot(0, 0, 'o', color=COLORS['blue'], markersize=8, zorder=5)
    ax2.annotate(r'$\theta_{\mathrm{old}}$', (0, 0), xytext=(-30, -18),
                 textcoords='offset points', fontsize=11, color=COLORS['blue'],
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    # Euclidean step arrow
    ax2.annotate('', xy=euc_step, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2.2))
    label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)
    ax2.text(euc_step[0] + 0.05, euc_step[1] - 0.15,
             r'$\nabla_\theta J / \|\nabla_\theta J\|$', fontsize=10, color=COLORS['red'],
             bbox=label_bbox)

    # Natural gradient step arrow
    ax2.annotate('', xy=nat_step, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.2))
    ax2.text(nat_step[0] + 0.05, nat_step[1] + 0.08,
             r'$F^{-1}g$ (natural)', fontsize=10,
             color=COLORS['green'], bbox=label_bbox)

    # Labels for regions
    ax2.text(0.85, -1.05, r'$\|\Delta\theta\|_2 \leq \varepsilon$',
             fontsize=10, color=COLORS['red'], alpha=0.8, bbox=label_bbox)
    ax2.text(-1.25, 0.95, r'$\Delta\theta^\top F\, \Delta\theta \leq \delta$',
             fontsize=10, color=COLORS['green'], alpha=0.8, bbox=label_bbox)

    lim = 1.45
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$\Delta\theta_1$')
    ax2.set_ylabel(r'$\Delta\theta_2$')
    ax2.set_title('(b) Tangent plane at $\\theta_{\\mathrm{old}}$')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'info_geometry_npg.png')
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
