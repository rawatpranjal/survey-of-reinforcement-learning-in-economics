"""Deadly triad geometry: orthogonal vs oblique projection in R^2.

Chapter 3 — Theory. Shows why off-policy + function approximation diverges:
oblique projection can expand the Bellman operator, breaking the contraction.
"""

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_outputs():
    # ── Geometry setup ──────────────────────────────────────────────────────────

    theta_sub = np.radians(20)          # subspace angle from x-axis
    e_sub = np.array([np.cos(theta_sub), np.sin(theta_sub)])  # unit along span(Φ)
    n_sub = np.array([-np.sin(theta_sub), np.cos(theta_sub)]) # normal to span(Φ)

    # TV constructed with controlled along/perp components so projections
    # produce moderate, visually clear expansion/contraction.
    TV_along = 2.0    # component along subspace
    TV_perp = 0.4     # component perpendicular to subspace (small)
    TV = TV_along * e_sub + TV_perp * n_sub
    norm_TV = np.linalg.norm(TV)

    # Panel 1: orthogonal projection — drops perpendicular component
    proj_orth = TV_along * e_sub
    norm_orth = np.linalg.norm(proj_orth)

    # Panel 2: oblique projection along direction d (20° below subspace → 0°)
    # d nearly parallel to subspace but slightly tilted, causing overshoot.
    delta_deg = 20     # angle below subspace
    theta_d = theta_sub - np.radians(delta_deg)  # = 0° (along x-axis)
    d = np.array([np.cos(theta_d), np.sin(theta_d)])
    t_obl = -np.dot(TV, n_sub) / np.dot(d, n_sub)
    proj_oblique = TV + t_obl * d
    norm_oblique = np.linalg.norm(proj_oblique)

    # ── Print results ───────────────────────────────────────────────────────────

    print("Deadly Triad Geometry — Projection Computations")
    print("=" * 55)
    print(f"Subspace angle:           {np.degrees(theta_sub):.1f} deg")
    print(f"Oblique direction angle:  {np.degrees(theta_d):.1f} deg  (delta = {delta_deg} deg below subspace)")
    print(f"TV (Cartesian):           [{TV[0]:.4f}, {TV[1]:.4f}]")
    print(f"TV along subspace:        {TV_along:.4f}")
    print(f"TV perp to subspace:      {TV_perp:.4f}")
    print()
    print(f"Orthogonal projection:    [{proj_orth[0]:.4f}, {proj_orth[1]:.4f}]")
    print(f"Oblique projection:       [{proj_oblique[0]:.4f}, {proj_oblique[1]:.4f}]")
    print()
    print(f"||TV||           = {norm_TV:.4f}")
    print(f"||Proj_orth TV|| = {norm_orth:.4f}")
    print(f"||Proj_obl  TV|| = {norm_oblique:.4f}")
    print()
    print(f"||Proj_orth TV|| < ||TV||:    {norm_orth < norm_TV}  ({norm_orth:.4f} < {norm_TV:.4f})")
    print(f"||TV|| < ||Proj_obl  TV||:    {norm_TV < norm_oblique}  ({norm_TV:.4f} < {norm_oblique:.4f})")
    print(f"Expansion ratio (oblique/TV): {norm_oblique / norm_TV:.4f}")
    print()

    # ── Drawing helpers ─────────────────────────────────────────────────────────

    def draw_arrow(ax, tail, tip, color, lw=2.0, zorder=5):
        """Draw a vector arrow from tail to tip."""
        ax.annotate('', xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle='->', lw=lw, color=color),
                    zorder=zorder)

    def draw_right_angle(ax, vertex, dir1, dir2, size=0.12, color=COLORS['gray'], lw=1.2):
        """Draw a right-angle marker at vertex along two perpendicular directions."""
        u1 = dir1 / np.linalg.norm(dir1) * size
        u2 = dir2 / np.linalg.norm(dir2) * size
        pts = np.array([vertex + u1, vertex + u1 + u2, vertex + u2])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=6)

    def draw_subspace(ax, e_sub, lo=-0.8, hi=3.5):
        """Draw the 1D subspace line."""
        t_vals = np.array([lo, hi])
        xs = t_vals * e_sub[0]
        ys = t_vals * e_sub[1]
        ax.plot(xs, ys, color=COLORS['gray'], lw=3, zorder=1, alpha=0.6)
        # label at the far end
        label_pos = hi * e_sub + np.array([0.05, -0.15])
        ax.text(label_pos[0], label_pos[1], r'span($\Phi$)',
                fontsize=11, color=COLORS['gray'], ha='left', va='top')

    # ── Figure ──────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for ax in (ax1, ax2):
        ax.set_aspect('equal')
        ax.axis('off')

    # ── Panel 1: Orthogonal projection ─────────────────────────────────────────

    draw_subspace(ax1, e_sub)

    # Origin dot
    ax1.plot(0, 0, 'o', color=COLORS['black'], ms=5, zorder=7)

    # TV arrow (blue)
    draw_arrow(ax1, [0, 0], TV, COLORS['blue'])
    ax1.text(TV[0] + 0.05, TV[1] + 0.10, r'$TV$',
             fontsize=14, color=COLORS['blue'], ha='left', va='bottom', weight='bold')

    # Projection arrow (green)
    draw_arrow(ax1, [0, 0], proj_orth, COLORS['green'])
    ax1.text(proj_orth[0] + 0.08, proj_orth[1] - 0.18, r'$\Pi_\mu\, TV$',
             fontsize=14, color=COLORS['green'], ha='left', va='top', weight='bold')

    # Dashed line TV -> orthogonal projection
    ax1.plot([TV[0], proj_orth[0]], [TV[1], proj_orth[1]],
             '--', color=COLORS['gray'], lw=1.4, zorder=4)

    # Right-angle marker at proj_orth
    perp_dir = TV - proj_orth  # perpendicular to subspace (along n_sub)
    draw_right_angle(ax1, proj_orth, e_sub, perp_dir, size=0.10)

    # Length annotation
    ax1.text(0.50, -0.02, r'$\|\Pi_\mu\, TV\| < \|TV\|$  — contraction preserved',
             fontsize=11, color=COLORS['black'], ha='center', va='top',
             transform=ax1.transAxes)

    ax1.set_title('(a) On-policy: orthogonal projection', fontsize=13, pad=12)
    ax1.set_xlim(-0.6, 3.8)
    ax1.set_ylim(-0.6, 1.8)

    # ── Panel 2: Oblique projection ────────────────────────────────────────────

    draw_subspace(ax2, e_sub)

    # Origin dot
    ax2.plot(0, 0, 'o', color=COLORS['black'], ms=5, zorder=7)

    # TV arrow (blue)
    draw_arrow(ax2, [0, 0], TV, COLORS['blue'])
    ax2.text(TV[0] + 0.05, TV[1] + 0.10, r'$TV$',
             fontsize=14, color=COLORS['blue'], ha='left', va='bottom', weight='bold')

    # Oblique projection arrow (red)
    draw_arrow(ax2, [0, 0], proj_oblique, COLORS['red'])
    ax2.text(proj_oblique[0] + 0.08, proj_oblique[1] - 0.18, r'$\Pi_\nu\, TV$',
             fontsize=14, color=COLORS['red'], ha='left', va='top', weight='bold')

    # Dashed line TV -> oblique projection (along direction d)
    ax2.plot([TV[0], proj_oblique[0]], [TV[1], proj_oblique[1]],
             '--', color=COLORS['gray'], lw=1.4, zorder=4)

    # Length annotation
    ax2.text(0.50, -0.02, r'$\|\Pi_\nu\, TV\| > \|TV\|$  — expansion causes divergence',
             fontsize=11, color=COLORS['black'], ha='center', va='top',
             transform=ax2.transAxes)

    ax2.set_title('(b) Off-policy: oblique projection', fontsize=13, pad=12)
    ax2.set_xlim(-0.6, 4.2)
    ax2.set_ylim(-0.6, 1.8)

    # ── Save ────────────────────────────────────────────────────────────────────

    plt.tight_layout(w_pad=2.0)
    out_path = os.path.join(os.path.dirname(__file__), 'deadly_triad_geometry.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Output: {out_path}")
    print(f"File exists: {os.path.exists(out_path)}")


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
