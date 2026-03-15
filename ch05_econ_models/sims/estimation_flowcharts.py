# Estimation Flowcharts: NFXP vs RL-Based Structural Estimation
# Chapter 5: Solving Economic Models with RL
# Two-panel diagram contrasting nested fixed-point and single-loop approaches.

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Output path ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__))
OUT_PNG = os.path.join(OUT_DIR, 'estimation_flowcharts.png')

# ── Drawing helpers (ch09-style) ─────────────────────────────────────────────

ARROW_LW = 1.4


def draw_rect_node(ax, center, size, label, edgecolor='black', facecolor='white',
                   linewidth=1.4, fontsize=10, linestyle='-', alpha=1.0,
                   label_offset=(0, 0), zorder=3):
    """Draw a rounded rectangle node with a centred label."""
    x = center[0] - size[0] / 2
    y = center[1] - size[1] / 2
    rect = mpatches.FancyBboxPatch(
        (x, y), size[0], size[1],
        boxstyle='round,pad=0.08',
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    if label:
        ax.text(center[0] + label_offset[0], center[1] + label_offset[1],
                label, ha='center', va='center', fontsize=fontsize, zorder=zorder + 1)


def draw_container(ax, center, size, label='', edgecolor='black', facecolor='white',
                   linewidth=1.5, fontsize=10, linestyle='-', alpha=0.10,
                   label_pos='top_left', zorder=1):
    """Draw a large container rectangle (dashed or solid) with a label."""
    x = center[0] - size[0] / 2
    y = center[1] - size[1] / 2
    rect = mpatches.FancyBboxPatch(
        (x, y), size[0], size[1],
        boxstyle='round,pad=0.12',
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    if label:
        if label_pos == 'top_left':
            lx = x + 0.12
            ly = y + size[1] - 0.10
            ha, va = 'left', 'top'
        elif label_pos == 'top_center':
            lx = center[0]
            ly = y + size[1] - 0.10
            ha, va = 'center', 'top'
        else:
            lx, ly = center[0], center[1]
            ha, va = 'center', 'center'
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=fontsize,
                color=edgecolor, fontstyle='italic', zorder=zorder + 1)


def _rect_edge_point(center, size, target):
    """Find the point on the boundary of a rectangle closest to `target`,
    along the line from `center` to `target`."""
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center
    hw, hh = size[0] / 2, size[1] / 2
    # Scale factors to hit each edge
    scales = []
    if abs(dx) > 1e-9:
        scales.append(hw / abs(dx))
    if abs(dy) > 1e-9:
        scales.append(hh / abs(dy))
    t = min(scales)
    return (center[0] + t * dx, center[1] + t * dy)


def draw_edge(ax, p1, p2, p1_size=None, p2_size=None, dashed=False,
              lw=ARROW_LW, color='black', curve=0.0, shrink=0.04):
    """Draw a directed edge between two rectangular nodes (or raw points).
    If p1_size / p2_size are given, clip to rectangle boundaries."""
    if p1_size is not None:
        start = _rect_edge_point(p1, p1_size, p2)
    else:
        start = p1
    if p2_size is not None:
        end = _rect_edge_point(p2, p2_size, p1)
    else:
        end = p2
    # Small shrink to avoid touching the rounded corners
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = np.hypot(dx, dy)
    if dist > 2 * shrink:
        ux, uy = dx / dist, dy / dist
        start = (start[0] + shrink * ux, start[1] + shrink * uy)
        end = (end[0] - shrink * ux, end[1] - shrink * uy)
    props = dict(arrowstyle='->', lw=lw, color=color, shrinkA=0, shrinkB=0)
    if dashed:
        props['linestyle'] = (0, (5, 4))
    if abs(curve) > 1e-6:
        props['connectionstyle'] = f'arc3,rad={curve}'
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=2)


def draw_self_loop(ax, center, size, side='right', color='black', lw=ARROW_LW,
                   label='', label_fontsize=9):
    """Draw a visible self-loop arc on one side of a rectangle."""
    hw, hh = size[0] / 2, size[1] / 2
    if side == 'right':
        edge_x = center[0] + hw
        y_lo = center[1] - hh * 0.7
        y_hi = center[1] + hh * 0.7
        start = (edge_x + 0.02, y_lo)
        end = (edge_x + 0.02, y_hi)
        rad = -1.2
        label_pos = (edge_x + 0.55, center[1])
    elif side == 'left':
        edge_x = center[0] - hw
        y_lo = center[1] - hh * 0.7
        y_hi = center[1] + hh * 0.7
        start = (edge_x - 0.02, y_hi)
        end = (edge_x - 0.02, y_lo)
        rad = -1.2
        label_pos = (edge_x - 0.55, center[1])
    else:
        return

    props = dict(
        arrowstyle='->', lw=lw, color=color,
        shrinkA=0, shrinkB=0,
        connectionstyle=f'arc3,rad={rad}',
        mutation_scale=12,
    )
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=5)
    if label:
        ax.text(label_pos[0], label_pos[1], label,
                fontsize=label_fontsize, ha='center', va='center', color=color)


# ── Main figure ──────────────────────────────────────────────────────────────

def generate_outputs():
    red = COLORS['red']
    blue = COLORS['blue']
    black = COLORS['black']
    gray = COLORS['gray']

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax in (ax_l, ax_r):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.grid(False)

    # ==================================================================
    # LEFT PANEL — NFXP (Nested Fixed-Point)
    # ==================================================================
    ax = ax_l
    ax.set_xlim(-2.2, 2.8)
    ax.set_ylim(-2.6, 3.2)

    ax.set_title('NFXP (Nested Fixed-Point)', fontsize=13, pad=14, color=black)

    # -- Data input node at top --
    data_center = (0.0, 2.6)
    data_size = (2.0, 0.45)
    draw_rect_node(ax, data_center, data_size,
                   r"Data $\{(s_i, a_i)\}_{i=1}^N$",
                   edgecolor=black, facecolor='#f5f5f5', fontsize=10)

    # -- Outer container: MLE loop (dashed) --
    outer_center = (0.0, 0.15)
    outer_size = (4.0, 3.6)
    draw_container(ax, outer_center, outer_size,
                   label=r'Outer loop: MLE over $\theta$',
                   edgecolor=red, facecolor=red,
                   linestyle=(0, (5, 3)), linewidth=2.2,
                   fontsize=10, alpha=0.06, label_pos='top_center')

    # -- Inner container: Bellman solve (solid) --
    inner_center = (-0.15, -0.15)
    inner_size = (3.0, 2.0)
    draw_container(ax, inner_center, inner_size,
                   label='Inner loop: Bellman equation',
                   edgecolor=red, facecolor=red,
                   linestyle='-', linewidth=1.4,
                   fontsize=9.5, alpha=0.12, label_pos='top_center')

    # -- Bellman update box --
    bellman_center = (-0.15, -0.30)
    bellman_size = (2.2, 0.55)
    draw_rect_node(ax, bellman_center, bellman_size,
                   r'$V_{k+1} = T_\theta V_k$',
                   edgecolor=red, facecolor='white', fontsize=11)

    # -- Self-loop on Bellman box --
    draw_self_loop(ax, bellman_center, bellman_size, side='right',
                   color=red, lw=1.4, label='VI\niters')

    # -- Arrow from data to outer loop --
    outer_top_y = outer_center[1] + outer_size[1] / 2
    draw_edge(ax, data_center, (0.0, outer_top_y),
              p1_size=data_size, color=black, lw=1.4)

    # -- Output: V*(theta) -> L(theta) --
    output_center = (0.0, -2.15)
    output_size = (2.6, 0.50)
    draw_rect_node(ax, output_center, output_size,
                   r'$V^*(\theta) \;\to\; \mathcal{L}(\theta)$',
                   edgecolor=black, facecolor='#f5f5f5', fontsize=11)

    # Arrow from outer loop bottom to output
    outer_bot_y = outer_center[1] - outer_size[1] / 2
    draw_edge(ax, (0.0, outer_bot_y), output_center,
              p2_size=output_size, color=black, lw=1.4)

    # -- Complexity annotation --
    ax.text(0.0, -2.60,
            r'$\mathcal{O}(|\mathcal{S}|^2 \cdot N_{\mathrm{VI}})$ per $\theta$-evaluation',
            fontsize=9.5, ha='center', va='top', color=red)

    # ==================================================================
    # RIGHT PANEL — RL-Based Estimation (Single Loop)
    # ==================================================================
    ax = ax_r
    ax.set_xlim(-2.2, 2.8)
    ax.set_ylim(-2.6, 3.2)

    ax.set_title('RL-Based Estimation (Single Loop)', fontsize=13, pad=14, color=black)

    # -- Data input node at top --
    data_center_r = (0.0, 2.6)
    data_size_r = (2.4, 0.45)
    draw_rect_node(ax, data_center_r, data_size_r,
                   r"Data batch $(s_t, a_t, s_{t+1})$",
                   edgecolor=black, facecolor='#f5f5f5', fontsize=10)

    # -- Main container: single-loop SA --
    main_center = (0.0, 0.15)
    main_size = (4.0, 3.6)
    draw_container(ax, main_center, main_size,
                   label='Single-loop stochastic approximation',
                   edgecolor=blue, facecolor=blue,
                   linestyle='-', linewidth=1.6,
                   fontsize=10, alpha=0.08, label_pos='top_center')

    # -- Arrow from data to main container --
    draw_edge(ax, data_center_r, (0.0, main_center[1] + main_size[1] / 2),
              p1_size=data_size_r, color=black, lw=1.4)

    # -- Left sub-box: Update theta --
    theta_center = (-0.85, 0.0)
    theta_size = (1.7, 1.1)
    draw_rect_node(ax, theta_center, theta_size,
                   r'Update $\theta$' + '\n(structural\nparameters)',
                   edgecolor=blue, facecolor=blue, alpha=0.15,
                   fontsize=9.5, linewidth=1.2)

    # -- Right sub-box: Update omega --
    omega_center = (0.95, 0.0)
    omega_size = (1.7, 1.1)
    draw_rect_node(ax, omega_center, omega_size,
                   r'Update $\omega$' + '\n(value/policy\nweights)',
                   edgecolor=blue, facecolor=blue, alpha=0.15,
                   fontsize=9.5, linewidth=1.2)

    # -- Bidirectional coupling arrows between theta and omega --
    mid_y_hi = 0.22
    mid_y_lo = -0.22
    draw_edge(ax, (theta_center[0] + theta_size[0] / 2, mid_y_hi),
              (omega_center[0] - omega_size[0] / 2, mid_y_hi),
              color=blue, lw=1.2, shrink=0.06)
    draw_edge(ax, (omega_center[0] - omega_size[0] / 2, mid_y_lo),
              (theta_center[0] + theta_size[0] / 2, mid_y_lo),
              color=blue, lw=1.2, shrink=0.06)

    # -- Output --
    output_center_r = (0.0, -2.15)
    output_size_r = (2.6, 0.50)
    draw_rect_node(ax, output_center_r, output_size_r,
                   r'$\hat{\theta},\; \hat{\omega}$',
                   edgecolor=black, facecolor='#f5f5f5', fontsize=11)

    # Arrow from main container bottom to output
    draw_edge(ax, (0.0, main_center[1] - main_size[1] / 2), output_center_r,
              p2_size=output_size_r, color=black, lw=1.4)

    # -- Complexity annotation --
    ax.text(0.0, -2.60,
            r'Two-timescale SA: $\mathcal{O}(1)$ per gradient step',
            fontsize=9.5, ha='center', va='top', color=blue)

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Output file: {OUT_PNG}")
    print("Estimation flowcharts diagram generated.")


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
