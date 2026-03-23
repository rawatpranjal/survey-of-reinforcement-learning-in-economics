"""
Identification Strategy DAGs — Chapter 9 (Causal RL)
Generates two publication-quality DAG figures:
  1. identification_dags.png — 1x3 grid: front-door, IV, proximal
  2. simulation_dag.png — complete DGP with all variables
"""

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS
apply_style()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# DAG drawing utilities
# ---------------------------------------------------------------------------

NODE_RADIUS = 0.13          # in data coordinates
FONT_SIZE = 11
ARROW_LW = 1.4
DASH_STYLE = (0, (5, 4))   # dash pattern for unobserved edges

OBSERVED_STYLE = dict(
    facecolor='white',
    edgecolor='black',
    linewidth=1.4,
    linestyle='-',
    zorder=3,
)

UNOBSERVED_STYLE = dict(
    facecolor='#e0e0e0',
    edgecolor='black',
    linewidth=1.4,
    linestyle='--',
    zorder=3,
)


def _edge_endpoints(p1, p2, r=NODE_RADIUS):
    """Return (start, end) on the boundary of two circles so arrows
    connect at circle edges rather than centres."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        return p1, p2
    ux, uy = dx / dist, dy / dist
    start = (p1[0] + r * ux, p1[1] + r * uy)
    end   = (p2[0] - r * ux, p2[1] - r * uy)
    return start, end


def draw_node(ax, xy, label, observed=True, radius=NODE_RADIUS, fontsize=FONT_SIZE):
    """Draw a circular node with a LaTeX label."""
    style = OBSERVED_STYLE if observed else UNOBSERVED_STYLE
    circle = mpatches.Circle(xy, radius, **style)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=fontsize, zorder=4,
            usetex=False)


def draw_edge(ax, p1, p2, dashed=False, radius=NODE_RADIUS, lw=ARROW_LW,
              color='black', curve=0.0):
    """Draw a directed edge (arrow) between two node centres,
    clipping to circle boundaries."""
    start, end = _edge_endpoints(p1, p2, r=radius)
    style = '->'
    props = dict(
        arrowstyle=style,
        lw=lw,
        color=color,
        shrinkA=0, shrinkB=0,
    )
    if dashed:
        props['linestyle'] = DASH_STYLE
    if abs(curve) > 1e-6:
        props['connectionstyle'] = f'arc3,rad={curve}'
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=2)


# ===================================================================
# Figure 1: 1×3 identification DAGs
# ===================================================================

def make_identification_dags():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax in axes:
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.grid(False)

    # ------------------------------------------------------------------
    # (a) Front-door criterion
    # ------------------------------------------------------------------
    ax = axes[0]
    # Node positions: U top-centre, A left, S' right, M bottom-centre
    pos = dict(
        U  = (0.5,  1.05),
        A  = (0.0,  0.45),
        S  = (1.0,  0.45),
        M  = (0.5, -0.15),
    )
    draw_node(ax, pos['U'], r'$U_t$', observed=False)
    draw_node(ax, pos['A'], r'$A_t$')
    draw_node(ax, pos['S'], r'$S_{t+1}$', fontsize=10)
    draw_node(ax, pos['M'], r'$M_t$')

    draw_edge(ax, pos['U'], pos['A'], dashed=True)
    draw_edge(ax, pos['U'], pos['S'], dashed=True)
    draw_edge(ax, pos['A'], pos['M'])
    draw_edge(ax, pos['M'], pos['S'])

    ax.set_title(r'(a) Front-door', fontsize=12, pad=10)

    # ------------------------------------------------------------------
    # (b) Instrumental variables
    # ------------------------------------------------------------------
    ax = axes[1]
    pos = dict(
        Z  = (-0.25, 0.45),
        U  = (0.5,   1.05),
        A  = (0.5,   0.45),
        S  = (1.15,  0.45),
    )
    draw_node(ax, pos['Z'], r'$Z_t$')
    draw_node(ax, pos['U'], r'$U_t$', observed=False)
    draw_node(ax, pos['A'], r'$A_t$')
    draw_node(ax, pos['S'], r'$S_{t+1}$', fontsize=10)

    draw_edge(ax, pos['Z'], pos['A'])
    draw_edge(ax, pos['U'], pos['A'], dashed=True)
    draw_edge(ax, pos['U'], pos['S'], dashed=True)
    draw_edge(ax, pos['A'], pos['S'])

    ax.set_title(r'(b) Instrumental variables', fontsize=12, pad=10)

    # ------------------------------------------------------------------
    # (c) Proximal causal inference
    # ------------------------------------------------------------------
    ax = axes[2]
    pos = dict(
        U  = (0.5,   1.05),
        A  = (0.05,  0.45),
        S  = (0.95,  0.45),
        W1 = (0.05, -0.15),
        W2 = (0.95, -0.15),
    )
    draw_node(ax, pos['U'],  r'$U_t$', observed=False)
    draw_node(ax, pos['A'],  r'$A_t$')
    draw_node(ax, pos['S'],  r'$S_{t+1}$', fontsize=10)
    draw_node(ax, pos['W1'], r'$W_t^{(1)}$', fontsize=9)
    draw_node(ax, pos['W2'], r'$W_t^{(2)}$', fontsize=9)

    draw_edge(ax, pos['U'], pos['A'],  dashed=True)
    draw_edge(ax, pos['U'], pos['S'],  dashed=True)
    draw_edge(ax, pos['U'], pos['W1'], dashed=True)
    draw_edge(ax, pos['U'], pos['W2'], dashed=True)
    draw_edge(ax, pos['A'], pos['S'])

    ax.set_title(r'(c) Proximal causal inference', fontsize=12, pad=10)

    fig.tight_layout(pad=1.5)
    outpath = os.path.join(os.path.dirname(__file__), 'identification_dags.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# ===================================================================
# Figure 2: Complete simulation DAG
# ===================================================================

def make_simulation_dag():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)

    # Positions — left-to-right flow
    # S_t slightly above centre so the S_t→A_t edge doesn't pass through U_t
    pos = dict(
        St   = (0.0,  0.35),
        Zt   = (0.85, 0.9),
        Ut   = (0.85, -0.05),
        IVt  = (0.85, -0.85),
        At   = (1.75, 0.35),
        Mt   = (2.65, 0.75),
        W1   = (2.65, -0.15),
        W2   = (2.65, -0.75),
        Sp   = (3.55, 0.35),
    )

    R = 0.15  # slightly larger radius for the full DAG

    draw_node(ax, pos['St'],  r'$S_t$',         radius=R, fontsize=12)
    draw_node(ax, pos['Zt'],  r'$Z_t$',         radius=R, fontsize=12)
    draw_node(ax, pos['Ut'],  r'$U_t$',         radius=R, fontsize=12, observed=False)
    draw_node(ax, pos['IVt'], r'$IV_t$',        radius=R, fontsize=11)
    draw_node(ax, pos['At'],  r'$A_t$',         radius=R, fontsize=12)
    draw_node(ax, pos['Mt'],  r'$M_t$',         radius=R, fontsize=12)
    draw_node(ax, pos['W1'],  r'$W_t^{(1)}$',   radius=R, fontsize=10)
    draw_node(ax, pos['W2'],  r'$W_t^{(2)}$',   radius=R, fontsize=10)
    draw_node(ax, pos['Sp'],  r'$S_{t+1}$',     radius=R, fontsize=11)

    # Edges — observed (solid)
    draw_edge(ax, pos['IVt'], pos['At'],  radius=R)
    draw_edge(ax, pos['At'],  pos['Mt'],  radius=R)
    draw_edge(ax, pos['Mt'],  pos['Sp'],  radius=R)
    draw_edge(ax, pos['St'],  pos['Sp'],  radius=R, curve=-0.35)
    draw_edge(ax, pos['St'],  pos['At'],  radius=R)
    draw_edge(ax, pos['Zt'],  pos['Sp'],  radius=R, curve=-0.25)

    # Edges — unobserved / from hidden U (dashed)
    draw_edge(ax, pos['Zt'],  pos['Ut'],  radius=R, dashed=True)
    draw_edge(ax, pos['Ut'],  pos['At'],  radius=R, dashed=True)
    draw_edge(ax, pos['Ut'],  pos['W1'],  radius=R, dashed=True)
    draw_edge(ax, pos['Ut'],  pos['W2'],  radius=R, dashed=True)

    # Legend for observed / unobserved
    legend_y = -1.20
    legend_x = 0.9
    obs_circle = mpatches.Circle((legend_x, legend_y), 0.08, **OBSERVED_STYLE)
    ax.add_patch(obs_circle)
    ax.text(legend_x + 0.15, legend_y, 'Observed', fontsize=10, va='center')

    unobs_circle = mpatches.Circle((legend_x + 1.2, legend_y), 0.08, **UNOBSERVED_STYLE)
    ax.add_patch(unobs_circle)
    ax.text(legend_x + 1.35, legend_y, 'Unobserved', fontsize=10, va='center')

    # Set axis limits with some padding
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad = 0.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad - 0.6, max(all_y) + pad)

    fig.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'simulation_dag.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# ===================================================================
# Main
# ===================================================================

def generate_outputs():
    make_identification_dags()
    make_simulation_dag()
    print("Done.")


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
