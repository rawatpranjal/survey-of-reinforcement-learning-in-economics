"""Algorithm architecture diagrams for Chapter 2.
Three-panel comparison: DQN (value-based), REINFORCE (policy gradient), Actor-Critic."""

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS, FIG_WIDE
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# DAG-style drawing utilities
# ---------------------------------------------------------------------------

NODE_RADIUS = 0.22
RECT_PAD = 0.12          # padding inside rounded rectangles
FONT_SIZE = 12
LABEL_FONT = 10
ARROW_LW = 1.4
DASH_STYLE = (0, (5, 4))

CIRCLE_STYLE = dict(
    facecolor='white',
    edgecolor='black',
    linewidth=1.4,
    linestyle='-',
    zorder=3,
)

RECT_STYLE_TEMPLATE = dict(
    linewidth=1.4,
    linestyle='-',
    zorder=3,
)


def _circle_edge_point(center, target, r=NODE_RADIUS):
    """Point on circle boundary in direction of target."""
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        return center
    ux, uy = dx / dist, dy / dist
    return (center[0] + r * ux, center[1] + r * uy)


def _rect_edge_point(center, size, target):
    """Point on rectangle boundary in direction of target.

    center, size, target are (x, y) tuples; size is (width, height).
    Returns the intersection of the line from center to target with
    the rectangle boundary.
    """
    cx, cy = center
    w, h = size
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (cx + w / 2, cy)
    scales = []
    if dx != 0:
        scales.append(abs((w / 2) / dx))
    if dy != 0:
        scales.append(abs((h / 2) / dy))
    s = min(scales) if scales else 1.0
    return (cx + dx * s, cy + dy * s)


def draw_circle_node(ax, xy, label, radius=NODE_RADIUS, fontsize=FONT_SIZE):
    """Draw a circular data node (states, actions)."""
    circle = mpatches.Circle(xy, radius, **CIRCLE_STYLE)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=fontsize, zorder=4, usetex=False)


def draw_rect_node(ax, xy, size, label, color, alpha=0.15,
                   fontsize=LABEL_FONT):
    """Draw a rounded rectangle operation node (networks, operations)."""
    w, h = size
    x0 = xy[0] - w / 2
    y0 = xy[1] - h / 2
    box = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f'round,pad={RECT_PAD}',
        facecolor=color, alpha=alpha,
        edgecolor='black',
        **RECT_STYLE_TEMPLATE,
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=fontsize, zorder=4, usetex=False)


def draw_edge(ax, p1, p2, p1_shape='circle', p2_shape='circle',
              p1_size=None, p2_size=None,
              dashed=False, lw=ARROW_LW, color='black', curve=0.0,
              label='', label_fontsize=9, label_offset=(0, 0.18)):
    """Draw a directed arrow between two nodes, clipping to boundaries.

    p1_shape/p2_shape: 'circle' or 'rect'
    p1_size/p2_size: for 'circle' = radius, for 'rect' = (w, h)
    """
    # Default sizes
    if p1_size is None:
        p1_size = NODE_RADIUS if p1_shape == 'circle' else (1.0, 0.5)
    if p2_size is None:
        p2_size = NODE_RADIUS if p2_shape == 'circle' else (1.0, 0.5)

    # Compute clipped endpoints
    if p1_shape == 'circle':
        start = _circle_edge_point(p1, p2, r=p1_size if isinstance(p1_size, (int, float)) else p1_size[0])
    else:
        start = _rect_edge_point(p1, p1_size, p2)

    if p2_shape == 'circle':
        end = _circle_edge_point(p2, p1, r=p2_size if isinstance(p2_size, (int, float)) else p2_size[0])
    else:
        end = _rect_edge_point(p2, p2_size, p1)

    props = dict(arrowstyle='->', lw=lw, color=color, shrinkA=0, shrinkB=0)
    if dashed:
        props['linestyle'] = DASH_STYLE
    if abs(curve) > 1e-6:
        props['connectionstyle'] = f'arc3,rad={curve}'
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=2)

    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, ha='center', va='center',
                fontsize=label_fontsize, color=color)


# ---------------------------------------------------------------------------
# Convenience: connect two nodes by name from a registry
# ---------------------------------------------------------------------------

def _connect(ax, nodes, name1, name2, **kwargs):
    """Draw edge between two registered nodes."""
    n1 = nodes[name1]
    n2 = nodes[name2]
    draw_edge(ax, n1['xy'], n2['xy'],
              p1_shape=n1['shape'], p2_shape=n2['shape'],
              p1_size=n1['size'], p2_size=n2['size'],
              **kwargs)


# ---------------------------------------------------------------------------
# Panel dimensions
# ---------------------------------------------------------------------------

XLO, XHI = -0.6, 6.0
YLO, YHI = -1.0, 3.5
YMID = 1.5


# ---------------------------------------------------------------------------
# Panel 1: DQN (Value-Based)
# ---------------------------------------------------------------------------

def draw_dqn(ax):
    col = ALGO_COLORS['DQN']

    # Node registry: name -> {xy, shape, size}
    nodes = {
        's':      {'xy': (0.0,  YMID), 'shape': 'circle', 'size': NODE_RADIUS},
        'qnet':   {'xy': (1.6,  YMID), 'shape': 'rect',   'size': (1.6, 0.65)},
        'argmax': {'xy': (3.5,  YMID), 'shape': 'rect',   'size': (1.1, 0.55)},
        'a':      {'xy': (5.0,  YMID), 'shape': 'circle', 'size': NODE_RADIUS},
    }

    draw_circle_node(ax, nodes['s']['xy'], r'$s_t$')
    draw_rect_node(ax, nodes['qnet']['xy'], nodes['qnet']['size'],
                   r'$Q(s,\cdot\,;\theta)$', color=col, fontsize=12)
    draw_rect_node(ax, nodes['argmax']['xy'], nodes['argmax']['size'],
                   r'$\arg\max_a$', color=col, alpha=0.08, fontsize=11)
    draw_circle_node(ax, nodes['a']['xy'], r'$a_t^*$')

    _connect(ax, nodes, 's', 'qnet')
    _connect(ax, nodes, 'qnet', 'argmax')
    _connect(ax, nodes, 'argmax', 'a')

    # Environment feedback loop below
    env_xy = (2.5, -0.15)
    env_size = (1.6, 0.55)
    draw_rect_node(ax, env_xy, env_size, 'Environment', color=COLORS['gray'],
                   alpha=0.10, fontsize=10)
    env_node = {'xy': env_xy, 'shape': 'rect', 'size': env_size}

    # a -> env (action feeds into environment)
    draw_edge(ax, nodes['a']['xy'], env_xy,
              p1_shape='circle', p2_shape='rect',
              p1_size=NODE_RADIUS, p2_size=env_size,
              curve=0.4, color=COLORS['gray'])

    # env -> s (environment produces next state)
    draw_edge(ax, env_xy, nodes['s']['xy'],
              p1_shape='rect', p2_shape='circle',
              p1_size=env_size, p2_size=NODE_RADIUS,
              curve=0.4, color=COLORS['gray'])

    # Label the feedback
    ax.text(4.3, 0.35, r'$a_t$', fontsize=9, color=COLORS['gray'],
            ha='center', va='center')
    ax.text(0.65, 0.35, r'$r_{t+1}, s_{t+1}$', fontsize=9, color=COLORS['gray'],
            ha='center', va='center')

    ax.set_title('(a)  Value-Based (DQN)', fontsize=13, pad=12)


# ---------------------------------------------------------------------------
# Panel 2: REINFORCE (Policy Gradient)
# ---------------------------------------------------------------------------

def draw_reinforce(ax):
    col = ALGO_COLORS['REINFORCE']

    nodes = {
        's':       {'xy': (0.0,  YMID), 'shape': 'circle', 'size': NODE_RADIUS},
        'policy':  {'xy': (1.6,  YMID), 'shape': 'rect',   'size': (1.6, 0.65)},
        'sample':  {'xy': (3.5,  YMID), 'shape': 'rect',   'size': (1.1, 0.55)},
        'a':       {'xy': (5.0,  YMID), 'shape': 'circle', 'size': NODE_RADIUS},
    }

    draw_circle_node(ax, nodes['s']['xy'], r'$s_t$')
    draw_rect_node(ax, nodes['policy']['xy'], nodes['policy']['size'],
                   r'$\pi_\theta(a|s)$', color=col, fontsize=12)
    draw_rect_node(ax, nodes['sample']['xy'], nodes['sample']['size'],
                   r'$a \sim \pi(\cdot|s)$', color=col, alpha=0.08, fontsize=10)
    draw_circle_node(ax, nodes['a']['xy'], r'$a_t$')

    _connect(ax, nodes, 's', 'policy')
    _connect(ax, nodes, 'policy', 'sample')
    _connect(ax, nodes, 'sample', 'a')

    # Environment feedback loop below
    env_xy = (2.5, -0.15)
    env_size = (1.6, 0.55)
    draw_rect_node(ax, env_xy, env_size, 'Environment', color=COLORS['gray'],
                   alpha=0.10, fontsize=10)

    draw_edge(ax, nodes['a']['xy'], env_xy,
              p1_shape='circle', p2_shape='rect',
              p1_size=NODE_RADIUS, p2_size=env_size,
              curve=0.4, color=COLORS['gray'])

    draw_edge(ax, env_xy, nodes['s']['xy'],
              p1_shape='rect', p2_shape='circle',
              p1_size=env_size, p2_size=NODE_RADIUS,
              curve=0.4, color=COLORS['gray'])

    ax.text(4.3, 0.35, r'$a_t$', fontsize=9, color=COLORS['gray'],
            ha='center', va='center')
    ax.text(0.65, 0.35, r'$r_{t+1}, s_{t+1}$', fontsize=9, color=COLORS['gray'],
            ha='center', va='center')

    ax.set_title('(b)  Policy Gradient (REINFORCE)', fontsize=13, pad=12)


# ---------------------------------------------------------------------------
# Panel 3: Actor-Critic
# ---------------------------------------------------------------------------

def draw_actor_critic(ax):
    col = ALGO_COLORS['Actor-Critic']
    y_actor = 2.3
    y_critic = 0.7

    nodes = {
        's':      {'xy': (0.0,  YMID), 'shape': 'circle', 'size': NODE_RADIUS},
        'actor':  {'xy': (2.0,  y_actor), 'shape': 'rect', 'size': (1.8, 0.60)},
        'a':      {'xy': (4.2,  y_actor), 'shape': 'circle', 'size': NODE_RADIUS},
        'critic': {'xy': (2.0,  y_critic), 'shape': 'rect', 'size': (1.8, 0.60)},
        'td':     {'xy': (4.2,  y_critic), 'shape': 'circle', 'size': NODE_RADIUS},
    }

    draw_circle_node(ax, nodes['s']['xy'], r'$s_t$')
    draw_rect_node(ax, nodes['actor']['xy'], nodes['actor']['size'],
                   r'Actor $\pi_\theta(a|s)$', color=col, fontsize=11)
    draw_circle_node(ax, nodes['a']['xy'], r'$a_t$')
    draw_rect_node(ax, nodes['critic']['xy'], nodes['critic']['size'],
                   r'Critic $V_w(s)$', color=col, fontsize=11)
    draw_circle_node(ax, nodes['td']['xy'], r'$\delta_t$')

    # s -> actor, s -> critic
    _connect(ax, nodes, 's', 'actor')
    _connect(ax, nodes, 's', 'critic')

    # actor -> a
    _connect(ax, nodes, 'actor', 'a')

    # critic -> td
    _connect(ax, nodes, 'critic', 'td')

    # Feedback: td -> actor (dashed, curved upward)
    draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'],
              p1_shape='circle', p2_shape='rect',
              p1_size=NODE_RADIUS, p2_size=nodes['actor']['size'],
              dashed=True, color=col, curve=-0.35)

    # TD error formula below the delta node
    ax.text(nodes['td']['xy'][0], nodes['td']['xy'][1] - 0.45,
            r'$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$',
            ha='center', va='top', fontsize=9, color=col,
            fontstyle='italic')

    ax.set_title('(c)  Actor-Critic', fontsize=13, pad=12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_outputs():
    fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE)

    for ax in axes:
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(YLO, YHI)
        ax.set_aspect('equal')
        ax.axis('off')

    draw_dqn(axes[0])
    draw_reinforce(axes[1])
    draw_actor_critic(axes[2])

    plt.tight_layout(pad=1.5)

    outpath = os.path.join(os.path.dirname(__file__), 'algorithm_architectures.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"Output: {os.path.abspath(outpath)}")
    print("Algorithm architectures diagram generated.")


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
