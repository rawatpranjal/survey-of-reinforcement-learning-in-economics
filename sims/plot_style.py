# Shared matplotlib style for all simulation scripts.
# Import and call apply_style() at the top of each script.

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette (muted/desaturated, named for convenience)
# ---------------------------------------------------------------------------

COLORS = {
    'blue':   '#4878A8',
    'orange': '#D4915C',
    'green':  '#6BA368',
    'red':    '#C25B56',
    'purple': '#8B7BAF',
    'brown':  '#9B7B6B',
    'pink':   '#C99BBD',
    'gray':   '#888888',
    'olive':  '#A3A95E',
    'cyan':   '#5DA5A8',
    'black':  '#2D2D2D',
}

# ---------------------------------------------------------------------------
# Algorithm color mapping (consistent across all figures)
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    'VI':             COLORS['black'],
    'PI':             COLORS['gray'],
    'Q-Learning':     COLORS['blue'],
    'SARSA':          COLORS['orange'],
    'Expected SARSA': COLORS['green'],
    'MC Control':     COLORS['red'],
    'SARSA(\u03bb)':  COLORS['purple'],
    'Q(\u03bb)':      COLORS['brown'],
    'REINFORCE':      COLORS['pink'],
    'NPG':            '#C47832',
    'PPO':            COLORS['olive'],
    'LP':             '#B07070',
    'DQN':            COLORS['cyan'],
    'SAC':            '#6B8F5E',
    'Actor-Critic':   '#7055A0',
}

# ---------------------------------------------------------------------------
# Standard sequential colormap for all heatmaps/contour plots
# ---------------------------------------------------------------------------

CMAP_SEQ = 'viridis'

# ---------------------------------------------------------------------------
# Benchmark/reference line style (Nash equilibria, oracle, optimal, analytical)
# ---------------------------------------------------------------------------

BENCH_STYLE = {'color': '#2D2D2D', 'linestyle': '--', 'linewidth': 1.0, 'zorder': 1}

# ---------------------------------------------------------------------------
# Domain-specific color maps (non-algorithm concepts)
# ---------------------------------------------------------------------------

DOMAIN_COLORS = {
    # Causal estimators (ch09)
    'Naive':      COLORS['orange'],
    'Backdoor':   COLORS['blue'],
    'Front-door': COLORS['green'],
    'IV':         COLORS['purple'],
    'Proximal':   COLORS['red'],
    # Bandit algorithms (ch07)
    'UCB1':       COLORS['blue'],
    'Thompson':   COLORS['green'],
    'ε-greedy':   COLORS['red'],
    'Oracle':     COLORS['black'],
    # Game theory agents (ch06)
    'IQL':        COLORS['blue'],
    'Nash-Q':     COLORS['red'],
    'WoLF-PHC':   COLORS['green'],
    # RLHF methods (ch09)
    'RLHF':       COLORS['blue'],
    'DPO':        COLORS['orange'],
    # Offline RL methods (ch08)
    'FQI':        COLORS['blue'],
    'CQL':        COLORS['red'],
    'IQL':        COLORS['green'],
    'BCQ':        COLORS['purple'],
    'BC':         COLORS['gray'],
}

# ---------------------------------------------------------------------------
# Figure size constants (inches)
# ---------------------------------------------------------------------------

FIG_SINGLE = (8, 5)
FIG_DOUBLE = (12, 5)
FIG_TRIPLE = (14, 5)
FIG_WIDE   = (16, 5)
FIG_SQUARE = (7, 7)

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

_RC = {
    'font.size':           11,
    'axes.labelsize':      12,
    'axes.titlesize':      13,
    'legend.fontsize':     9,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'axes.grid':           True,
    'grid.alpha':          0.3,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'lines.linewidth':     1.8,
    'figure.dpi':          300,
    'savefig.dpi':         300,
}


def apply_style():
    """Apply rcParams. Idempotent."""
    for k, v in _RC.items():
        mpl.rcParams[k] = v
