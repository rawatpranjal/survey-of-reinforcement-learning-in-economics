# Shared matplotlib style for all simulation scripts.
# Import and call apply_style() at the top of each script.

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette (tab10 defaults, named for convenience)
# ---------------------------------------------------------------------------

COLORS = {
    'blue':   '#1f77b4',
    'orange': '#ff7f0e',
    'green':  '#2ca02c',
    'red':    '#d62728',
    'purple': '#9467bd',
    'brown':  '#8c564b',
    'pink':   '#e377c2',
    'gray':   '#7f7f7f',
    'olive':  '#bcbd22',
    'cyan':   '#17becf',
    'black':  '#333333',
}

# ---------------------------------------------------------------------------
# Algorithm color mapping (consistent across all figures)
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    'VI':             '#333333',
    'PI':             '#7f7f7f',
    'Q-Learning':     '#1f77b4',
    'SARSA':          '#ff7f0e',
    'Expected SARSA': '#2ca02c',
    'MC Control':     '#d62728',
    'SARSA(\u03bb)':  '#9467bd',
    'Q(\u03bb)':      '#8c564b',
    'REINFORCE':      '#e377c2',
    'NPG':            '#e6550d',
    'PPO':            '#bcbd22',
    'DQN':            '#17becf',
    'SAC':            '#637939',
    'Actor-Critic':   '#6a3d9a',
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
