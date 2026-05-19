# Draw the Sutton blocking-maze layout for the §3 sim subsection.
# One-shot script. Reuses BlockingMaze constants. Two side-by-side panels
# (Phase 1, Phase 2) with grid lines, walls in row 2, S and G marked.

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS
apply_style()

from dyna_maze_env import BlockingMaze

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'dyna_maze_layout.png')


def draw_phase(ax, wall_cols, title):
    n_rows, n_cols = BlockingMaze.N_ROWS, BlockingMaze.N_COLS
    wall_row = BlockingMaze.WALL_ROW
    start = BlockingMaze.START
    goal = BlockingMaze.GOAL

    # Draw cells as a grid of squares.
    for r in range(n_rows):
        for c in range(n_cols):
            is_wall = (r == wall_row) and (c in wall_cols)
            color = COLORS['black'] if is_wall else 'white'
            rect = mpatches.Rectangle(
                (c, n_rows - 1 - r), 1, 1,
                facecolor=color, edgecolor=COLORS['gray'], linewidth=0.6,
            )
            ax.add_patch(rect)

    # Mark start (S) and goal (G).
    sr, sc = start
    gr, gc = goal
    ax.text(sc + 0.5, n_rows - 1 - sr + 0.5, 'S',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=COLORS['blue'])
    ax.text(gc + 0.5, n_rows - 1 - gr + 0.5, 'G',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=COLORS['green'])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, n_cols + 1))
    ax.set_yticks(np.arange(0, n_rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    draw_phase(ax1, BlockingMaze.PHASE1_WALL_COLS,
               'Phase 1 ($t \\leq 1000$): opening at column 8')
    draw_phase(ax2, BlockingMaze.PHASE2_WALL_COLS,
               'Phase 2 ($t > 1000$): opening at column 0')
    fig.suptitle('Sutton blocking maze: $6 \\times 9$ gridworld with phase-switchable wall',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
