# Elementary Objects (diagram-only)
# Appendix A - Mathematical Preliminaries
# Two panels backing the "Elementary Objects and Inequalities" entries: (a) two affine
# lines and their upper envelope, a convex piecewise-linear curve with a kink where the
# maximizer switches; (b) the unit balls of the L1, L2, and L-infinity norms in the
# plane, showing how the sup norm isolates the largest coordinate.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

OUTPUT_DIR = os.path.dirname(__file__)

# Fixed geometry (a diagram, no Monte Carlo): the two lines of the worked example.
# f1(x) = 1 + 0.5 x, f2(x) = 2 - x, crossing at x = 2/3 where both equal 4/3.


def generate_outputs():
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Panel (a): affine lines and their upper envelope
    ax = axes[0]
    x = np.linspace(-1.0, 2.5, 400)
    f1 = 1.0 + 0.5 * x
    f2 = 2.0 - x
    env = np.maximum(f1, f2)
    ax.plot(
        x,
        f1,
        color=COLORS["blue"],
        linewidth=1.2,
        linestyle="--",
        label=r"$f_1(x) = 1 + \frac{1}{2}x$",
    )
    ax.plot(
        x,
        f2,
        color=COLORS["orange"],
        linewidth=1.2,
        linestyle="--",
        label=r"$f_2(x) = 2 - x$",
    )
    ax.plot(
        x, env, color=COLORS["red"], linewidth=2.4, label=r"$g(x) = \max\{f_1, f_2\}$"
    )
    xk, yk = 2.0 / 3.0, 4.0 / 3.0
    ax.plot([xk], [yk], marker="o", color=COLORS["red"], markersize=6, zorder=5)
    ax.annotate(
        r"kink at $x = \frac{2}{3}$",
        xy=(xk, yk),
        xytext=(xk + 0.35, yk - 0.55),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("value")
    ax.set_title("(a) Upper envelope of two affine functions")
    ax.legend(loc="upper center", fontsize=8)

    # Panel (b): unit balls of the L1, L2, Linf norms
    ax = axes[1]
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        np.cos(t),
        np.sin(t),
        color=COLORS["orange"],
        linewidth=1.6,
        label=r"$\|x\|_2 = 1$",
    )
    ax.plot(
        [1, 0, -1, 0, 1],
        [0, 1, 0, -1, 0],
        color=COLORS["blue"],
        linewidth=1.6,
        label=r"$\|x\|_1 = 1$",
    )
    ax.plot(
        [1, -1, -1, 1, 1],
        [1, 1, -1, -1, 1],
        color=COLORS["green"],
        linewidth=1.6,
        label=r"$\|x\|_\infty = 1$",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color="#888888", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.5, zorder=0)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("(b) Unit balls of three norms")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "elementary_concepts.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Elementary objects diagrams (diagram-only)"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    print("=" * 70)
    print("ELEMENTARY OBJECTS (diagram-only)")
    print("=" * 70)
    print()
    if args.data_only:
        print("Diagram-only sim: no data to compute. Use default or --plots-only.")
        return
    generate_outputs()
    print("\nDone.")


if __name__ == "__main__":
    main()
