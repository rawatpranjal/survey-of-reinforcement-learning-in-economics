# RL Theory Geometry (diagram-only)
# Appendix A - Mathematical Preliminaries
# Projection mismatch, occupancy coverage, Fisher trust regions, and rate classes.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

apply_style()

OUTPUT_DIR = os.path.dirname(__file__)


def draw_vector(ax, endpoint, color, label, linestyle="-"):
    ax.annotate(
        "",
        xy=endpoint,
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linestyle": linestyle,
            "linewidth": 1.8,
        },
    )
    ax.text(endpoint[0], endpoint[1], label, color=color, fontsize=9)


def projection_panel(ax):
    feature_direction = np.array([1.0, 0.45])
    feature_direction /= np.linalg.norm(feature_direction)
    target = np.array([0.7, 1.9])
    orthogonal = feature_direction * (target @ feature_direction)

    projection_direction = np.array([1.0, -0.2])
    system = np.column_stack((projection_direction, -feature_direction))
    alpha, beta = np.linalg.solve(system, -target)
    oblique = beta * feature_direction

    line_scale = np.linspace(-0.4, 4.0, 100)
    line = np.outer(line_scale, feature_direction)
    ax.plot(
        line[:, 0],
        line[:, 1],
        color=COLORS["gray"],
        linewidth=2.0,
        label=r"$\operatorname{span}(\Phi)$",
    )
    draw_vector(ax, target, COLORS["black"], r"$T^\pi V$")
    draw_vector(ax, orthogonal, COLORS["blue"], r"$\Pi_{d^\pi}T^\pi V$")
    draw_vector(ax, oblique, COLORS["red"], r"$\Pi_\mu T^\pi V$")
    ax.plot(
        [target[0], orthogonal[0]],
        [target[1], orthogonal[1]],
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.2,
    )
    ax.plot(
        [target[0], oblique[0]],
        [target[1], oblique[1]],
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlim(-0.35, 3.55)
    ax.set_ylim(-0.2, 2.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    ax.set_title("(a) Projection mismatch")
    ax.legend(loc="lower right", fontsize=7)


def coverage_panel(ax):
    states = np.arange(1, 9)
    target = np.array([0.08, 0.10, 0.13, 0.16, 0.18, 0.16, 0.11, 0.08])
    behavior = np.array([0.19, 0.18, 0.17, 0.16, 0.14, 0.10, 0.06, 0.00])
    width = 0.38
    ax.bar(
        states - width / 2,
        behavior,
        width,
        color=COLORS["gray"],
        label=r"behavior $\mu$",
    )
    ax.bar(
        states + width / 2,
        target,
        width,
        color=COLORS["orange"],
        label=r"target $d^\pi$",
    )
    ax.annotate(
        "support failure",
        xy=(8 + width / 2, target[-1]),
        xytext=(6.1, 0.205),
        fontsize=8,
        color=COLORS["red"],
        arrowprops={"arrowstyle": "->", "color": COLORS["red"], "linewidth": 1.0},
    )
    ax.set_xticks(states)
    ax.set_xlabel("state")
    ax.set_ylabel("probability mass")
    ax.set_ylim(0.0, 0.235)
    ax.set_title("(b) Occupancy and coverage")
    ax.legend(loc="upper right", fontsize=8)


def trust_region_panel(ax):
    circle = Ellipse(
        (0.0, 0.0),
        width=2.0,
        height=2.0,
        fill=False,
        edgecolor=COLORS["gray"],
        linewidth=1.8,
        linestyle="--",
        label=r"Euclidean ball",
    )
    fisher_ellipse = Ellipse(
        (0.0, 0.0),
        width=1.0,
        height=4.0,
        fill=False,
        edgecolor=COLORS["blue"],
        linewidth=2.0,
        label=r"Fisher ball",
    )
    ax.add_patch(circle)
    ax.add_patch(fisher_ellipse)

    gradient = np.array([1.0, 0.6])
    euclidean_step = gradient / np.linalg.norm(gradient)
    fisher = np.diag([4.0, 0.25])
    natural_direction = np.linalg.solve(fisher, gradient)
    natural_step = natural_direction / np.sqrt(natural_direction @ fisher @ natural_direction)
    draw_vector(ax, euclidean_step, COLORS["gray"], r"$\nabla J$")
    draw_vector(ax, natural_step, COLORS["blue"], r"$F^{-1}\nabla J$")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta\theta_1$")
    ax.set_ylabel(r"$\Delta\theta_2$")
    ax.set_title("(c) Local trust-region geometry")
    ax.legend(loc="lower right", fontsize=8)


def rates_panel(ax):
    iterations = np.arange(1, 201)
    ax.plot(
        iterations,
        0.92 ** (iterations - 1),
        color=COLORS["blue"],
        label=r"geometric $0.92^{k-1}$",
    )
    ax.plot(
        iterations,
        1.0 / iterations,
        color=COLORS["green"],
        label=r"inverse-linear $k^{-1}$",
    )
    ax.plot(
        iterations,
        1.0 / np.sqrt(iterations),
        color=COLORS["orange"],
        label=r"inverse-square-root $k^{-1/2}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration or sample size")
    ax.set_ylabel("normalized error")
    ax.set_title("(d) Rate classes")
    ax.legend(loc="lower left", fontsize=8)


def generate_outputs():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(0.78 * FIG_DOUBLE[0], 1.60 * FIG_DOUBLE[1]),
    )
    projection_panel(axes[0, 0])
    coverage_panel(axes[0, 1])
    trust_region_panel(axes[1, 0])
    rates_panel(axes[1, 1])
    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "rl_theory_geometry.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Output figure\t{output_path}")


def main():
    parser = argparse.ArgumentParser(description="RL theory geometry diagrams")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    print("RL THEORY GEOMETRY")
    print("Artifact type\tdiagram-only")
    print("Panels\t4")
    print("Random seeds\tnot applicable")
    if args.data_only:
        print("Computation\tno data generated")
        return
    generate_outputs()


if __name__ == "__main__":
    main()
