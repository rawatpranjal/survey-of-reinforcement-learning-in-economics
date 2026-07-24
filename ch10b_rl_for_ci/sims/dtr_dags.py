"""
Dynamic Treatment Regime DAGs -- Chapter (OPE and Dynamic Treatment Effects)
Generates one publication-quality illustrative figure, dtr_dags.png:
  (a) point treatment with the two potential outcomes drawn explicitly,
  (b) two-stage regime with the time-varying confounder highlighted and the
      counterfactual outcome ghosted,
  (c) the history-state decision process induced by the same observed law.
Diagram-only script: no simulation, no cache.
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS

apply_style()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# DAG drawing utilities (pattern of ch10_causal/sims/identification_dags.py)
# ---------------------------------------------------------------------------

NODE_RADIUS = 0.15
FONT_SIZE = 11
ARROW_LW = 1.4
DASH_STYLE = (0, (5, 4))

REALIZED_STYLE = dict(
    facecolor="white",
    edgecolor="black",
    linewidth=1.4,
    linestyle="-",
    zorder=3,
)

# Grey dashed nodes denote potential (counterfactual) quantities here, not
# unobserved confounders; sequential ignorability is maintained throughout.
POTENTIAL_STYLE = dict(
    facecolor="#e0e0e0",
    edgecolor="black",
    linewidth=1.4,
    linestyle="--",
    zorder=3,
)


def _edge_endpoints(p1, p2, r1, r2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        return p1, p2
    ux, uy = dx / dist, dy / dist
    start = (p1[0] + r1 * ux, p1[1] + r1 * uy)
    end = (p2[0] - r2 * ux, p2[1] - r2 * uy)
    return start, end


def draw_node(ax, xy, label, realized=True, radius=NODE_RADIUS, fontsize=FONT_SIZE):
    style = REALIZED_STYLE if realized else POTENTIAL_STYLE
    circle = mpatches.Circle(xy, radius, **style)
    ax.add_patch(circle)
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=4,
        usetex=False,
    )


def draw_edge(
    ax,
    p1,
    p2,
    dashed=False,
    r1=NODE_RADIUS,
    r2=NODE_RADIUS,
    lw=ARROW_LW,
    color="black",
    curve=0.0,
    arrow=True,
):
    start, end = _edge_endpoints(p1, p2, r1, r2)
    props = dict(
        arrowstyle="->" if arrow else "-",
        lw=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    if dashed:
        props["linestyle"] = DASH_STYLE
    if abs(curve) > 1e-6:
        props["connectionstyle"] = f"arc3,rad={curve}"
    ax.annotate("", xy=end, xytext=start, arrowprops=props, zorder=2)


# ===================================================================
# The figure: three stacked panels
# ===================================================================


def make_dtr_dags():
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.grid(False)

    orange = COLORS["orange"]

    # ------------------------------------------------------------------
    # (a) Point treatment: two potential outcomes, one realized
    # ------------------------------------------------------------------
    ax = axes[0]
    pos = dict(
        S=(1.9, 1.05),
        A=(1.0, 0.45),
        Y1=(2.9, 0.80),
        Y0=(2.9, 0.10),
    )
    draw_node(ax, pos["S"], r"$S$")
    draw_node(ax, pos["A"], r"$A$")
    draw_node(ax, pos["Y1"], r"$Y^{*}(1)$", realized=False, radius=0.20, fontsize=9)
    draw_node(ax, pos["Y0"], r"$Y^{*}(0)$", realized=False, radius=0.20, fontsize=9)

    draw_edge(ax, pos["S"], pos["A"])
    draw_edge(ax, pos["S"], pos["Y1"])
    draw_edge(ax, pos["S"], pos["Y0"])
    # Dashed, noncausal selection links encode which fixed potential outcome is
    # revealed by A. Arrowheads would incorrectly imply that A causes Y*(a).
    draw_edge(ax, pos["A"], pos["Y1"], dashed=True, arrow=False)
    draw_edge(ax, pos["A"], pos["Y0"], dashed=True, arrow=False)
    ax.text(1.95, 0.76, r"$a=1$", fontsize=9, ha="center", color="0.35")
    ax.text(1.60, 0.08, r"$a=0$", fontsize=9, ha="center", color="0.35")
    ax.text(
        2.9,
        -0.42,
        r"observed: $Y = Y^{*}(A)$ (consistency)",
        fontsize=10,
        ha="center",
        color="0.25",
    )

    # legend, panel (a) only
    lx, ly = -0.45, -0.42
    ax.add_patch(mpatches.Circle((lx, ly), 0.08, **REALIZED_STYLE))
    ax.text(lx + 0.14, ly, "realized", fontsize=9, va="center")
    ax.add_patch(mpatches.Circle((lx, ly + 0.35), 0.08, **POTENTIAL_STYLE))
    ax.text(lx + 0.14, ly + 0.35, "potential (counterfactual)", fontsize=9, va="center")
    ax.plot(
        [lx - 0.08, lx + 0.08],
        [ly + 0.70, ly + 0.70],
        color="black",
        linestyle=DASH_STYLE,
        lw=1.4,
    )
    ax.text(
        lx + 0.14,
        ly + 0.70,
        "consistency selector (not causal)",
        fontsize=9,
        va="center",
    )

    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(-0.6, 1.45)
    ax.set_title(
        "(a) Point treatment: two potential outcomes, one realized", fontsize=11, pad=6
    )

    # ------------------------------------------------------------------
    # (b) Two-stage regime: the time-varying confounder
    # ------------------------------------------------------------------
    ax = axes[1]
    pos = dict(
        S1=(0.0, 0.45),
        A1=(0.95, 0.45),
        S2=(1.9, 0.45),
        A2=(2.85, 0.45),
        Y=(3.8, 0.45),
        Ystar=(3.8, 1.35),
    )
    draw_node(ax, pos["S1"], r"$S_1$")
    draw_node(ax, pos["A1"], r"$A_1$")
    draw_node(ax, pos["S2"], r"$S_2$")
    draw_node(ax, pos["A2"], r"$A_2$")
    draw_node(ax, pos["Y"], r"$Y$")
    draw_node(
        ax,
        pos["Ystar"],
        r"$Y^{*}(a_1, a_2)$",
        realized=False,
        radius=0.30,
        fontsize=7.5,
    )

    draw_edge(ax, pos["S1"], pos["A1"])
    draw_edge(ax, pos["A1"], pos["S2"], color=orange, lw=2.0)
    draw_edge(ax, pos["S2"], pos["A2"], color=orange, lw=2.0)
    draw_edge(ax, pos["A2"], pos["Y"])
    draw_edge(ax, pos["S1"], pos["S2"], curve=-0.35)
    draw_edge(ax, pos["S2"], pos["Y"], curve=-0.35, color=orange, lw=2.0)
    draw_edge(ax, pos["A1"], pos["Y"], curve=0.40)
    draw_edge(ax, pos["Y"], pos["Ystar"], dashed=True, arrow=False, r2=0.30)
    ax.text(
        4.05,
        0.9,
        r"$=Y$ when" + "\n" + r"$(A_1, A_2)=(a_1, a_2)$",
        fontsize=8,
        ha="left",
        color="0.35",
    )
    ax.text(
        1.9,
        -0.55,
        r"$S_2$ is caused by $A_1$ and confounds $A_2 \rightarrow Y$"
        " (highlighted)",
        fontsize=10,
        ha="center",
        color="0.25",
    )

    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-0.75, 1.75)
    ax.set_title(
        "(b) Two-stage regime: the time-varying confounder $S_2$", fontsize=11, pad=6
    )

    # ------------------------------------------------------------------
    # (c) The induced history-state decision process
    # ------------------------------------------------------------------
    ax = axes[2]
    pos = dict(
        H1=(0.0, 0.45),
        A1=(1.15, 0.45),
        H2=(2.55, 0.45),
        A2=(3.95, 0.45),
        Y=(5.10, 0.45),
    )
    draw_node(ax, pos["H1"], r"$\bar H_1$", radius=0.21)
    draw_node(ax, pos["A1"], r"$A_1$")
    draw_node(ax, pos["H2"], r"$\bar H_2$", radius=0.21)
    draw_node(ax, pos["A2"], r"$A_2$")
    draw_node(ax, pos["Y"], r"$Y$")

    draw_edge(ax, pos["H1"], pos["A1"], r1=0.21)
    draw_edge(ax, pos["H1"], pos["H2"], r1=0.21, r2=0.21, curve=-0.35)
    draw_edge(ax, pos["A1"], pos["H2"], r2=0.21)
    draw_edge(ax, pos["H2"], pos["A2"], r1=0.21)
    draw_edge(ax, pos["H2"], pos["Y"], r1=0.21, curve=-0.35)
    draw_edge(ax, pos["A2"], pos["Y"])

    ax.text(
        pos["H1"][0],
        -0.20,
        r"$\bar H_1=S_1$",
        fontsize=8.5,
        ha="center",
        color="0.35",
    )
    ax.text(
        pos["A1"][0],
        -0.20,
        r"$\pi_b(A_1\mid\bar H_1)$",
        fontsize=8.5,
        ha="center",
        color="0.35",
    )
    ax.text(
        pos["H2"][0],
        -0.20,
        r"$\bar H_2=(S_1,A_1,S_2)$",
        fontsize=8.5,
        ha="center",
        color="0.35",
    )
    ax.text(
        pos["A2"][0],
        -0.20,
        r"$\pi_b(A_2\mid\bar H_2)$",
        fontsize=8.5,
        ha="center",
        color="0.35",
    )
    ax.text(
        pos["Y"][0],
        -0.20,
        "terminal reward",
        fontsize=8.5,
        ha="center",
        color="0.35",
    )
    ax.text(
        2.55,
        -0.70,
        r"state $=\bar H_k$ makes the observed law Markov by construction"
        "\n"
        r"intermediate rewards $=0$; terminal reward $=Y$; $\gamma=1$",
        fontsize=8.7,
        ha="center",
        color="0.25",
    )

    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.95, 1.25)
    ax.set_title(
        "(c) The induced history-state decision process", fontsize=11, pad=6
    )

    fig.tight_layout(pad=1.2)
    outpath = os.path.join(os.path.dirname(__file__), "dtr_dags.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def generate_outputs():
    make_dtr_dags()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="No computation to cache (diagram-only script)",
    )
    parser.add_argument(
        "--plots-only", action="store_true", help="Runs normally (same as no flags)"
    )
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
