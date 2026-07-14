# Hilbert-Space Projection Geometry (diagram-only)
# Appendix A - Mathematical Preliminaries
# Orthogonal projection onto a subspace is nonexpansive. The Pythagorean identity
# ||x||^2 = ||Pi x||^2 + ||x - Pi x||^2 gives ||Pi x|| <= ||x|| directly. This is the geometric
# fact behind the on-policy TD / projected-Bellman contraction in the linear-approximation proofs.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_SQUARE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

OUTPUT_DIR = os.path.dirname(__file__)

# Fixed geometry (a diagram, no Monte Carlo): subspace spanned by u, target point x.
U = np.array([1.0, 0.4])  # direction spanning the 1-D subspace
X = np.array([1.6, 1.7])  # the point being projected


def project(x, u):
    u = u / np.linalg.norm(u)
    return (x @ u) * u


def generate_outputs():
    u = U / np.linalg.norm(U)
    px = project(X, U)  # orthogonal projection of X onto span(u)
    resid = X - px

    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    # subspace line
    ts = np.linspace(-0.6, 2.2, 2)
    line = np.outer(ts, u)
    ax.plot(line[:, 0], line[:, 1], color=COLORS["gray"], linewidth=1.5, zorder=1)
    ax.annotate(
        r"$\mathrm{span}(\Phi)$",
        xy=(2.0 * u[0], 2.0 * u[1]),
        color=COLORS["gray"],
        fontsize=12,
    )
    # vectors from origin
    ax.annotate(
        "",
        xy=X,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"], lw=2),
    )
    ax.annotate(
        "",
        xy=px,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["green"], lw=2),
    )
    # residual x - Pi x
    ax.annotate(
        "",
        xy=X,
        xytext=px,
        arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2),
    )
    ax.text(X[0] + 0.03, X[1] + 0.03, r"$x$", color=COLORS["blue"], fontsize=13)
    ax.text(px[0] + 0.03, px[1] - 0.14, r"$\Pi x$", color=COLORS["green"], fontsize=13)
    ax.text(
        0.5 * (X[0] + px[0]) + 0.05,
        0.5 * (X[1] + px[1]),
        r"$x - \Pi x$",
        color=COLORS["red"],
        fontsize=12,
    )
    # right-angle marker at Pi x
    m = 0.12
    d1 = -u
    d2 = resid / np.linalg.norm(resid)
    corner = px + m * d1
    ax.plot(
        [corner[0], corner[0] + m * d2[0]],
        [corner[1], corner[1] + m * d2[1]],
        color=COLORS["black"],
        linewidth=1.0,
    )
    ax.plot(
        [px[0] + m * d2[0], corner[0] + m * d2[0]],
        [px[1] + m * d2[1], corner[1] + m * d2[1]],
        color=COLORS["black"],
        linewidth=1.0,
    )

    ax.set_xlim(-0.5, 2.3)
    ax.set_ylim(-0.3, 2.1)
    ax.set_aspect("equal")
    ax.set_title("Orthogonal projection is nonexpansive")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig_path = os.path.join(OUTPUT_DIR, "hilbert_projection.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # small numeric verification table (Pythagoras + nonexpansiveness for the drawn example)
    nx = np.linalg.norm(X)
    npx = np.linalg.norm(px)
    nres = np.linalg.norm(resid)
    print(f"  ||x||^2 = {nx**2:.4f}")
    print(f"  ||Pi x||^2 + ||x - Pi x||^2 = {npx**2 + nres**2:.4f}")
    print(f"  ||Pi x|| = {npx:.4f}")
    print(f"  ||x|| = {nx:.4f}")
    tex_path = os.path.join(OUTPUT_DIR, "hilbert_projection.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Numeric check of the Pythagorean identity and nonexpansiveness for the "
            "point and subspace drawn in Figure~\\ref{fig:prelim_hilbert}. The squared lengths "
            "add exactly, and the projection is no longer than the original.}\n"
        )
        f.write("\\label{tab:prelim_hilbert}\n")
        f.write("\\begin{tabular}{lc}\n\\hline\n")
        f.write("Quantity & Value \\\\\n\\hline\n")
        f.write(f"$\\|x\\|^2$ & {nx**2:.4f} \\\\\n")
        f.write(
            f"$\\|\\Pi x\\|^2 + \\|x - \\Pi x\\|^2$ & {npx**2 + nres**2:.4f} \\\\\n"
        )
        f.write(f"$\\|\\Pi x\\|$ & {npx:.4f} \\\\\n")
        f.write(f"$\\|x\\|$ & {nx:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hilbert projection geometry (diagram-only)"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    print("=" * 70)
    print("HILBERT-SPACE PROJECTION GEOMETRY (diagram-only)")
    print("=" * 70)
    print()
    if args.data_only:
        print("Diagram-only sim: no data to compute. Use default or --plots-only.")
        return
    generate_outputs()
    print("\nDone.")


if __name__ == "__main__":
    main()
