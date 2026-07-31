# Three Geometric Pictures for the Appendix
# Appendix A - Mathematical Preliminaries
# Diagram-only. Emits three separate figures, one per appendix section that needs it:
#   coverage_geometry.png  (A.5) which state-action pairs the log actually contains,
#                                drawn from the Engine Replacement MDP occupancies
#   norm_balls.png         (A.6) why "which norm" is not bookkeeping: the supremum-norm
#                                square against two weighted ellipses
#   curvature_geometry.png (A.8) a Euclidean ball against a Fisher ellipse, and an
#                                optimized value as the upper envelope of its family
# No Monte Carlo, so nothing is cached. The coverage panel reads its numbers from
# running_example.py rather than restating them.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE, FIG_SINGLE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_example import (  # noqa: E402
    KEEP,
    REPLACE,
    STATE_NAMES,
    ACTION_NAMES,
    build_mdp,
    policy_matrices,
    stochastic_policy_matrices,
    discounted_occupancy,
    compute_data as running_compute_data,
)

apply_style()

OUTPUT_DIR = os.path.dirname(__file__)


def coverage_figure():
    """A.5: the target's state-action occupancy against the log's, on the same axis."""
    params = running_compute_data()["params"]
    gamma = params["gamma"]
    P, r = build_mdp(
        params["r_keep_good"],
        params["r_keep_worn"],
        params["replace_cost"],
        params["degrade_prob"],
    )
    P_pi, _ = policy_matrices(P, r, [KEEP, REPLACE])
    P_mu, _, b = stochastic_policy_matrices(P, r, params["behavior_keep_prob"])
    nu = np.array([1.0, 0.0])
    occ_pi = discounted_occupancy(P_pi, gamma, nu)
    occ_mu = discounted_occupancy(P_mu, gamma, nu)

    pi_star = [KEEP, REPLACE]
    d_pi = np.zeros(4)
    d_mu = np.zeros(4)
    labels = []
    idx = 0
    for s in range(2):
        for a in range(2):
            d_pi[idx] = occ_pi[s] if pi_star[s] == a else 0.0
            d_mu[idx] = occ_mu[s] * b[s, a]
            labels.append(f"{STATE_NAMES[s]}\n{ACTION_NAMES[a]}")
            idx += 1

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, d_pi, w, color=COLORS["blue"], label=r"target $d^{\pi^\star}$")
    ax.bar(x + w / 2, d_mu, w, color=COLORS["orange"], label=r"log $d^\mu$")
    for i in range(4):
        if d_pi[i] > 0:
            ratio = d_pi[i] / d_mu[i]
            ax.annotate(
                f"ratio {ratio:.2f}",
                (i, max(d_pi[i], d_mu[i])),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=COLORS["black"],
            )
        else:
            ax.annotate(
                "target never\ngoes here",
                (i, d_mu[i]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=COLORS["gray"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("discounted state-action occupancy")
    ax.set_title("Where the log spends its weight, and where the target needs it")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.02)
    path = os.path.join(OUTPUT_DIR, "coverage_geometry.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")
    print("  target occupancy   :", np.round(d_pi, 4).tolist())
    print("  log occupancy      :", np.round(d_mu, 4).tolist())
    covered = all(d_mu[i] > 0 for i in range(4) if d_pi[i] > 0)
    print(f"  every pair the target uses is present in the log: {covered}")
    assert covered, "the log fails to cover a state-action pair the target uses"


def norm_balls_figure():
    """A.6: the supremum-norm ball against weighted L2 balls at two weightings."""
    fig, ax = plt.subplots(figsize=(FIG_SINGLE[0] * 0.75, FIG_SINGLE[1] * 0.9))
    # Supremum-norm unit ball in the plane is the square with corners (+-1, +-1).
    ax.plot(
        [-1, 1, 1, -1, -1],
        [-1, -1, 1, 1, -1],
        color=COLORS["black"],
        lw=1.8,
        label=r"$\|x\|_\infty = 1$",
    )
    theta = np.linspace(0, 2 * np.pi, 400)
    # A d-weighted L2 ball {x : d_1 x_1^2 + d_2 x_2^2 = 1} has semi-axes 1/sqrt(d_i).
    for d, color, name in [
        ((0.5, 0.5), COLORS["blue"], r"$d = (0.5, 0.5)$"),
        ((0.9474, 0.0526), COLORS["red"], r"$d = (0.947, 0.053)$"),
    ]:
        ax.plot(
            np.cos(theta) / np.sqrt(d[0]),
            np.sin(theta) / np.sqrt(d[1]),
            color=color,
            lw=1.6,
            label=name + r", $\|x\|_d = 1$",
        )
    ax.axhline(0, color=COLORS["gray"], lw=0.6)
    ax.axvline(0, color=COLORS["gray"], lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel(r"$x(\mathrm{good})$")
    ax.set_ylabel(r"$x(\mathrm{worn})$")
    ax.set_title("One vector, three sizes")
    ax.legend(loc="upper right", fontsize=8)
    path = os.path.join(OUTPUT_DIR, "norm_balls.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")
    x = np.array([1.0, 3.0])
    for d in [(0.5, 0.5), (0.9474, 0.0526)]:
        print(
            f"  x = (1, 3): sup-norm {np.max(np.abs(x)):.4f}, "
            f"d-weighted norm at d = {d} is {np.sqrt(d[0] * 1 + d[1] * 9):.4f}"
        )
    print("  the same vector is large under one weighting and small under the other")


def curvature_figure():
    """A.8: a Euclidean ball against a Fisher ellipse, and an upper envelope."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=COLORS["gray"],
        lw=1.6,
        label=r"$\|\Delta\theta\|_2 \leq 1$",
    )
    # A Fisher matrix with eigenvalues 4 and 1/4 rotated by 30 degrees. The trust region
    # {Delta : Delta^T F Delta <= 1} has semi-axes 1/sqrt(eigenvalue) along eigenvectors.
    ev = np.array([4.0, 0.25])
    phi = np.radians(30)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    F = R @ np.diag(ev) @ R.T
    pts = R @ np.vstack(
        [np.cos(theta) / np.sqrt(ev[0]), np.sin(theta) / np.sqrt(ev[1])]
    )
    ax.plot(
        pts[0],
        pts[1],
        color=COLORS["purple"],
        lw=1.8,
        label=r"$\Delta\theta^\top F \Delta\theta \leq 1$",
    )
    ax.set_aspect("equal")
    ax.axhline(0, color=COLORS["gray"], lw=0.5)
    ax.axvline(0, color=COLORS["gray"], lw=0.5)
    ax.set_xlabel(r"$\Delta\theta_1$")
    ax.set_ylabel(r"$\Delta\theta_2$")
    ax.set_title("(a) equal steps in parameters, unequal in distribution")
    ax.legend(loc="upper right", fontsize=8)
    print(
        f"  Fisher eigenvalues {ev.tolist()}, condition number {ev.max() / ev.min():.1f}"
    )
    print(f"  semi-axes of the trust region {np.round(1 / np.sqrt(ev), 4).tolist()}")
    assert abs(np.linalg.cond(F) - ev.max() / ev.min()) < 1e-9

    ax = axes[1]
    th = np.linspace(-2.0, 2.0, 400)
    # A family of affine payoffs and their upper envelope.
    members = [(-0.6, -1.4), (0.0, -0.2), (0.5, 0.6), (1.1, 1.6)]
    env = np.full_like(th, -np.inf)
    for i, (slope, intercept) in enumerate(members):
        y = intercept + slope * th
        ax.plot(
            th,
            y,
            color=COLORS["gray"],
            lw=0.9,
            alpha=0.8,
            label="family $f(a, \\theta)$" if i == 0 else None,
        )
        env = np.maximum(env, y)
    ax.plot(
        th,
        env,
        color=COLORS["green"],
        lw=2.2,
        label=r"$V(\theta) = \max_a f(a, \theta)$",
    )
    # Mark a point where the envelope is smooth and the active member is tangent.
    j = np.argmin(np.abs(th - 1.2))
    ax.plot(th[j], env[j], "o", color=COLORS["red"], ms=6, zorder=5)
    ax.annotate(
        "active member\nsupports here",
        (th[j], env[j]),
        textcoords="offset points",
        xytext=(-92, 6),
        fontsize=8,
        color=COLORS["red"],
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("value")
    ax.set_title("(b) the optimized value is an upper envelope")
    ax.legend(loc="upper left", fontsize=8)

    path = os.path.join(OUTPUT_DIR, "curvature_geometry.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")
    # The envelope is convex and each member lies weakly below it, which is the claim
    # the panel is making.
    for slope, intercept in members:
        assert np.all(intercept + slope * th <= env + 1e-12), (
            "a member rises above the envelope"
        )
    second = np.diff(env, 2)
    assert np.all(second >= -1e-9), "the upper envelope is not convex"
    print("  every member lies weakly below the envelope and the envelope is convex")


def compute_data(force=None):
    return {}


def generate_outputs(data=None):
    print("A.5 coverage")
    coverage_figure()
    print()
    print("A.6 norm balls")
    norm_balls_figure()
    print()
    print("A.8 curvature and envelope")
    curvature_figure()
    print()


def main():
    parser = argparse.ArgumentParser(description="Geometric pictures for the appendix")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-only", action="store_true")
    group.add_argument("--plots-only", action="store_true")
    parser.add_argument("--algo", type=str, action="append", default=None)
    args = parser.parse_args()
    print("=" * 70)
    print("GEOMETRIC PICTURES FOR THE APPENDIX (diagram-only)")
    print("=" * 70)
    print()
    if args.data_only:
        print("Diagram-only script: nothing to compute.")
        return
    generate_outputs()
    print("Done.")


if __name__ == "__main__":
    main()
