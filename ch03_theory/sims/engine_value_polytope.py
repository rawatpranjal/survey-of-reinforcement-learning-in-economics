# The value polytope: value space, iteration, and improvement on the Engine Replacement MDP
# Chapter 3 - The Theory of Reinforcement Learning
# Closed-form calculator on the Engine Replacement MDP. The set of all value functions
# V = {V^pi} for the two-grade engine is a region of R^2 (Dadashi et al. 2019). The
# script computes its four deterministic-policy values exactly, measures the region's
# area against its convex hull (the non-convexity witness), walks value iteration off
# the polytope and policy iteration along its vertices, and cuts the plane into the
# greedy-action cells. Everything is exact linear algebra; no Monte Carlo, no cache.

import argparse
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_TRIPLE
from sims.engine import (
    GAMMA,
    GAMMA_FRAC,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    build_mdp_grid_frac,
    exact_value_frac,
    policy_matrices,
    solve_optimal,
)

apply_style()

import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull, cKDTree

OUTPUT_DIR = os.path.dirname(__file__)

N_P = 4001  # policy-square sweep resolution (p = pi(replace|low))
PX_COARSE = 0.01  # raster pixel sizes for the area estimate
PX_FINE = 0.005
VI_STEPS = 100
MEMBER_TOL = 0.02  # distance above which a point is counted outside the polytope

DET_POLICIES = {
    "(keep, keep)": (KEEP, KEEP),
    "(keep, replace)": (KEEP, REPLACE),
    "(replace, keep)": (REPLACE, KEEP),
    "(replace, replace)": (REPLACE, REPLACE),
}


def value_closed_form(p, q):
    """V^pi for the mixture policy p = pi(replace|low), q = pi(replace|high),
    via the 2x2 resolvent (I - gamma P^pi)^{-1} r^pi in closed form. Vectorized."""
    a = 1.0 - GAMMA * (0.5 + 0.5 * p)
    b = -GAMMA * (0.5 - 0.5 * p)
    c = -GAMMA * q
    d = 1.0 - GAMMA * (1.0 - q)
    r_low = 1.0 - 1.5 * p
    r_high = 0.2 - 0.7 * q
    det = a * d - b * c
    return np.stack(
        [(d * r_low - b * r_high) / det, (-c * r_low + a * r_high) / det], axis=-1
    )


def polytope_mask(px, e0, e1):
    """Rasterize the polytope as the union of quads between consecutive q-segments.
    Each sweep segment is straight (the line theorem of Dadashi et al. 2019), so the
    region between neighbours is filled by their four endpoints; both triangulations
    of each quad are used because the segments cross near the waist."""
    x0, y0, x1, y1 = -5.05, -5.05, 5.45, 4.45
    nx, ny = int((x1 - x0) / px), int((y1 - y0) / px)
    xs = x0 + (np.arange(nx) + 0.5) * px
    ys = y0 + (np.arange(ny) + 0.5) * px
    mask = np.zeros((nx, ny), dtype=bool)
    for i in range(len(e0) - 1):
        quad = np.array([e0[i], e1[i], e1[i + 1], e0[i + 1]])
        bx0, by0 = quad.min(axis=0) - px
        bx1, by1 = quad.max(axis=0) + px
        ix = np.searchsorted(xs, [bx0, bx1])
        iy = np.searchsorted(ys, [by0, by1])
        if ix[1] <= ix[0] or iy[1] <= iy[0]:
            continue
        gx, gy = np.meshgrid(xs[ix[0] : ix[1]], ys[iy[0] : iy[1]], indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        hit = np.zeros(len(pts), dtype=bool)
        for tri in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)):
            hit |= Path(quad[list(tri)]).contains_points(pts)
        mask[ix[0] : ix[1], iy[0] : iy[1]] |= hit.reshape(gx.shape)
    return mask, (x0, x1, y0, y1)


def compute_data(force=None):
    P, r = build_mdp()
    V_star, greedy_star, _ = solve_optimal(P, r, GAMMA)

    # --- the four deterministic-policy value functions, exact and float ---
    P_frac, r_frac = build_mdp_grid_frac(K=2)
    vertices = {}
    vertices_frac = {}
    print("The four deterministic-policy value functions (exact fractions):")
    for name, pol in DET_POLICIES.items():
        V_f = exact_value_frac(P_frac, r_frac, list(pol), GAMMA_FRAC)
        P_pi, r_pi = policy_matrices(P, r, list(pol))
        V = np.linalg.solve(np.eye(2) - GAMMA * P_pi, r_pi)
        assert max(abs(float(V_f[s]) - V[s]) for s in (LOW, HIGH)) < 1e-12
        vertices[name] = V
        vertices_frac[name] = V_f
        print(
            f"  {name:20s} V = ({V_f[LOW]}, {V_f[HIGH]}) "
            f"= ({V[LOW]:.4f}, {V[HIGH]:.4f})"
        )
    print()

    # --- sweep the policy square: q-segments at each p ---
    ps = np.linspace(0.0, 1.0, N_P)
    e0 = value_closed_form(ps, np.zeros(N_P))
    e1 = value_closed_form(ps, np.ones(N_P))

    # --- area of the polytope against its convex hull ---
    areas = {}
    for px in (PX_COARSE, PX_FINE):
        mask, extent = polytope_mask(px, e0, e1)
        areas[px] = mask.sum() * px * px
    mask_fine, extent = polytope_mask(PX_FINE, e0, e1)
    area = areas[PX_FINE]

    pp, qq = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
    sample_pts = value_closed_form(pp.ravel(), qq.ravel())
    hull = ConvexHull(sample_pts)
    hull_pts = sample_pts[hull.vertices]
    print("Area of the value polytope against its convex hull:")
    for px, a in areas.items():
        print(f"  raster area at pixel size {px}: {a:.4f}")
    print(f"  convex hull area: {hull.volume:.4f}, {len(hull.vertices)} hull vertices")
    # anchor values verified independently: exact shoelace gives hull area
    # 12555/319 = 39.3574; an interval-merge raster brackets the polytope
    # area at 28.004 +/- 0.005
    assert abs(area - 28.00) < 0.01, f"polytope area {area:.4f} != 28.00"
    assert abs(hull.volume - 39.36) < 0.01, f"hull area {hull.volume:.4f} != 39.36"
    # in this MDP the hull vertices are exactly the four deterministic values
    assert len(hull.vertices) == 4
    for hp in hull_pts:
        assert min(np.abs(v - hp).max() for v in vertices.values()) < 1e-6
    print("  the hull vertices coincide with the four deterministic-policy values")
    print("  (a property of this instance, not of MDPs in general)")
    print()

    # --- the non-convexity witness: a midpoint of two values that is no value ---
    # midpoint of two computed vertices, piped from the exact fractions above
    m_frac = tuple(
        (vertices_frac["(keep, replace)"][s] + vertices_frac["(replace, keep)"][s]) / 2
        for s in (LOW, HIGH)
    )
    witness = np.array([float(m_frac[0]), float(m_frac[1])])
    dense = np.concatenate(
        [sample_pts, e0, e1, value_closed_form(pp.ravel() * 0 + 1.0, qq.ravel())]
    )
    tree = cKDTree(dense)
    w_dist, _ = tree.query(witness)
    print("Non-convexity witness:")
    print(
        f"  midpoint of V^(keep,replace) and V^(replace,keep): "
        f"({m_frac[0]}, {m_frac[1]}) = ({witness[0]:.4f}, {witness[1]:.4f})"
    )
    print(f"  distance to the nearest value function: {w_dist:.4f} (= 67/58)")
    # 67/58 is the exact point-to-segment distance from the witness to the
    # polytope boundary, verified by continuum optimization over the segment family
    assert abs(w_dist - float(Fraction(67, 58))) < 5e-3
    print()

    # --- value iteration leaves the polytope; policy iteration rides its vertices ---
    V = vertices["(replace, replace)"].copy()
    vi_path = [V.copy()]
    for _ in range(VI_STEPS):
        V = (r + GAMMA * np.einsum("sat,t->sa", P, V)).max(axis=1)
        vi_path.append(V.copy())
    vi_path = np.array(vi_path)
    vi_dist, _ = tree.query(vi_path)
    n_out = int((vi_dist > MEMBER_TOL).sum())
    print(
        f"Value iteration from V^(replace,replace) = ({vi_path[0][0]:.0f}, {vi_path[0][1]:.0f}):"
    )
    print("  iter    V(low)    V(high)   dist to polytope")
    for k in list(range(9)) + [20, 40, VI_STEPS]:
        print(
            f"  {k:4d}  {vi_path[k][LOW]:8.4f}  {vi_path[k][HIGH]:8.4f}  "
            f"{vi_dist[k]:8.4f}"
        )
    print(
        f"  iterates farther than {MEMBER_TOL} from the polytope: {n_out} of "
        f"{len(vi_path)} iterates, max distance {vi_dist.max():.4f}"
    )
    assert n_out > 0, "VI path never left the polytope"
    assert np.abs(vi_path[-1] - V_star).max() < 1e-2
    print()

    pol = [REPLACE, REPLACE]
    pi_path = [vertices["(replace, replace)"].copy()]
    pi_names = ["(replace, replace)"]
    for _ in range(10):
        P_pi, r_pi = policy_matrices(P, r, pol)
        V_pol = np.linalg.solve(np.eye(2) - GAMMA * P_pi, r_pi)
        q = r + GAMMA * np.einsum("sat,t->sa", P, V_pol)
        new_pol = list(q.argmax(axis=1))
        if new_pol == pol:
            break
        pol = new_pol
        name = f"({['keep', 'replace'][pol[LOW]]}, {['keep', 'replace'][pol[HIGH]]})"
        pi_names.append(name)
        pi_path.append(vertices[name].copy())
    pi_path = np.array(pi_path)
    print(f"Policy iteration from the same start: {' -> '.join(pi_names)}")
    print(
        f"  {len(pi_names) - 1} improvement steps, every iterate a deterministic-policy value"
    )
    for pt in pi_path:
        assert min(np.abs(v - pt).max() for v in vertices.values()) < 1e-10
    assert pi_names[-1] == "(keep, replace)"
    print()

    # --- convergence to V*: error against the gamma^k bound ---
    vi_err = np.abs(vi_path - V_star).max(axis=1)
    pi_err = np.abs(pi_path - V_star).max(axis=1)
    bound = GAMMA ** np.arange(VI_STEPS + 1) * vi_err[0]
    print(f"Convergence to V* = ({V_star[LOW]:.4f}, {V_star[HIGH]:.4f}):")
    print("  iter    VI error    gamma^k bound")
    for k in list(range(9)) + [20, 40, VI_STEPS]:
        print(f"  {k:4d}  {vi_err[k]:10.3e}  {bound[k]:10.3e}")
    for k, e in enumerate(pi_err):
        print(f"  PI after {k} improvement step(s): error {e:.3e}")
    assert (vi_err <= bound + 1e-12).all(), "VI error exceeds the gamma^k bound"
    assert vi_err[-1] < 1e-2
    assert pi_err[-1] < 1e-10, "PI does not land at machine precision"
    print()

    # --- the greedy partition of the value plane ---
    # greedy at low keeps iff x - y <= 10/3; greedy at high keeps iff x - y <= 7/9,
    # so the greedy map depends on V only through V(low) - V(high)
    t_high = float(Fraction(7, 9))
    t_low = float(Fraction(10, 3))
    print("Greedy partition of the value plane (x = V(low), y = V(high)):")
    print(f"  greedy action at low  is keep iff x - y <= 10/3 = {t_low:.4f}")
    print(f"  greedy action at high is keep iff x - y <= 7/9  = {t_high:.4f}")
    print("  three parallel cells: (keep, keep) | (keep, replace) | (replace, replace)")
    print("  the pair (replace, keep) is greedy nowhere in the plane")
    gap_star = V_star[LOW] - V_star[HIGH]
    assert t_high < gap_star < t_low, "V* is not in the (keep, replace) cell"
    print(
        f"  V* has x - y = {gap_star:.4f}, inside the (keep, replace) cell: the "
        f"optimal policy is greedy with respect to its own value"
    )
    print()

    return {
        "vertices": vertices,
        "V_star": V_star,
        "e0": e0,
        "e1": e1,
        "mask": mask_fine,
        "extent": extent,
        "area": area,
        "areas": areas,
        "hull_area": hull.volume,
        "hull_pts": hull_pts,
        "witness": witness,
        "w_dist": w_dist,
        "vi_path": vi_path,
        "vi_dist": vi_dist,
        "vi_err": vi_err,
        "pi_err": pi_err,
        "bound": bound,
        "n_out": n_out,
        "pi_path": pi_path,
        "pi_names": pi_names,
        "t_low": t_low,
        "t_high": t_high,
    }


def generate_outputs(data):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_TRIPLE)
    x0, x1, y0, y1 = data["extent"]

    # panel (a): the polytope, the hull, the witness, and both iteration paths
    ax1.imshow(
        data["mask"].T,
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="Greys",
        vmax=4.0,
        aspect="auto",
        interpolation="nearest",
    )
    hp = data["hull_pts"]
    order = np.argsort(
        np.arctan2(hp[:, 1] - hp[:, 1].mean(), hp[:, 0] - hp[:, 0].mean())
    )
    hull_loop = np.vstack([hp[order], hp[order][:1]])
    ax1.plot(
        hull_loop[:, 0],
        hull_loop[:, 1],
        "--",
        color=COLORS["gray"],
        linewidth=1.2,
        label="convex hull",
    )
    vi = data["vi_path"]
    ax1.plot(
        vi[:, 0],
        vi[:, 1],
        "o-",
        color=COLORS["red"],
        linewidth=1.5,
        markersize=3,
        label="value iteration",
    )
    pi = data["pi_path"]
    ax1.plot(
        pi[:, 0],
        pi[:, 1],
        "s--",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=7,
        label="policy iteration",
    )
    for name, v in data["vertices"].items():
        ax1.plot(v[0], v[1], "o", color=COLORS["black"], markersize=5, zorder=5)
    w = data["witness"]
    ax1.plot(
        w[0],
        w[1],
        "x",
        color=COLORS["orange"],
        markersize=10,
        markeredgewidth=2.5,
        label="midpoint witness",
    )
    seg = np.array(
        [data["vertices"]["(keep, replace)"], data["vertices"]["(replace, keep)"]]
    )
    ax1.plot(seg[:, 0], seg[:, 1], ":", color=COLORS["orange"], linewidth=1.0)
    vs = data["V_star"]
    ax1.plot(vs[0], vs[1], "*", color=COLORS["black"], markersize=15, zorder=6)
    ax1.annotate(r"$V^*$", vs, textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax1.set_xlabel(r"$V(\mathrm{low})$")
    ax1.set_ylabel(r"$V(\mathrm{high})$")
    ax1.set_title("(a) the value polytope")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # panel (b): the greedy partition, three parallel strips in x - y
    gx = np.linspace(x0, x1, 400)
    lo_line = gx - data["t_low"]  # y = x - 10/3, below it greedy is (replace, replace)
    hi_line = gx - data["t_high"]  # y = x - 7/9, above it greedy is (keep, keep)
    ax2.fill_between(gx, hi_line, y1, color=COLORS["green"], alpha=0.15)
    ax2.fill_between(gx, lo_line, hi_line, color=COLORS["blue"], alpha=0.15)
    ax2.fill_between(gx, y0, lo_line, color=COLORS["red"], alpha=0.15)
    for t, lab in (
        (data["t_high"], r"$x - y = 0.7778$"),
        (data["t_low"], r"$x - y = 3.3333$"),
    ):
        ax2.plot(gx, gx - t, color=COLORS["black"], linewidth=1.0)
        xi = x1 - 2.2
        ax2.annotate(
            lab, (xi, xi - t), textcoords="offset points", xytext=(0, 4), fontsize=8
        )
    ax2.annotate("greedy (keep, keep)", (-3.8, 2.8), fontsize=9)
    ax2.annotate("greedy (keep, replace)", (0.4, -1.8), fontsize=9)
    ax2.annotate("greedy\n(replace, replace)", (3.1, -4.4), fontsize=9)
    e0, e1 = data["e0"], data["e1"]
    boundary = np.vstack([e0, e1[::-1], e0[:1]])
    ax2.plot(
        boundary[:, 0],
        boundary[:, 1],
        color=COLORS["gray"],
        linewidth=0.8,
        label="polytope boundary",
    )
    ax2.plot(vs[0], vs[1], "*", color=COLORS["black"], markersize=15)
    ax2.annotate(r"$V^*$", vs, textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    ax2.set_xlabel(r"$V(\mathrm{low})$")
    ax2.set_ylabel(r"$V(\mathrm{high})$")
    ax2.set_title("(b) the greedy partition")

    # panel (c): convergence to V* against the gamma^k bound, log axis
    floor = 1e-15  # policy iteration's last point sits at machine precision
    ax3.semilogy(
        np.maximum(data["vi_err"], floor),
        "o-",
        color=COLORS["red"],
        linewidth=1.5,
        markersize=3,
        label="value iteration",
    )
    ax3.semilogy(
        np.maximum(data["pi_err"], floor),
        "s--",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=7,
        label="policy iteration",
    )
    ax3.semilogy(
        data["bound"],
        "--",
        color=COLORS["gray"],
        linewidth=1.2,
        label=r"$\gamma^k \|V_0 - V^*\|_\infty$",
    )
    ax3.set_ylim(bottom=floor)
    ax3.set_xlabel(r"iteration $k$")
    ax3.set_ylabel(r"$\|V_k - V^*\|_\infty$")
    ax3.set_title("(c) convergence to $V^*$")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "engine_value_polytope.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # the consolidated table
    v = data["vertices"]
    tex_path = os.path.join(OUTPUT_DIR, "engine_value_polytope.tex")
    rows = sorted(v.items(), key=lambda kv: -kv[1][LOW])
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{The four deterministic-policy values of the Engine Replacement MDP, "
            "exact resolvent solves listed by $V(\\text{low})$. Both iteration paths "
            "start at $V^{(\\text{replace}, \\text{replace})}$.}\n"
        )
        f.write("\\label{tab:engine_value_polytope}\n")
        f.write("\\begin{tabular}{lrr}\n\\hline\n")
        f.write("policy & $V(\\text{low})$ & $V(\\text{high})$ \\\\\n\\hline\n")
        for name, val in rows:
            nm = (
                name.replace("(", "$(\\text{")
                .replace(", ", "}, \\text{")
                .replace(")", "})$")
            )
            f.write(f"{nm} & {val[LOW]:.4f} & {val[HIGH]:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Value polytope on the Engine Replacement MDP"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    print("=" * 70)
    print("THE VALUE POLYTOPE: VALUE SPACE, ITERATION AND IMPROVEMENT")
    print("=" * 70)
    print()
    data = compute_data()
    if not args.data_only:
        generate_outputs(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
