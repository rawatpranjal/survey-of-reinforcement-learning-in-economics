# The policy square: PG, NPG, the step-size sweep and the PI jump on one unit square
# Chapter 3 - The Theory of Reinforcement Learning
# Closed-form calculator on the running example. Under the one-logit-per-state
# parameterization pi(replace|s) = sigmoid(theta_s), policy space is the unit square.
# The script draws J(pi) contours, walks vanilla and natural policy gradient ascent,
# sweeps the NPG step size to the greedy vertex of the current policy, and overlays the
# two policy-iteration jumps. Everything is exact linear algebra; no Monte Carlo, no cache.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, CMAP_SEQ, FIG_SQUARE
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    exact_value,
    fisher_matrix,
    natural_gradient,
    policy_from_logits,
    policy_gradient,
    policy_kernel,
    policy_performance,
    q_values,
    solve_optimal,
)

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)

NU = np.array([1.0, 0.0])  # start at low mileage, the appendix convention
THETA0 = np.array([0.0, 0.0])  # the uniform policy, the center of the square
PG_ETA = 0.5
NPG_ETA = 0.5
N_STEPS = 300
ALPHA_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
CONV_TOL = 1e-3  # iterations to bring J* - J below this


def policy_point(theta):
    """A logit vector's image in the square, (pi(replace|low), pi(replace|high))."""
    b = policy_from_logits(theta)
    return np.array([b[LOW, REPLACE], b[HIGH, REPLACE]])


def exact_J(P, r, b):
    J, _, _ = policy_performance(P, r, GAMMA, b, NU)
    return J


def compute_data(force=None):
    P, r = build_mdp()
    V_star, greedy_star, Q_star = solve_optimal(P, r, GAMMA)
    J_star = float(NU @ V_star)

    print("Setup")
    print(f"  nu = [{NU[LOW]:.0f}, {NU[HIGH]:.0f}] (start at low mileage)")
    print(f"  V* = [low {V_star[LOW]:.4f}, high {V_star[HIGH]:.4f}], J* = {J_star:.4f}")
    print(
        f"  optimal vertex: (pi(replace|low), pi(replace|high)) = "
        f"({float(greedy_star[LOW]):.0f}, {float(greedy_star[HIGH]):.0f})"
    )
    print()

    # --- the Fisher matrix and the natural gradient at the uniform policy ---
    grad0, aux0 = policy_gradient(P, r, GAMMA, THETA0, NU)
    F0 = fisher_matrix(THETA0, aux0["rho"])
    ngrad0, _ = natural_gradient(P, r, GAMMA, THETA0, NU)
    det_F0 = float(np.linalg.det(F0))
    assert np.array_equal(F0, np.diag(np.diag(F0))), "Fisher is not diagonal"
    assert det_F0 > 0.0, "Fisher determinant is not positive"
    assert np.max(np.abs(ngrad0 - aux0["gap"])) < 1e-12, "NPG != action-value gap"
    print("At the uniform policy theta = (0, 0):")
    print(
        f"  occupancy rho = [{aux0['rho'][LOW]:.4f}, {aux0['rho'][HIGH]:.4f}] "
        f"(sums to 1/(1-gamma) = {aux0['rho'].sum():.4f})"
    )
    print(f"  Fisher F = diag({F0[0, 0]:.4f}, {F0[1, 1]:.4f}), det = {det_F0:.4f}")
    print(f"  gradient dJ/dtheta = [{grad0[LOW]:.4f}, {grad0[HIGH]:.4f}]")
    print(
        f"  action-value gaps Q(s,replace) - Q(s,keep) = "
        f"[{aux0['gap'][LOW]:.4f}, {aux0['gap'][HIGH]:.4f}]"
    )
    print(
        f"  natural gradient F^-1 dJ = [{ngrad0[LOW]:.4f}, {ngrad0[HIGH]:.4f}]"
        "  (equals the gaps exactly)"
    )
    print("  the gap at high mileage is negative at the uniform policy: under uniform")
    print("  continuation, keeping a high-mileage engine beats replacing it")
    print()

    # --- vanilla PG and NPG ascent paths ---
    paths = {}
    iters_to_tol = {}
    for name, eta, natural in (("PG", PG_ETA, False), ("NPG", NPG_ETA, True)):
        theta = THETA0.copy()
        pts = [policy_point(theta)]
        Js = [exact_J(P, r, policy_from_logits(theta))]
        hit = None
        for k in range(1, N_STEPS + 1):
            if natural:
                step, _ = natural_gradient(P, r, GAMMA, theta, NU)
            else:
                step, _ = policy_gradient(P, r, GAMMA, theta, NU)
            theta = theta + eta * step
            pts.append(policy_point(theta))
            Js.append(exact_J(P, r, policy_from_logits(theta)))
            if hit is None and J_star - Js[-1] < CONV_TOL:
                hit = k
        paths[name] = {"points": np.array(pts), "J": np.array(Js)}
        iters_to_tol[name] = hit
        print(
            f"{name}, step size {eta}: J rises {Js[0]:.4f} -> {Js[-1]:.4f} "
            f"over {N_STEPS} steps; J* - J < {CONV_TOL} first at "
            f"{'step ' + str(hit) if hit else 'never'}"
        )
    print()

    # --- the step-size sweep: one NPG step from the uniform policy ---
    b0 = policy_from_logits(THETA0)
    _, V0, _ = policy_performance(P, r, GAMMA, b0, NU)
    Q0 = q_values(P, r, V0, GAMMA)
    greedy0 = Q0.argmax(axis=1)
    greedy0_pt = np.array([float(greedy0[LOW]), float(greedy0[HIGH])])
    print("One NPG step from the uniform policy, sweeping the step size alpha:")
    print(
        f"  greedy policy of the CURRENT (uniform) iterate: "
        f"({['keep', 'replace'][greedy0[LOW]]}, {['keep', 'replace'][greedy0[HIGH]]}), "
        f"vertex ({greedy0_pt[0]:.0f}, {greedy0_pt[1]:.0f})"
    )
    sweep_pts = []
    for alpha in ALPHA_SWEEP:
        pt = policy_point(THETA0 + alpha * ngrad0)
        sweep_pts.append(pt)
        print(f"  alpha = {alpha:5.1f}: policy ({pt[0]:.4f}, {pt[1]:.4f})")
    sweep_pts = np.array(sweep_pts)
    gap_to_vertex = np.max(np.abs(sweep_pts[-1] - greedy0_pt))
    print(
        f"  at alpha = {ALPHA_SWEEP[-1]}, distance to the greedy vertex: "
        f"{gap_to_vertex:.6f}"
    )
    assert gap_to_vertex < 1e-3, "large-alpha NPG step does not reach the greedy vertex"
    print("  the large-alpha NPG step IS the policy-improvement half-step: it lands on")
    print("  the greedy vertex of the current policy, not on the optimal vertex")
    print()

    # --- the policy-iteration jumps ---
    # PI starts at the uniform policy itself; its first step evaluates uniform and
    # jumps to greedy0. Track the vertex sequence from there.
    pi_path = [policy_point(THETA0)]
    pi_seq = []
    b = b0
    for _ in range(10):
        _, V_b, _ = policy_performance(P, r, GAMMA, b, NU)
        Q_b = q_values(P, r, V_b, GAMMA)
        pi_next = Q_b.argmax(axis=1)
        pi_seq.append([int(pi_next[LOW]), int(pi_next[HIGH])])
        pt = np.array([float(pi_next[LOW]), float(pi_next[HIGH])])
        pi_path.append(pt)
        b_next = np.zeros((2, 2))
        b_next[LOW, pi_next[LOW]] = 1.0
        b_next[HIGH, pi_next[HIGH]] = 1.0
        if np.array_equal(b_next, b):
            break
        b = b_next
    pi_path = np.array(pi_path)
    names = [
        f"({['keep', 'replace'][a]}, {['keep', 'replace'][b_]})" for a, b_ in pi_seq
    ]
    print(f"Policy iteration from the uniform policy: {' -> '.join(names)}")
    assert pi_seq[-1] == [KEEP, REPLACE], "PI did not terminate at the optimal policy"
    # the final greedy step confirms the fixed point; improvement steps exclude it
    n_jumps = len(pi_seq) - 1
    print(f"  reaches the optimal vertex in {n_jumps} improvement steps")
    print()

    # --- J over the whole square, for the contours ---
    n_grid = 101
    grid = np.linspace(0.0, 1.0, n_grid)
    J_grid = np.zeros((n_grid, n_grid))
    for i, p_low in enumerate(grid):
        for j, p_high in enumerate(grid):
            b_ij = np.array([[1.0 - p_low, p_low], [1.0 - p_high, p_high]])
            P_b, r_b = policy_kernel(P, r, b_ij)
            J_grid[j, i] = float(NU @ exact_value(P_b, r_b, GAMMA))
    print(
        f"J over the {n_grid}x{n_grid} policy square: "
        f"min {J_grid.min():.4f} at "
        f"({grid[int(np.argmin(J_grid) % n_grid)]:.2f}, "
        f"{grid[int(np.argmin(J_grid) // n_grid)]:.2f}), "
        f"max {J_grid.max():.4f} (J* = {J_star:.4f})"
    )
    corner_names = [
        "(keep, keep)",
        "(replace, keep)",
        "(keep, replace)",
        "(replace, replace)",
    ]
    corners = [(0, 0), (n_grid - 1, 0), (0, n_grid - 1), (n_grid - 1, n_grid - 1)]
    print("  the four deterministic vertices:")
    for cname, (ci, cj) in zip(corner_names, corners):
        print(f"    {cname:20s} J = {J_grid[cj, ci]:.4f}")
    print()

    return {
        "grid": grid,
        "J_grid": J_grid,
        "J_star": J_star,
        "V_star": V_star,
        "greedy_star": np.asarray(greedy_star),
        "F0": F0,
        "grad0": grad0,
        "ngrad0": ngrad0,
        "gap0": aux0["gap"],
        "rho0": aux0["rho"],
        "paths": paths,
        "iters_to_tol": iters_to_tol,
        "sweep": sweep_pts,
        "alpha_sweep": ALPHA_SWEEP,
        "greedy0_pt": greedy0_pt,
        "pi_path": pi_path,
        "pi_seq": pi_seq,
        "n_jumps": n_jumps,
    }


def generate_outputs(data):
    import matplotlib.pyplot as plt

    grid, J_grid = data["grid"], data["J_grid"]
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    cs = ax.contourf(grid, grid, J_grid, levels=25, cmap=CMAP_SEQ, alpha=0.85)
    fig.colorbar(cs, ax=ax, label=r"$J(\pi) = V^\pi(\mathrm{low})$")

    pg = data["paths"]["PG"]["points"]
    npg = data["paths"]["NPG"]["points"]
    ax.plot(
        pg[:, 0],
        pg[:, 1],
        color=COLORS["red"],
        linewidth=2.0,
        label=f"policy gradient ({len(pg) - 1} steps)",
    )
    ax.plot(
        npg[:, 0],
        npg[:, 1],
        color=COLORS["green"],
        linewidth=2.0,
        label=f"natural gradient ({len(npg) - 1} steps)",
    )

    sweep = data["sweep"]
    ax.plot(
        sweep[:, 0],
        sweep[:, 1],
        "o",
        color=COLORS["orange"],
        markersize=5,
        label=r"one NPG step, $\alpha = \frac{1}{2}, 1, \ldots, 32$",
    )

    pi_path = data["pi_path"]
    ax.plot(
        pi_path[:, 0],
        pi_path[:, 1],
        "s--",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=7,
        label="policy iteration jumps",
    )

    ax.plot([0.5], [0.5], marker="o", color=COLORS["gray"], markersize=8, zorder=5)
    ax.annotate(
        "uniform start",
        (0.5, 0.5),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
    )
    g = data["greedy_star"]
    ax.plot(
        [float(g[LOW])],
        [float(g[HIGH])],
        marker="*",
        color=COLORS["black"],
        markersize=16,
        zorder=5,
        label=r"optimal vertex $(\mathrm{keep}, \mathrm{replace})$",
    )

    ax.set_xlabel(r"$\pi(\mathrm{replace} \mid \mathrm{low})$")
    ax.set_ylabel(r"$\pi(\mathrm{replace} \mid \mathrm{high})$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig_path = os.path.join(OUTPUT_DIR, "engine_policy_square.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # the consolidated table
    F0, grad0, ngrad0, gap0 = data["F0"], data["grad0"], data["ngrad0"], data["gap0"]
    it = data["iters_to_tol"]
    tex_path = os.path.join(OUTPUT_DIR, "engine_policy_square.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Policy-gradient objects on the running example at the uniform "
            "policy $\\theta = (0, 0)$, start distribution $\\nu = \\delta_{\\text{low}}$. "
            "The Fisher matrix is diagonal under the one-logit-per-state "
            "parameterization; its inverse is taken entrywise, never a pseudo-inverse. "
            "Convergence counts iterations until $J^\\star - J < 10^{-3}$ at step size "
            f"{PG_ETA}.}}\n"
        )
        f.write("\\label{tab:engine_policy_square}\n")
        f.write("\\begin{tabular}{lrr}\n\\hline\n")
        f.write(" & low & high \\\\\n\\hline\n")
        f.write(
            f"occupancy $\\rho(s)$ & {data['rho0'][LOW]:.4f} & {data['rho0'][HIGH]:.4f} \\\\\n"
        )
        f.write(
            f"Fisher diagonal $\\rho(s)\\pi_k(s)\\pi_r(s)$ & {F0[0, 0]:.4f} & {F0[1, 1]:.4f} \\\\\n"
        )
        f.write(
            f"gradient $\\partial J / \\partial \\theta_s$ & {grad0[LOW]:.4f} & {grad0[HIGH]:.4f} \\\\\n"
        )
        f.write(
            f"action-value gap $Q(s, \\text{{r}}) - Q(s, \\text{{k}})$ & {gap0[LOW]:.4f} & {gap0[HIGH]:.4f} \\\\\n"
        )
        f.write(
            f"natural gradient $F^{{-1}} \\nabla J$ & {ngrad0[LOW]:.4f} & {ngrad0[HIGH]:.4f} \\\\\n"
        )
        f.write("\\hline\n")
        pg_it = it["PG"] if it["PG"] is not None else f"not within {N_STEPS}"
        npg_it = it["NPG"] if it["NPG"] is not None else f"not within {N_STEPS}"
        f.write(
            f"\\multicolumn{{3}}{{l}}{{iterations to $J^\\star - J < 10^{{-3}}$: "
            f"policy gradient {pg_it}, natural gradient {npg_it}; "
            f"policy iteration: {data['n_jumps']} jumps}} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Policy square on the running example")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    print("=" * 70)
    print("THE POLICY SQUARE: PG, NPG, STEP-SIZE SWEEP AND PI ON THE RUNNING EXAMPLE")
    print("=" * 70)
    print()
    data = compute_data()
    if not args.data_only:
        generate_outputs(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
