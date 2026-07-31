# Replacement budgets, occupancy measures, and KL tilting on the Engine Replacement MDP
# Chapter 11 - Quantile, Robust and Constrained Reinforcement Learning
# This deterministic calculation solves the constrained occupancy LP, checks the shadow
# price against budget perturbations, and computes the KL worst-case transition tilt.

import argparse
import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    discounted_occupancy,
    exact_value,
    policy_matrices,
    policy_kernel,
    solve_optimal,
)
from sims.plot_style import BENCH_STYLE, COLORS, FIG_DOUBLE, apply_style

apply_style()

import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = os.path.dirname(__file__)
SCRIPT_NAME = "engine_occupancy_kl"

# The appendix evaluates returns from a low-mileage engine. The figure uses a separate
# full-support reference distribution so the four deterministic occupancies are distinct.
EVALUATION_START = np.array([1.0, 0.0])
GEOMETRY_START = np.array([0.5, 0.5])
REPLACEMENT_BUDGET = 0.2
THETAS = np.array([10.0, 0.5])
SLOPE_STEPS = np.array([0.01, 0.005, 0.001, 0.0005])

DET_POLICIES = {
    "(keep, keep)": (KEEP, KEEP),
    "(keep, replace)": (KEEP, REPLACE),
    "(replace, keep)": (REPLACE, KEEP),
    "(replace, replace)": (REPLACE, REPLACE),
}


def flow_constraints(P, gamma, start):
    """Return A_eq and b_eq for normalized discounted state-action occupancies."""
    n_states, n_actions = P.shape[:2]
    A_eq = np.zeros((n_states, n_states * n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            A_eq[state, state * n_actions + action] += 1.0
        for previous_state in range(n_states):
            for action in range(n_actions):
                index = previous_state * n_actions + action
                A_eq[state, index] -= gamma * P[previous_state, action, state]
    b_eq = (1.0 - gamma) * np.asarray(start, dtype=float)
    return A_eq, b_eq


def solve_constrained_lp(P, rewards, gamma, start, budget):
    """Maximize discounted return with a cap on normalized replacement occupancy."""
    A_eq, b_eq = flow_constraints(P, gamma, start)
    replacement_cost = np.zeros_like(rewards)
    replacement_cost[:, REPLACE] = 1.0
    result = linprog(
        c=-(rewards / (1.0 - gamma)).ravel(),
        A_ub=replacement_cost.ravel()[None, :],
        b_ub=np.array([budget]),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Constrained occupancy LP failed: {result.message}")

    occupancy = result.x.reshape(rewards.shape)
    state_occupancy = occupancy.sum(axis=1)
    if np.any(state_occupancy <= 0.0):
        raise AssertionError("Optimal occupancy does not reach every state")
    policy = occupancy / state_occupancy[:, None]
    P_pi, r_pi = policy_kernel(P, rewards, policy)
    value = exact_value(P_pi, r_pi, gamma)
    flow_residual = A_eq @ result.x - b_eq
    if np.max(np.abs(flow_residual)) >= 1e-11:
        raise AssertionError("Constrained occupancy violates the flow equations")
    if abs(float(np.asarray(start) @ value) + result.fun) >= 1e-10:
        raise AssertionError("Occupancy objective does not match policy evaluation")
    lambda_star = float(-result.ineqlin.marginals[0])
    return {
        "return": float(-result.fun),
        "replacement_occupancy": float(occupancy[:, REPLACE].sum()),
        "occupancy": occupancy,
        "policy": policy,
        "lambda_star": lambda_star,
    }


def deterministic_results(P, rewards, gamma, start):
    """Evaluate every deterministic stationary policy from one start distribution."""
    rows = {}
    for name, policy in DET_POLICIES.items():
        P_pi, r_pi = policy_matrices(P, rewards, policy)
        value = exact_value(P_pi, r_pi, gamma)
        state_occupancy = discounted_occupancy(P_pi, gamma, start)
        occupancy = np.zeros_like(rewards)
        occupancy[np.arange(P.shape[0]), policy] = state_occupancy
        rows[name] = {
            "return": float(start @ value),
            "replacement_occupancy": float(occupancy[:, REPLACE].sum()),
            "occupancy": occupancy,
        }
    return rows


def assert_full_support(start):
    """Reject a start distribution that collapses the occupancy figure."""
    start = np.asarray(start, dtype=float)
    if np.any(start <= 0.0):
        raise AssertionError(
            "Occupancy-polytope figure requires positive mass on every starting state"
        )


def full_support_occupancy_vertices(P, rewards, gamma, start):
    """Compute the four distinct deterministic vertices for the occupancy figure."""
    assert_full_support(start)
    rows = deterministic_results(P, rewards, gamma, start)
    vertices = np.array(
        [
            [
                row["occupancy"][LOW, REPLACE],
                row["occupancy"][HIGH, REPLACE],
            ]
            for row in rows.values()
        ]
    )
    if len(np.unique(np.round(vertices, 12), axis=0)) != len(DET_POLICIES):
        raise AssertionError("Deterministic occupancy vertices are not distinct")
    return rows, vertices


def kl_worst_case(nominal, continuation_value, gamma, theta):
    """Exponential tilt for the KL multiplier problem."""
    nominal = np.asarray(nominal, dtype=float)
    continuation_value = np.asarray(continuation_value, dtype=float)
    log_weight = np.log(nominal) - gamma * continuation_value / theta
    log_weight -= log_weight.max()
    tilted = np.exp(log_weight)
    return tilted / tilted.sum()


def compute_data(force=None):
    P, rewards = build_mdp()
    V_star, greedy_star, _ = solve_optimal(P, rewards, GAMMA)

    print("Parameters")
    print(f"  discount factor                 {GAMMA:.4f}")
    print(
        "  evaluation start               "
        f"[{EVALUATION_START[LOW]:.1f}, {EVALUATION_START[HIGH]:.1f}]"
    )
    print(
        "  geometry start                 "
        f"[{GEOMETRY_START[LOW]:.1f}, {GEOMETRY_START[HIGH]:.1f}]"
    )
    print(f"  normalized replacement budget  {REPLACEMENT_BUDGET:.4f}")
    print()

    evaluation_det = deterministic_results(P, rewards, GAMMA, EVALUATION_START)
    geometry_det, geometry_vertices = full_support_occupancy_vertices(
        P, rewards, GAMMA, GEOMETRY_START
    )

    print("Deterministic policies from the low-mileage start")
    print("  policy                 return    replacement occupancy    feasible")
    feasible = []
    for name, row in evaluation_det.items():
        is_feasible = row["replacement_occupancy"] <= REPLACEMENT_BUDGET + 1e-12
        if is_feasible:
            feasible.append(row["return"])
        print(
            f"  {name:22s} {row['return']:8.4f}"
            f" {row['replacement_occupancy']:24.4f}"
            f"    {'yes' if is_feasible else 'no'}"
        )
    best_deterministic = max(feasible)
    assert abs(best_deterministic - 3.4545454545454546) < 1e-12
    print(f"  best feasible deterministic return  {best_deterministic:.4f}")
    print()

    low_start_solution = solve_constrained_lp(
        P, rewards, GAMMA, EVALUATION_START, REPLACEMENT_BUDGET
    )
    geometry_solution = solve_constrained_lp(
        P, rewards, GAMMA, GEOMETRY_START, REPLACEMENT_BUDGET
    )
    assert abs(low_start_solution["return"] - 4.672727272727273) < 1e-12
    assert abs(low_start_solution["policy"][HIGH, REPLACE] - 0.4074074074074074) < 1e-12
    assert abs(low_start_solution["replacement_occupancy"] - REPLACEMENT_BUDGET) < 1e-12
    assert abs(low_start_solution["lambda_star"] - 6.090909090909091) < 1e-10

    print("Constrained occupancy solutions")
    print(
        "  start             return    pi(replace|low)    pi(replace|high)    cost    lambda"
    )
    for label, solution in (
        ("low only", low_start_solution),
        ("uniform", geometry_solution),
    ):
        print(
            f"  {label:12s} {solution['return']:9.4f}"
            f" {solution['policy'][LOW, REPLACE]:18.4f}"
            f" {solution['policy'][HIGH, REPLACE]:19.4f}"
            f" {solution['replacement_occupancy']:7.4f}"
            f" {solution['lambda_star']:9.4f}"
        )
    print()

    budget_slopes = []
    print("Budget sensitivity around the cap")
    print("  half-width       V(B-h)       V(B+h)    centered slope")
    for step in SLOPE_STEPS:
        lower = solve_constrained_lp(
            P, rewards, GAMMA, EVALUATION_START, REPLACEMENT_BUDGET - step
        )
        upper = solve_constrained_lp(
            P, rewards, GAMMA, EVALUATION_START, REPLACEMENT_BUDGET + step
        )
        slope = (upper["return"] - lower["return"]) / (2.0 * step)
        budget_slopes.append(slope)
        print(
            f"  {step:10.4f} {lower['return']:12.6f}"
            f" {upper['return']:12.6f} {slope:17.6f}"
        )
    budget_slopes = np.asarray(budget_slopes)
    slope_error = np.max(np.abs(budget_slopes - low_start_solution["lambda_star"]))
    assert slope_error < 1e-9, "Budget slope does not match the LP multiplier"
    print(f"  maximum slope versus multiplier error  {slope_error:.3e}")
    print()

    nominal = P[LOW, KEEP]
    tilts = {}
    print("KL worst-case continuation kernels")
    print("  theta      p(low)     p(high)    shift low    shift high")
    for theta in THETAS:
        tilted = kl_worst_case(nominal, V_star, GAMMA, theta)
        tilts[float(theta)] = tilted
        shift = tilted - nominal
        print(
            f"  {theta:5.1f} {tilted[LOW]:11.6f} {tilted[HIGH]:11.6f}"
            f" {shift[LOW]:12.6f} {shift[HIGH]:13.6f}"
        )
    assert abs(tilts[10.0][HIGH] - nominal[HIGH] - 0.02325906) < 1e-8
    assert abs(tilts[0.5][HIGH] - nominal[HIGH] - 0.36553792) < 1e-8
    print()

    print("Degenerate-start guard")
    try:
        full_support_occupancy_vertices(P, rewards, GAMMA, np.array([1.0, 0.0]))
    except AssertionError as error:
        degenerate_guard_message = str(error)
        print(f"  raised AssertionError as required  {degenerate_guard_message}")
    else:
        raise AssertionError("Degenerate-start guard did not fail")
    print()

    geometry_point = np.array(
        [
            geometry_solution["occupancy"][LOW, REPLACE],
            geometry_solution["occupancy"][HIGH, REPLACE],
        ]
    )
    hull = ConvexHull(geometry_vertices)
    ordered_vertices = geometry_vertices[hull.vertices]

    return {
        "P": P,
        "rewards": rewards,
        "V_star": V_star,
        "greedy_star": greedy_star,
        "evaluation_det": evaluation_det,
        "geometry_det": geometry_det,
        "geometry_vertices": geometry_vertices,
        "ordered_vertices": ordered_vertices,
        "geometry_point": geometry_point,
        "low_start_solution": low_start_solution,
        "geometry_solution": geometry_solution,
        "geometry_start": GEOMETRY_START.copy(),
        "budget_slopes": budget_slopes,
        "nominal": nominal,
        "tilts": tilts,
        "degenerate_guard_message": degenerate_guard_message,
    }


def generate_outputs(data):
    assert_full_support(data["geometry_start"])
    fig, (ax_occ, ax_kl) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    polygon = np.vstack([data["ordered_vertices"], data["ordered_vertices"][0]])
    ax_occ.fill(
        polygon[:, 0],
        polygon[:, 1],
        color=COLORS["blue"],
        alpha=0.16,
        label="Feasible occupancy polytope",
    )
    ax_occ.plot(polygon[:, 0], polygon[:, 1], color=COLORS["blue"])

    for (name, _), point in zip(DET_POLICIES.items(), data["geometry_vertices"]):
        ax_occ.scatter(point[0], point[1], color=COLORS["black"], s=35, zorder=4)
        short_name = name.replace("replace", "R").replace("keep", "K")
        ax_occ.annotate(
            short_name,
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    budget_grid = np.linspace(0.0, REPLACEMENT_BUDGET, 200)
    ax_occ.fill_between(
        budget_grid,
        0.0,
        REPLACEMENT_BUDGET - budget_grid,
        color=COLORS["green"],
        alpha=0.20,
        label="Replacement budget",
    )
    ax_occ.plot(
        budget_grid,
        REPLACEMENT_BUDGET - budget_grid,
        color=COLORS["green"],
        linestyle="--",
        label=r"$x_{\mathrm{low},R}+x_{\mathrm{high},R}=0.2$",
    )
    ax_occ.scatter(
        data["geometry_point"][0],
        data["geometry_point"][1],
        color=COLORS["red"],
        marker="*",
        s=150,
        zorder=5,
        label="Constrained optimum",
    )
    ax_occ.set_xlabel(r"$x(\mathrm{low},\mathrm{replace})$")
    ax_occ.set_ylabel(r"$x(\mathrm{high},\mathrm{replace})$")
    ax_occ.set_title("(a) Full-support occupancy geometry")
    ax_occ.set_xlim(-0.03, 1.0)
    ax_occ.set_ylim(-0.03, 0.39)
    ax_occ.legend(loc="upper right")

    theta_grid = np.geomspace(0.25, 20.0, 300)
    p_high = np.array(
        [
            kl_worst_case(data["nominal"], data["V_star"], GAMMA, theta)[HIGH]
            for theta in theta_grid
        ]
    )
    ax_kl.plot(
        theta_grid,
        p_high,
        color=COLORS["purple"],
        label=r"Worst-case $p(\mathrm{high})$",
    )
    ax_kl.axhline(
        data["nominal"][HIGH],
        **BENCH_STYLE,
        label=r"Nominal $p(\mathrm{high})=0.5$",
    )
    for theta in sorted(data["tilts"]):
        probability = data["tilts"][theta][HIGH]
        ax_kl.scatter(
            theta,
            probability,
            color=COLORS["red"],
            s=45,
            zorder=4,
        )
        ax_kl.annotate(
            rf"$\theta={theta:g}$",
            (theta, probability),
            xytext=(6, -14 if theta > 1 else 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax_kl.set_xscale("log")
    ax_kl.set_xlabel(r"KL multiplier $\theta$")
    ax_kl.set_ylabel(r"Worst-case probability of high mileage")
    ax_kl.set_title("(b) Exponential transition tilt")
    ax_kl.set_ylim(0.47, 0.91)
    ax_kl.legend(loc="upper right")

    fig.tight_layout()
    figure_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    low = data["low_start_solution"]
    geometry = data["geometry_solution"]
    rows = [
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Calculation & Setting & Estimate & Check \\",
        r"\hline",
        (
            "CMDP & Best deterministic return"
            f" & {max(row['return'] for row in data['evaluation_det'].values() if row['replacement_occupancy'] <= REPLACEMENT_BUDGET + 1e-12):.4f}"
            r" & feasible \\"
        ),
        (
            f"CMDP & Randomized return & {low['return']:.4f}"
            f" & $\\pi(R\\mid\\mathrm{{high}})={low['policy'][HIGH, REPLACE]:.4f}$ \\\\"
        ),
        (
            f"CMDP & Replacement occupancy & {low['replacement_occupancy']:.4f}"
            f" & budget ${REPLACEMENT_BUDGET:.4f}$ \\\\"
        ),
        (
            f"CMDP & Shadow price $\\lambda^*$ & {low['lambda_star']:.4f}"
            f" & slope ${data['budget_slopes'].mean():.4f}$ \\\\"
        ),
        (
            f"Geometry & Uniform-start return & {geometry['return']:.4f}"
            f" & $\\pi(R\\mid\\mathrm{{high}})={geometry['policy'][HIGH, REPLACE]:.4f}$ \\\\"
        ),
        (
            f"KL tilt & $\\theta=10$ high-state probability"
            f" & {data['tilts'][10.0][HIGH]:.4f}"
            f" & shift ${data['tilts'][10.0][HIGH] - data['nominal'][HIGH]:+.4f}$ \\\\"
        ),
        (
            f"KL tilt & $\\theta=0.5$ high-state probability"
            f" & {data['tilts'][0.5][HIGH]:.4f}"
            f" & shift ${data['tilts'][0.5][HIGH] - data['nominal'][HIGH]:+.4f}$ \\\\"
        ),
        r"\hline",
        r"\end{tabular}",
    ]
    table_path = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_table.tex")
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write("\n".join(rows) + "\n")

    print("Output files")
    print(f"  figure  {figure_path}")
    print(f"  table   {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Engine replacement occupancy and KL calculations"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="compute checks without writing outputs",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="regenerate deterministic outputs after recomputing checks",
    )
    args = parser.parse_args()
    if args.data_only and args.plots_only:
        parser.error("--data-only and --plots-only are mutually exclusive")

    data = compute_data()
    if args.data_only:
        print("Data-only mode completed. No output files were written.")
        return
    generate_outputs(data)


if __name__ == "__main__":
    main()
