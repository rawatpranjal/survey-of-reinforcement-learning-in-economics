# Lagrangian Duality
# Appendix A - Mathematical Preliminaries
# A convex quadratic program min 1/2 x'Q x + c'x s.t. Ax <= b, solved through its
# dual by projected gradient ascent. The dual value is a lower bound on the primal
# optimum at every step (weak duality), and it closes to the primal optimum because
# the problem is convex with a strictly feasible point (Slater => strong duality,
# zero gap). The primal optimum p* is found independently by scipy (SLSQP) as ground
# truth; the dual iteration never uses it.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE, BENCH_STYLE

import numpy as np
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "lagrangian_duality"
CONFIG = {
    "n_vars": 8,
    "m_constraints": 5,
    "n_iters": 1500,
    "n_seeds": 15,
    "slack": 0.5,  # positive slack at a chosen interior point => Slater holds
    "seed_base": 55000,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Problem generation and the primal ground truth.
# ---------------------------------------------------------------------------


def make_problem(seed):
    rng = np.random.RandomState(seed)
    n, m = CONFIG["n_vars"], CONFIG["m_constraints"]
    M = rng.normal(size=(n, n))
    Q = M @ M.T + n * np.eye(n)  # symmetric positive definite
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    # choose an interior point and set b so it is strictly feasible (Slater)
    x_center = rng.normal(size=n)
    b = A @ x_center + CONFIG["slack"] * (1.0 + np.abs(rng.normal(size=m)))
    return Q, c, A, b


def primal_solution(Q, c, A, b):
    """Independent primal optimum via SLSQP (ground truth, not used by the dual)."""
    n = len(c)
    f = lambda x: 0.5 * x @ Q @ x + c @ x
    jac = lambda x: Q @ x + c
    cons = {"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}
    x0 = np.zeros(n)
    res = minimize(
        f,
        x0,
        jac=jac,
        constraints=[cons],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x, float(res.fun)


# ---------------------------------------------------------------------------
# Dual: g(lambda) = -1/2 (c + A'l)' Q^{-1} (c + A'l) - l'b, concave, l >= 0.
# Projected gradient ascent with step 1/L_dual, L_dual = lambda_max(A Q^{-1} A').
# ---------------------------------------------------------------------------


def dual_ascent(Q, c, A, b, n_iters):
    Qinv = np.linalg.inv(Q)
    AQiAT = A @ Qinv @ A.T
    L_dual = np.linalg.eigvalsh(AQiAT).max()  # smoothness of -g
    m = A.shape[0]
    lam = np.zeros(m)
    g_hist = np.zeros(n_iters + 1)

    def g_and_grad(lam):
        v = c + A.T @ lam
        x_star = -Qinv @ v  # argmin_x L(x, lam)
        g = -0.5 * v @ Qinv @ v - lam @ b
        grad = A @ x_star - b  # dg/dlam = A x*(lam) - b (constraint residual)
        return g, grad, x_star

    g_hist[0] = g_and_grad(lam)[0]
    for k in range(1, n_iters + 1):
        g, grad, _ = g_and_grad(lam)
        lam = np.maximum(lam + grad / L_dual, 0.0)  # project onto lam >= 0
        g_hist[k] = g_and_grad(lam)[0]
    g_final, _, x_final = g_and_grad(lam)
    return g_hist, lam, x_final


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    n_seeds = CONFIG["n_seeds"]
    n_iters = CONFIG["n_iters"]

    print(
        "Lagrangian duality: convex QP solved via its dual (projected gradient ascent)"
    )
    print(
        f"  vars={CONFIG['n_vars']}, constraints={CONFIG['m_constraints']}, "
        f"seeds={n_seeds}, dual iters={n_iters}"
    )
    print()

    gaps = np.zeros((n_seeds, n_iters + 1))
    final_gap = np.zeros(n_seeds)
    weak_ok = np.zeros(n_seeds, dtype=bool)  # g(lam_k) <= p* at every step
    compl_slack = np.zeros(n_seeds)  # max_i |lam_i (Ax - b)_i| at the solution
    n_active = np.zeros(n_seeds)
    p_stars = np.zeros(n_seeds)
    example = None
    for si in range(n_seeds):
        Q, c, A, b = make_problem(CONFIG["seed_base"] + si)
        x_p, p_star = primal_solution(Q, c, A, b)
        g_hist, lam, x_d = dual_ascent(Q, c, A, b, n_iters)
        gap = p_star - g_hist  # weak duality => gap >= 0; strong duality => -> 0
        gaps[si] = gap
        final_gap[si] = gap[-1]
        weak_ok[si] = bool(np.all(g_hist <= p_star + 1e-9))
        resid = A @ x_d - b
        compl_slack[si] = float(np.max(np.abs(lam * resid)))
        n_active[si] = int(np.sum(lam > 1e-6))
        p_stars[si] = p_star
        if si == 0:
            example = {"g_hist": g_hist, "p_star": p_star}
        print(
            f"  seed {si:2d}: p*={p_star:8.4f}, dual g_final={g_hist[-1]:8.4f}, "
            f"final gap={gap[-1]:.2e}, weak duality holds={weak_ok[si]}, "
            f"active={int(n_active[si])}, compl.slack={compl_slack[si]:.1e}"
        )

    print()
    print(
        f"  mean final duality gap: {final_gap.mean():.2e} "
        f"+/- {final_gap.std() / np.sqrt(n_seeds):.1e}"
    )
    print(
        f"  weak duality held on all iterates for {int(weak_ok.sum())}/{n_seeds} seeds"
    )
    print(f"  mean complementary-slackness residual: {compl_slack.mean():.2e}")

    return {
        "config": CONFIG,
        "gaps": gaps,
        "final_gap": final_gap,
        "final_gap_mean": float(final_gap.mean()),
        "final_gap_se": float(final_gap.std() / np.sqrt(n_seeds)),
        # magnitude of the final gap: the signed gap dips to solver-precision noise
        # (SLSQP's p* is only accurate to ~1e-12), so |gap| is the honest closeness measure
        "final_gap_abs_mean": float(np.abs(final_gap).mean()),
        "final_gap_abs_max": float(np.abs(final_gap).max()),
        "weak_ok_frac": float(weak_ok.mean()),
        "compl_slack_mean": float(compl_slack.mean()),
        "compl_slack_max": float(compl_slack.max()),
        "n_active_mean": float(n_active.mean()),
        "example": example,
    }


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "duality",
        CONFIG,
        _run_experiment,
        force=("duality" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    config = data["config"]
    gaps = data["gaps"]
    ex = data["example"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: dual value rises to p* from below (weak + strong duality) ---
    axA = axes[0]
    g = ex["g_hist"]
    k = np.arange(len(g))
    axA.plot(
        k, g, color=COLORS["blue"], linewidth=1.8, label=r"dual value $g(\lambda_k)$"
    )
    axA.axhline(ex["p_star"], **BENCH_STYLE, label=r"primal optimum $p^\star$")
    axA.set_xlabel("Dual ascent iteration $k$")
    axA.set_ylabel("value")
    axA.set_title("Dual rises to the primal optimum")
    axA.legend(loc="lower right")

    # --- Panel B: |duality gap| -> 0 across seeds (absolute value, since the signed
    #     gap dips to solver-precision noise at the floor) ----------------------
    axB = axes[1]
    absgaps = np.abs(gaps)
    for si in range(absgaps.shape[0]):
        axB.semilogy(
            np.maximum(absgaps[si], 1e-16),
            color=COLORS["blue"],
            linewidth=0.7,
            alpha=0.35,
        )
    axB.semilogy(
        np.maximum(absgaps.mean(axis=0), 1e-16),
        color=COLORS["red"],
        linewidth=1.8,
        label=r"mean $|$gap$|$",
    )
    axB.set_xlabel("Dual ascent iteration $k$")
    axB.set_ylabel(r"duality gap $|p^\star - g(\lambda_k)|$")
    axB.set_title("Gap closes to zero (Slater)")
    axB.legend(loc="upper right")

    fig_path = os.path.join(OUTPUT_DIR, "lagrangian_duality.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "lagrangian_duality.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Lagrangian duality for a convex quadratic program "
            "$\\min \\tfrac12 x^\\top Q x + c^\\top x$ s.t.\\ $Ax \\leq b$, solved through its "
            "dual by projected gradient ascent, over "
            + str(config["n_seeds"])
            + " random instances. The dual value is a lower bound on "
            "the primal optimum $p^\\star$ at every iteration (weak duality holds on all "
            "iterates), and the duality gap closes to zero because a strictly feasible point "
            "exists (Slater's condition, strong duality). The complementary-slackness residual "
            "$\\max_i |\\lambda_i (Ax - b)_i|$ vanishes at the solution. $p^\\star$ is computed "
            "independently by SLSQP.}\n"
        )
        f.write("\\label{tab:prelim_duality}\n")
        f.write("\\begin{tabular}{lc}\n\\hline\n")
        f.write("Quantity & Value \\\\\n\\hline\n")
        f.write(
            f"Final duality gap $|p^\\star - g(\\lambda)|$ & "
            f"$\\leq {data['final_gap_abs_max']:.1e}$ (solver precision) \\\\\n"
        )
        f.write(
            f"Weak duality holds (fraction of seeds) & "
            f"{data['weak_ok_frac']:.3f} \\\\\n"
        )
        f.write(
            f"Complementary slackness $\\max_i|\\lambda_i(Ax-b)_i|$ & "
            f"${data['compl_slack_mean']:.1e}$ (max ${data['compl_slack_max']:.1e}$) \\\\\n"
        )
        f.write(
            f"Active constraints (mean) & {data['n_active_mean']:.1f} "
            f"of {config['m_constraints']} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Lagrangian duality for a convex QP")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("LAGRANGIAN DUALITY: WEAK AND STRONG DUALITY FOR A CONVEX QP")
    print("=" * 70)
    print()
    print("Primal: min 1/2 x'Q x + c'x s.t. Ax <= b (Q positive definite).")
    print("Dual:   g(l) = -1/2 (c + A'l)' Q^{-1} (c + A'l) - l'b, l >= 0.")
    print("Weak duality: g(l) <= p* always. Strong duality (Slater): max_l g(l) = p*.")
    print()

    if force:
        print(f"Force recompute: {sorted(force)}")

    if args.plots_only:
        data = compute_data()
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
