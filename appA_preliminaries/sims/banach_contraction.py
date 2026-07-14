# Banach Fixed-Point Theorem: geometric convergence of a gamma-contraction
# Appendix A - Mathematical Preliminaries
# Iterates the policy-evaluation Bellman operator (a gamma-contraction) and checks
# that the error decays as gamma^k, the rate the Banach fixed-point theorem predicts.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_SINGLE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "banach_contraction"
CONFIG = {
    "n_states": 50,
    "gammas": [0.5, 0.7, 0.9, 0.99],
    "n_iters": 3000,
    "n_seeds": 30,
    "tol": 1e-10,
    "version": 2,
}

OUTPUT_DIR = os.path.dirname(__file__)

GAMMA_COLORS = {
    0.5: COLORS["blue"],
    0.7: COLORS["orange"],
    0.9: COLORS["green"],
    0.99: COLORS["red"],
}

# ---------------------------------------------------------------------------
# Operator: policy-evaluation Bellman operator T V = r + gamma P V (a gamma-contraction
# in the sup norm, since P is row-stochastic so ||P||_inf = 1).
# ---------------------------------------------------------------------------


def random_mdp(n_states, seed):
    """Random row-stochastic transition matrix P and bounded reward r."""
    rng = np.random.RandomState(seed)
    P = rng.dirichlet(np.ones(n_states), size=n_states)  # each row sums to 1
    r = rng.uniform(-1.0, 1.0, size=n_states)
    return P, r


def iterate_operator(P, r, gamma, V0, n_iters):
    """Return the sup-norm error ||V_k - V*||_inf for k = 0, ..., n_iters.

    V* solves V = r + gamma P V exactly, i.e. V* = (I - gamma P)^{-1} r.
    """
    n = len(r)
    V_star = np.linalg.solve(np.eye(n) - gamma * P, r)
    V = V0.copy()
    err = np.zeros(n_iters + 1)
    err[0] = np.max(np.abs(V - V_star))
    for k in range(1, n_iters + 1):
        V = r + gamma * (P @ V)  # one application of the Bellman operator T
        err[k] = np.max(np.abs(V - V_star))
    return err, V_star


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    n_states = CONFIG["n_states"]
    gammas = CONFIG["gammas"]
    n_iters = CONFIG["n_iters"]
    n_seeds = CONFIG["n_seeds"]
    tol = CONFIG["tol"]

    print("Banach fixed-point: policy-evaluation operator T V = r + gamma P V")
    print(f"  States: {n_states}, seeds: {n_seeds}, iters: {n_iters}")
    print(f"  gammas: {gammas}")
    print()

    results = {}
    for gamma in gammas:
        errs = np.zeros((n_seeds, n_iters + 1))
        ratios = np.zeros(n_seeds)  # measured per-step contraction factor
        iters_to_tol = np.zeros(n_seeds)
        for si in range(n_seeds):
            P, r = random_mdp(n_states, seed=100 + si)
            err, _ = iterate_operator(P, r, gamma, np.zeros(n_states), n_iters)
            errs[si] = err
            # measured contraction factor: geometric mean of err[k+1]/err[k] over
            # the steps where the error is still above machine noise
            mask = err[:-1] > tol
            steps = np.maximum(mask.sum(), 1)
            ratios[si] = (err[steps] / err[0]) ** (1.0 / steps) if err[0] > 0 else 0.0
            below = np.where(err < tol)[0]
            iters_to_tol[si] = below[0] if len(below) > 0 else np.nan

        mean_err = errs.mean(axis=0)
        se_err = errs.std(axis=0) / np.sqrt(n_seeds)
        n_reached = int(np.sum(~np.isnan(iters_to_tol)))
        results[str(gamma)] = {
            "mean_err": mean_err,
            "se_err": se_err,
            "ratio_mean": float(ratios.mean()),
            "ratio_se": float(ratios.std() / np.sqrt(n_seeds)),
            "iters_to_tol_mean": float(np.nanmean(iters_to_tol))
            if n_reached
            else float("nan"),
            "iters_to_tol_count": n_reached,
            # theoretical iterations to reach tol from the mean initial error:
            "iters_pred": float(np.log(tol / mean_err[0]) / np.log(gamma))
            if mean_err[0] > 0
            else float("nan"),
        }
        print(
            f"  gamma={gamma:.2f}: measured contraction factor = "
            f"{results[str(gamma)]['ratio_mean']:.4f} +/- {results[str(gamma)]['ratio_se']:.4f} "
            f"(theory {gamma:.2f}); iters to tol = {results[str(gamma)]['iters_to_tol_mean']:.1f}"
        )

    return {"results": results, "config": CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "banach",
        CONFIG,
        _run_experiment,
        force=("banach" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    results = data["results"]
    config = data["config"]
    gammas = config["gammas"]
    tol = config["tol"]

    # --- Figure: log-scale error decay with the gamma^k bound overlaid ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for gamma in gammas:
        r = results[str(gamma)]
        mean = r["mean_err"]
        k = np.arange(len(mean))
        color = GAMMA_COLORS[gamma]
        ax.semilogy(k, mean, color=color, linewidth=1.8, label=f"$\\gamma = {gamma}$")
        # theoretical bound gamma^k * ||V_0 - V*||
        bound = mean[0] * (gamma**k)
        ax.semilogy(k, bound, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|V_k - V^*\|_\infty$")
    ax.set_title("Banach contraction: error decays as $\\gamma^k$")
    ax.set_ylim(tol, None)
    ax.legend(loc="upper right", title="solid: measured, dashed: $\\gamma^k$ bound")

    fig_path = os.path.join(OUTPUT_DIR, "banach_contraction.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---
    tex_path = os.path.join(OUTPUT_DIR, "banach_contraction.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Banach fixed-point convergence of the policy-evaluation operator "
            "$TV = r + \\gamma P V$ on a random "
            + str(config["n_states"])
            + "-state Markov reward process. Measured contraction factor is the geometric mean of "
            "$\\|V_{k+1}-V^*\\|_\\infty / \\|V_k-V^*\\|_\\infty$. Iterations to $10^{-10}$ are "
            "reported both as measured (mean over the seeds that reach tolerance within "
            + str(config["n_iters"])
            + " iterations) and as predicted from $\\log(\\text{tol}/\\text{err}_0)/\\log\\gamma$. "
            "Mean $\\pm$ SE over " + str(config["n_seeds"]) + " random MDPs.}\n"
        )
        f.write("\\label{tab:prelim_banach}\n")
        f.write("\\begin{tabular}{ccccc}\n\\hline\n")
        f.write(
            "$\\gamma$ & Measured factor & Theory ($\\gamma$) & "
            "Iterations to $10^{-10}$ (measured) & (predicted) \\\\\n"
        )
        f.write("\\hline\n")
        for gamma in gammas:
            r = results[str(gamma)]
            f.write(
                f"{gamma} & {r['ratio_mean']:.4f} $\\pm$ {r['ratio_se']:.4f} & "
                f"{gamma:.2f} & {r['iters_to_tol_mean']:.1f} & {r['iters_pred']:.0f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Banach fixed-point contraction convergence"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("BANACH FIXED-POINT THEOREM: GEOMETRIC CONVERGENCE")
    print("=" * 70)
    print()
    print("Operator: policy-evaluation Bellman operator T V = r + gamma P V,")
    print("  a gamma-contraction in the sup norm (P row-stochastic).")
    print("  Fixed point V* = (I - gamma P)^{-1} r.")
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
