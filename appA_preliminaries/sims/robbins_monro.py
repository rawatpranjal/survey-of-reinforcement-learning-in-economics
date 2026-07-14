# Robbins-Monro Stochastic Approximation: role of the two step-size conditions
# Appendix A - Mathematical Preliminaries
# A noisy fixed-point iteration x_{t+1} = x_t + alpha_t((gamma-1)x_t + b + noise) is run
# under four step-size schedules. It converges to the root x* = b/(1-gamma) only when the
# schedule satisfies BOTH Robbins-Monro conditions: sum alpha = inf and sum alpha^2 < inf.

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
SCRIPT_NAME = "robbins_monro"
CONFIG = {
    "gamma": 0.5,  # mean-field contraction toward x* = b/(1-gamma)
    "b": 1.0,  # so x* = 2.0
    "sigma": 1.0,  # observation noise std
    "x0": 0.0,
    "n_steps": 5000,
    "n_seeds": 100,
    # (label, exponent p for alpha_t = 1/(t+1)^p, or constant if p is None)
    "schedules": [
        ("$1/t$", 1.0, None),
        ("$1/t^{0.6}$", 0.6, None),
        ("constant $0.1$", None, 0.1),
        ("$1/t^{2}$", 2.0, None),
    ],
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)

SCHED_COLORS = {
    "$1/t$": COLORS["blue"],
    "$1/t^{0.6}$": COLORS["green"],
    "constant $0.1$": COLORS["orange"],
    "$1/t^{2}$": COLORS["red"],
}


def step_sizes(p, const, n_steps):
    t = np.arange(n_steps)
    if const is not None:
        return np.full(n_steps, const)
    return 1.0 / (t + 1.0) ** p


def run_sa(gamma, b, sigma, x0, alphas, seed):
    """Return |x_t - x*| for the noisy fixed-point recursion."""
    rng = np.random.RandomState(seed)
    x_star = b / (1.0 - gamma)
    x = x0
    err = np.zeros(len(alphas))
    for t, a in enumerate(alphas):
        err[t] = abs(x - x_star)
        noise = rng.normal(0.0, sigma)
        # x_{t+1} = x_t + a * ( (gamma-1) x_t + b + noise ); mean field contracts to x*
        x = x + a * ((gamma - 1.0) * x + b + noise)
    return err


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    gamma = CONFIG["gamma"]
    b = CONFIG["b"]
    sigma = CONFIG["sigma"]
    x0 = CONFIG["x0"]
    n_steps = CONFIG["n_steps"]
    n_seeds = CONFIG["n_seeds"]
    schedules = CONFIG["schedules"]
    x_star = b / (1.0 - gamma)

    print(f"Robbins-Monro SA: noisy fixed point of x = gamma x + b, x* = {x_star}")
    print(f"  gamma={gamma}, b={b}, sigma={sigma}, steps={n_steps}, seeds={n_seeds}")
    print()

    results = {}
    for label, p, const in schedules:
        alphas = step_sizes(p, const, n_steps)
        errs = np.zeros((n_seeds, n_steps))
        for si in range(n_seeds):
            errs[si] = run_sa(gamma, b, sigma, x0, alphas, seed=7 + si)
        mse = np.mean(errs**2, axis=0)
        mse_se = np.std(errs**2, axis=0) / np.sqrt(n_seeds)
        rmse = np.sqrt(mse)  # RMS error across seeds
        # the two Robbins-Monro conditions for this schedule are analytic properties
        # of the schedule exponent p, not read off a finite partial sum
        # a schedule with p <= 1 has sum alpha -> inf; p > 0.5 has sum alpha^2 < inf
        if const is not None:
            cond1 = True  # constant: sum alpha = inf
            cond2 = False  # sum alpha^2 = inf
        else:
            cond1 = p <= 1.0  # sum 1/t^p diverges iff p <= 1
            cond2 = p > 0.5  # sum 1/t^{2p} converges iff 2p > 1
        results[label] = {
            "rmse": rmse,
            "mse": mse,
            "mse_se": mse_se,
            "final_rmse": float(rmse[-1]),
            "sum_alpha_diverges": bool(cond1),
            "sum_alpha2_converges": bool(cond2),
            "converges": bool(cond1 and cond2),
        }
        print(
            f"  {label:18s}: final RMS error = {rmse[-1]:.4f}  "
            f"[sum a = inf: {cond1}, sum a^2 < inf: {cond2}]"
        )

    return {"results": results, "config": CONFIG, "x_star": x_star}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "rm", CONFIG, _run_experiment, force=("rm" in force)
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    results = data["results"]
    config = data["config"]
    schedules = config["schedules"]

    # --- Figure: RMS error vs step, log-log, with a +/-1 SE band on the MSE ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for label, _p, _c in schedules:
        rmse = results[label]["rmse"]
        mse = results[label]["mse"]
        mse_se = results[label]["mse_se"]
        t = np.arange(1, len(rmse) + 1)
        color = SCHED_COLORS[label]
        ax.loglog(t, rmse, color=color, linewidth=1.8, label=label)
        lower = np.sqrt(np.maximum(mse - mse_se, 0.0))
        upper = np.sqrt(mse + mse_se)
        ax.fill_between(t, lower, upper, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Step $t$")
    ax.set_ylabel(r"RMS error $|x_t - x^\star|$")
    ax.set_title("Robbins-Monro: convergence needs both step-size conditions")
    ax.legend(loc="lower left", title="step size $\\alpha_t$")

    fig_path = os.path.join(OUTPUT_DIR, "robbins_monro.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---
    tex_path = os.path.join(OUTPUT_DIR, "robbins_monro.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Robbins-Monro stochastic approximation for the root of "
            "$x = \\gamma x + b$ ($x^\\star = " + f"{data['x_star']:.0f}" + "$) with "
            "i.i.d. observation noise. A schedule converges to $x^\\star$ only when "
            "$\\sum_t \\alpha_t = \\infty$ and $\\sum_t \\alpha_t^2 < \\infty$ both hold. "
            "Final RMS error over "
            + str(config["n_seeds"])
            + " seeds after "
            + str(config["n_steps"])
            + " steps.}\n"
        )
        f.write("\\label{tab:prelim_robbins_monro}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write(
            "$\\alpha_t$ & $\\sum \\alpha_t = \\infty$ & $\\sum \\alpha_t^2 < \\infty$ "
            "& Final RMS error \\\\\n"
        )
        f.write("\\hline\n")
        for label, _p, _c in schedules:
            r = results[label]
            c1 = "yes" if r["sum_alpha_diverges"] else "no"
            c2 = "yes" if r["sum_alpha2_converges"] else "no"
            f.write(f"{label} & {c1} & {c2} & {r['final_rmse']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Robbins-Monro stochastic approximation"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("ROBBINS-MONRO STOCHASTIC APPROXIMATION")
    print("=" * 70)
    print()
    print("Recursion: x_{t+1} = x_t + alpha_t ((gamma-1) x_t + b + noise)")
    print("  Root x* = b/(1-gamma). Mean field contracts to x*.")
    print("  Converges to x* iff sum alpha_t = inf and sum alpha_t^2 < inf.")
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
