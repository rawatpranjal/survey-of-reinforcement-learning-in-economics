# Convexity and Gradient Descent
# Appendix A - Mathematical Preliminaries
# Gradient descent with step 1/L on an L-smooth convex objective. On a strongly
# convex quadratic the iterate error contracts at the exact rate 1 - mu/L per step
# (a linear rate set by the condition number kappa = L/mu). The general smooth-convex
# guarantee f(x_k) - f* <= L||x0 - x*||^2 / (2k) is verified as an upper bound that
# never breaks. f* = 0 and x* = 0 by construction (b = 0).

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "gradient_descent"
CONFIG = {
    "L": 1.0,  # smoothness constant; step = 1/L
    "kappas": [10, 100, 1000],  # condition numbers kappa = L/mu
    "dim": 50,
    "n_iters": 20000,
    "n_seeds": 20,  # random unit initial points
    "tail_frac": 0.2,  # measure the asymptotic per-step factor over the last 20%
    "tol": 1e-8,
    "seed_base": 44000,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)

KAPPA_COLORS = {10: COLORS["blue"], 100: COLORS["orange"], 1000: COLORS["red"]}


# ---------------------------------------------------------------------------
# Objective: f(x) = 1/2 x^T Q x with Q = diag(lambda), eigenvalues log-spaced in
# [mu, L]. mu = L/kappa. f* = 0 at x* = 0. GD step 1/L acts elementwise:
# x_i <- x_i (1 - lambda_i / L).
# ---------------------------------------------------------------------------


def _run_experiment():
    L = CONFIG["L"]
    kappas = CONFIG["kappas"]
    dim = CONFIG["dim"]
    n_iters = CONFIG["n_iters"]
    n_seeds = CONFIG["n_seeds"]
    tol = CONFIG["tol"]

    print("Gradient descent, step 1/L, on strongly convex quadratics f = 1/2 x^T Q x")
    print(f"  L={L}, kappas={kappas}, dim={dim}, iters={n_iters}, seeds={n_seeds}")
    print()

    results = {}
    for kappa in kappas:
        mu = L / kappa
        # Eigenvalues span [mu, L] with the slowest mode mu isolated below a cluster
        # in [sqrt(mu*L), L]. The isolation makes the slowest mode dominate the tail
        # quickly, so the measured asymptotic contraction is the exact rate 1 - mu/L
        # rather than a blend of neighboring modes.
        lam = np.empty(dim)
        lam[0] = mu
        lam[1:] = np.logspace(np.log10(np.sqrt(mu * L)), np.log10(L), dim - 1)
        contract = 1.0 - lam / L  # per-mode error contraction of GD with step 1/L

        # Diagonal Q makes the GD trajectory closed form: x_k[i] = x0[i] contract_i^k.
        # Build the per-mode power matrix once (same across seeds), then read every
        # curve off matrix-vector products instead of a 20000-step Python loop.
        karr = np.arange(n_iters + 1)
        c2 = contract**2
        C2K = np.power(c2[:, None], karr[None, :])  # dim x (n_iters+1): contract_i^{2k}

        f_curve = np.zeros((n_seeds, n_iters + 1))
        factors = np.zeros(n_seeds)  # measured asymptotic per-step iterate factor
        bound_ratio = np.zeros(n_seeds)  # sup_k f_k * 2k / (L ||x0||^2), must be <= 1
        iters_to_tol = np.full(n_seeds, np.nan)
        kk = np.arange(1, n_iters + 1)
        for si in range(n_seeds):
            rng = np.random.RandomState(CONFIG["seed_base"] + si)
            x0 = rng.normal(size=dim)
            x0 = x0 / np.linalg.norm(x0)  # unit initial error, so ||x0 - x*|| = 1
            # f_k = 1/2 sum_i lam_i x0_i^2 contract_i^{2k}; err_k^2 = sum_i x0_i^2 contract_i^{2k}
            fs = 0.5 * (lam * x0**2) @ C2K
            errsq = (x0**2) @ C2K
            errs = np.sqrt(np.maximum(errsq, 0.0))
            f_curve[si] = fs
            # asymptotic per-step factor of the iterate error, measured in a fixed
            # value window (post-transient, pre-underflow) that adapts to each kappa's
            # convergence speed. A fixed iteration tail fails: a well-conditioned run
            # underflows to zero long before iteration 20000.
            mask = (errs > 1e-11) & (errs < 1e-4)
            idx = np.where(mask)[0]
            if len(idx) >= 2:
                a, b = idx[0], idx[-1]
                ratio = (errs[b] / errs[a]) ** (1.0 / (b - a))
            else:
                ratio = np.nan
            factors[si] = ratio
            # Note: because the trajectory is the closed-form geometric sequence
            # x0*contract^k, this "factor" is the analytic base 1 - lam[0]/L read
            # back off the tail, not an independent empirical measurement.
            # O(1/k) upper-bound ratio (k >= 1); bound is f_k <= L ||x0||^2 / (2k)
            bound = L * errs[0] ** 2 / (2.0 * kk)
            bound_ratio[si] = float(np.max(fs[1:] / bound))
            below = np.where(fs < tol)[0]
            if len(below) > 0:
                iters_to_tol[si] = below[0]

        results[str(kappa)] = {
            "mu_over_L": mu / L,
            "theory_factor": 1.0 - mu / L,
            "factor_mean": float(np.nanmean(factors)),
            "factor_se": float(
                np.nanstd(factors) / np.sqrt(np.sum(~np.isnan(factors)))
            ),
            "bound_ratio_max": float(bound_ratio.max()),
            "iters_to_tol": float(np.nanmean(iters_to_tol)),
            "f_curve_mean": f_curve.mean(axis=0),
            "f0_mean": float(f_curve[:, 0].mean()),
        }
        r = results[str(kappa)]
        print(
            f"  kappa={kappa:5d}: mu/L={mu / L:.4f}, asymptotic per-step factor="
            f"{r['factor_mean']:.5f} +/- {r['factor_se']:.1e} (theory {1 - mu / L:.5f}); "
            f"O(1/k) bound ratio={r['bound_ratio_max']:.3f} (<=1); "
            f"iters to {tol:g}={r['iters_to_tol']:.0f}"
        )

    return {"results": results, "config": CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "gd",
        CONFIG,
        _run_experiment,
        force=("gd" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    results = data["results"]
    config = data["config"]
    kappas = config["kappas"]
    L = config["L"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: linear convergence, one curve per condition number ---------
    axA = axes[0]
    for kappa in kappas:
        r = results[str(kappa)]
        f = r["f_curve_mean"]
        k = np.arange(len(f))
        color = KAPPA_COLORS[kappa]
        axA.semilogy(
            k,
            np.maximum(f, 1e-16),
            color=color,
            linewidth=1.6,
            label=rf"$\kappa = {kappa}$",
        )
        # theory envelope: worst mode contracts error by (1 - mu/L), f-gap by its square
        env = r["f0_mean"] * (r["theory_factor"] ** (2 * k))
        axA.semilogy(
            k,
            np.maximum(env, 1e-16),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )
    axA.set_xlabel("Iteration $k$")
    axA.set_ylabel(r"$f(x_k) - f^\star$")
    axA.set_ylim(1e-16, None)
    axA.set_title("Strongly convex: linear rate $(1-\\mu/L)^{2k}$")
    axA.legend(loc="upper right", title="solid: measured, dashed: theory")

    # --- Panel B: the O(1/k) upper bound for the worst-conditioned case ------
    axB = axes[1]
    kappa = max(kappas)
    r = results[str(kappa)]
    f = r["f_curve_mean"]
    k = np.arange(1, len(f))
    axB.loglog(
        k,
        np.maximum(f[1:], 1e-16),
        color=KAPPA_COLORS[kappa],
        linewidth=1.6,
        label=rf"$f(x_k)-f^\star,\ \kappa={kappa}$",
    )
    bound = L * 1.0 / (2.0 * k)  # ||x0||^2 = 1
    axB.loglog(
        k,
        bound,
        color=COLORS["black"],
        linewidth=1.2,
        linestyle="--",
        label=r"$L\|x_0-x^\star\|^2/(2k)$",
    )
    axB.set_xlabel("Iteration $k$")
    axB.set_ylabel(r"$f(x_k) - f^\star$")
    axB.set_title("General convex bound holds")
    axB.legend(loc="lower left")

    fig_path = os.path.join(OUTPUT_DIR, "gradient_descent.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "gradient_descent.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Gradient descent with step $1/L$ on strongly convex quadratics "
            "at three condition numbers $\\kappa = L/\\mu$. The asymptotic per-step "
            "contraction of the iterate error, read off the tail of the closed-form "
            "trajectory, equals the linear rate $1 - \\mu/L$ by construction, so "
            "the iteration count to a fixed tolerance grows with $\\kappa$. The "
            "general smooth-convex bound $f(x_k) - f^\\star \\leq L\\|x_0 - x^\\star\\|^2/(2k)$ "
            "holds throughout (ratio $\\leq 1$). Mean $\\pm$ SE over "
            + str(config["n_seeds"])
            + " random initial points.}\n"
        )
        f.write("\\label{tab:prelim_gd}\n")
        f.write("\\begin{tabular}{ccccc}\n\\hline\n")
        f.write(
            "$\\kappa$ & $\\mu/L$ & Per-step factor & Theory $1-\\mu/L$ & "
            "$O(1/k)$ bound ratio \\\\\n\\hline\n"
        )
        for kappa in kappas:
            r = results[str(kappa)]
            f.write(
                f"{kappa} & {r['mu_over_L']:.4f} & "
                f"{r['factor_mean']:.5f} $\\pm$ {r['factor_se']:.0e} & "
                f"{r['theory_factor']:.5f} & {r['bound_ratio_max']:.3f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Gradient descent convergence rates")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("CONVEXITY AND GRADIENT DESCENT")
    print("=" * 70)
    print()
    print("GD with step 1/L on f(x) = 1/2 x^T Q x, Q = diag in [mu, L], x* = 0.")
    print("Strongly convex => linear rate (1 - mu/L) per step on the iterate error.")
    print("General smooth-convex => f(x_k) - f* <= L||x0 - x*||^2 / (2k).")
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
