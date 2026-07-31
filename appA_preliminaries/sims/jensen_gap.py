# Maximization Bias: the Jensen gap E[max_a Qhat_a] - max_a E[Qhat_a] >= 0
# Appendix A - Mathematical Preliminaries
# How big is the overestimation a bootstrapped max introduces? The maximum is convex, so
# Jensen's inequality forces the expected max of noisy action values above the max of their
# means. This measures that gap against the number of actions and the noise level, and
# checks it against the closed form for the equal-means case, where the gap is exactly
# sigma * E[max of n standard normals].

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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "jensen_gap"
OUTPUT_DIR = os.path.dirname(__file__)

CONFIG = {
    "n_actions": [2, 3, 4, 6, 8, 12, 16, 24, 32],
    "sigmas": [0.1, 0.25, 0.5, 1.0],
    "n_draws": 400000,  # Monte Carlo replicates per (n_actions, sigma) cell
    "seed": 20260725,
    # a spread of true action values, to show the gap shrinks once one action is clearly best
    "gap_between_means": [0.0, 0.25, 1.0],
    "version": 2,
}

SIGMA_COLORS = {
    0.1: COLORS["blue"],
    0.25: COLORS["orange"],
    0.5: COLORS["green"],
    1.0: COLORS["red"],
}
NACT_COLORS = {
    2: COLORS["blue"],
    4: COLORS["orange"],
    8: COLORS["green"],
    32: COLORS["red"],
}


def expected_max_standard_normal(n, n_draws, rng):
    """Monte Carlo E[max of n independent standard normals]."""
    return float(np.max(rng.standard_normal((n_draws, n)), axis=1).mean())


def _bias_grid():
    """The gap for equal true values, where the whole of E[max] is bias.

    With Q(a) equal across actions and estimates Qhat_a = Q(a) + sigma * Z_a, the max of
    the means is Q, so the gap is exactly sigma * E[max_a Z_a]. That closed form is the
    reference the Monte Carlo estimate is checked against.
    """
    rng = np.random.default_rng(CONFIG["seed"])
    rows = []
    print("Equal true action values: the entire expected maximum is overestimation.")
    print(f"  Monte Carlo over {CONFIG['n_draws']} replicates per cell.")
    print(
        f"  {'actions':>8s}  {'sigma':>6s}  {'measured gap':>13s}  "
        f"{'sigma*E[max Z]':>15s}  {'rel. error':>11s}"
    )
    emax_cache = {}
    for n in CONFIG["n_actions"]:
        emax_cache[n] = expected_max_standard_normal(n, CONFIG["n_draws"], rng)
        for sigma in CONFIG["sigmas"]:
            draws = sigma * rng.standard_normal((CONFIG["n_draws"], n))
            measured = float(np.max(draws, axis=1).mean())  # max_a E[Qhat_a] = 0 here
            reference = sigma * emax_cache[n]
            rel = abs(measured - reference) / max(reference, 1e-12)
            rows.append(
                {
                    "n_actions": n,
                    "sigma": sigma,
                    "measured": measured,
                    "reference": reference,
                    "rel_error": rel,
                }
            )
            if n in (2, 8, 32):
                print(
                    f"  {n:8d}  {sigma:6.2f}  {measured:13.4f}  {reference:15.4f}  {rel:11.2e}"
                )
    print()
    print("  E[max of n standard normals], the shape factor:")
    for n in CONFIG["n_actions"]:
        print(f"    n = {n:3d}: {emax_cache[n]:.4f}")
    print("  It grows like sqrt(2 log n), so doubling the actions adds less each time.")
    print()
    # The gap must be nonnegative, which is the inequality itself.
    assert all(r["measured"] > 0 for r in rows), (
        "a measured Jensen gap came out negative"
    )
    return {"rows": rows, "emax": emax_cache}


def _separation():
    """The gap once one action is genuinely better, which is the case that matters.

    Jensen still forces a nonnegative gap, but a clear winner makes the max nearly
    deterministic and the bias collapses. This is why maximization bias hurts most early
    in learning, when the action values are not yet separated.
    """
    rng = np.random.default_rng(CONFIG["seed"] + 1)
    rows = []
    print("Separated true action values: one action better than the rest by 'gap'.")
    print(f"  {'actions':>8s}  {'sigma':>6s}  {'true gap':>9s}  {'bias':>9s}")
    for n in [2, 8, 32]:
        for sigma in [0.25, 1.0]:
            for sep in CONFIG["gap_between_means"]:
                means = np.zeros(n)
                means[0] = sep
                draws = means + sigma * rng.standard_normal((CONFIG["n_draws"], n))
                measured = float(np.max(draws, axis=1).mean())
                bias = measured - float(means.max())
                rows.append(
                    {"n_actions": n, "sigma": sigma, "separation": sep, "bias": bias}
                )
                print(f"  {n:8d}  {sigma:6.2f}  {sep:9.2f}  {bias:9.4f}")
    print()
    print("  The bias never turns negative, as Jensen requires, and it falls as the")
    print("  best action pulls clear of the field relative to the noise.")
    assert all(r["bias"] > -1e-9 for r in rows), "a measured bias came out negative"
    print()
    return {"rows": rows}


def compute_data(force=None):
    force = force or set()
    grid = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "bias_grid",
        CONFIG,
        _bias_grid,
        force=("bias_grid" in force),
    )
    sep = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "separation",
        CONFIG,
        _separation,
        force=("separation" in force),
    )
    return {"grid": grid, "separation": sep}


def generate_outputs(data):
    rows = data["grid"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax = axes[0]
    for sigma in CONFIG["sigmas"]:
        sel = [r for r in rows if r["sigma"] == sigma]
        ns = [r["n_actions"] for r in sel]
        gaps = [r["measured"] for r in sel]
        ax.plot(
            ns,
            gaps,
            "o-",
            ms=3.5,
            color=SIGMA_COLORS[sigma],
            lw=1.5,
            label=rf"$\sigma = {sigma}$",
        )
        ax.plot(
            ns,
            [r["reference"] for r in sel],
            "--",
            lw=1.0,
            color=SIGMA_COLORS[sigma],
            alpha=0.7,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of actions")
    ax.set_ylabel(
        r"$\mathbb{E}[\max_a \widehat{Q}_a] - \max_a \mathbb{E}[\widehat{Q}_a]$"
    )
    ax.set_title("gap grows with the number of actions")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    sigmas_fine = CONFIG["sigmas"]
    for n in [2, 4, 8, 32]:
        sel = [r for r in rows if r["n_actions"] == n]
        ax.plot(
            [r["sigma"] for r in sel],
            [r["measured"] for r in sel],
            "o-",
            ms=3.5,
            color=NACT_COLORS[n],
            lw=1.5,
            label=f"{n} actions",
        )
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel("gap")
    ax.set_title("gap is linear in the noise level")
    ax.legend(loc="upper left", fontsize=7)
    del sigmas_fine

    fig_path = os.path.join(OUTPUT_DIR, "jensen_gap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "jensen_gap.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Maximization bias for independent Gaussian action-value estimates, "
            f"over {CONFIG['n_draws']} Monte Carlo replicates per cell. With equal true "
            "action values the whole expected maximum is bias, and it equals "
            "$\\sigma\\,\\mathbb{E}[\\max_a Z_a]$ in closed form. The last column gives the "
            "bias once the best action is separated from the rest by one unit, at which "
            "point the maximum is nearly deterministic and the bias collapses.}\n"
        )
        f.write("\\label{tab:prelim_jensen}\n")
        f.write("\\begin{tabular}{rrrrr}\n\\hline\n")
        f.write(
            "actions & $\\sigma$ & measured gap & $\\sigma\\,\\mathbb{E}[\\max_a Z_a]$ "
            "& bias at separation $1$ \\\\\n"
        )
        f.write("\\hline\n")
        sep_rows = {
            (r["n_actions"], r["sigma"], r["separation"]): r["bias"]
            for r in data["separation"]["rows"]
        }
        for n in [2, 8, 32]:
            for sigma in [0.25, 1.0]:
                r = next(x for x in rows if x["n_actions"] == n and x["sigma"] == sigma)
                sepbias = sep_rows[(n, sigma, 1.0)]
                f.write(
                    f"{n} & {sigma:.2f} & {r['measured']:.4f} & {r['reference']:.4f} "
                    f"& {sepbias:.4f} \\\\\n"
                )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Maximization bias as a Jensen gap")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("MAXIMIZATION BIAS AS A JENSEN GAP")
    print("=" * 70)
    print()
    if args.plots_only:
        generate_outputs(compute_data())
    elif args.data_only:
        compute_data(force=force)
    else:
        generate_outputs(compute_data(force=force))
    print("\nDone.")


if __name__ == "__main__":
    main()
