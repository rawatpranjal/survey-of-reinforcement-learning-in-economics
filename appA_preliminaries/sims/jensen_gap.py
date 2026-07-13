# Jensen's Inequality: the convexity gap E[phi(X)] - phi(E[X]) >= 0
# Appendix A - Mathematical Preliminaries
# Monte Carlo estimates the Jensen gap for convex phi and checks it against the
# closed-form value, and against zero, showing (i) the gap is nonnegative and
# (ii) the plug-in estimator converges to the analytical gap as the sample grows.

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
SCRIPT_NAME = "jensen_gap"
CONFIG = {
    "mu": 0.0,
    "sigmas": [0.5, 1.0, 1.5, 2.0],
    # sample sizes on a log grid; the plug-in gap converges to the analytical value
    "sample_sizes": [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
    "n_seeds": 40,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)

SIGMA_COLORS = {
    0.5: COLORS["blue"],
    1.0: COLORS["orange"],
    1.5: COLORS["green"],
    2.0: COLORS["red"],
}

# ---------------------------------------------------------------------------
# Convex test functions phi and their closed-form Jensen gap for X ~ N(mu, sigma^2).
#   square: E[X^2] - (E X)^2 = sigma^2
#   exp:    E[e^X] - e^{E X} = e^{mu + sigma^2/2} - e^{mu}
# Both phi are convex, so the gap is >= 0 (Jensen). A concave contrast (sqrt of a
# positive variable) is reported in stdout to show the inequality reverses.
# ---------------------------------------------------------------------------


def phi_square(x):
    return x**2


def phi_exp(x):
    return np.exp(x)


CONVEX_FUNCS = {
    "square": (phi_square, r"$\varphi(x)=x^2$"),
    "exp": (phi_exp, r"$\varphi(x)=e^x$"),
}


def analytical_gap(name, mu, sigma):
    if name == "square":
        return sigma**2
    if name == "exp":
        return np.exp(mu + 0.5 * sigma**2) - np.exp(mu)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    mu = CONFIG["mu"]
    sigmas = CONFIG["sigmas"]
    sample_sizes = CONFIG["sample_sizes"]
    n_seeds = CONFIG["n_seeds"]
    max_n = max(sample_sizes)

    print(
        "Jensen gap: plug-in estimator of E[phi(X)] - phi(E[X]) for X ~ N(mu, sigma^2)"
    )
    print(f"  mu: {mu}, sigmas: {sigmas}, seeds: {n_seeds}")
    print(f"  sample sizes: {sample_sizes}")
    print()

    results = {}  # results[func][str(sigma)] = {sizes, gap_mean, gap_se, analytical}
    for fname, (phi, _) in CONVEX_FUNCS.items():
        results[fname] = {}
        for sigma in sigmas:
            gaps = np.zeros((n_seeds, len(sample_sizes)))
            for si in range(n_seeds):
                rng = np.random.RandomState(1000 + si)
                # draw the largest sample once, then read nested prefixes so the
                # curve is a genuine within-seed refinement as N grows
                x_full = rng.normal(mu, sigma, size=max_n)
                for j, n in enumerate(sample_sizes):
                    x = x_full[:n]
                    # plug-in gap: mean of phi minus phi of the sample mean.
                    # The estimator never sees the true mu, only the sample.
                    gaps[si, j] = phi(x).mean() - phi(x.mean())
            gap_mean = gaps.mean(axis=0)
            gap_se = gaps.std(axis=0) / np.sqrt(n_seeds)
            g_star = analytical_gap(fname, mu, sigma)
            # fraction of individual (seed, N) gap estimates that are nonnegative
            frac_nonneg = float(np.mean(gaps >= 0))
            results[fname][str(sigma)] = {
                "sizes": sample_sizes,
                "gap_mean": gap_mean,
                "gap_se": gap_se,
                "analytical": float(g_star),
                "gap_at_max": float(gap_mean[-1]),
                "se_at_max": float(gap_se[-1]),
                "frac_nonneg": frac_nonneg,
            }
            print(
                f"  {fname:6s} sigma={sigma:.1f}: analytical gap = {g_star:.4f}, "
                f"MC gap (N={max_n}) = {gap_mean[-1]:.4f} +/- {gap_se[-1]:.4f}, "
                f"P(gap>=0) = {frac_nonneg:.3f}"
            )

    # Concave contrast: phi(x) = sqrt(x), X ~ Uniform(0.5, 1.5); Jensen reverses,
    # so E[sqrt(X)] <= sqrt(E[X]) and the gap E[phi]-phi(E) is <= 0.
    rng = np.random.RandomState(7)
    xc = rng.uniform(0.5, 1.5, size=max_n)
    concave_gap = float(np.sqrt(xc).mean() - np.sqrt(xc.mean()))
    print()
    print(
        f"  concave contrast phi(x)=sqrt(x), X~U(0.5,1.5): gap = {concave_gap:.4f} (<= 0)"
    )

    return {"results": results, "config": CONFIG, "concave_gap": concave_gap}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "jensen",
        CONFIG,
        _run_experiment,
        force=("jensen" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    results = data["results"]
    config = data["config"]
    sigmas = config["sigmas"]

    # --- Figure: two panels (square, exp), plug-in gap vs N converging to the
    #     dashed analytical gap for each sigma ---
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    panels = [("square", CONVEX_FUNCS["square"][1]), ("exp", CONVEX_FUNCS["exp"][1])]
    for ax, (fname, flabel) in zip(axes, panels):
        for sigma in sigmas:
            r = results[fname][str(sigma)]
            sizes = np.array(r["sizes"])
            color = SIGMA_COLORS[sigma]
            ax.semilogx(
                sizes,
                r["gap_mean"],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=f"$\\sigma = {sigma}$",
            )
            ax.fill_between(
                sizes,
                r["gap_mean"] - r["gap_se"],
                r["gap_mean"] + r["gap_se"],
                color=color,
                alpha=0.2,
            )
            # analytical gap (dashed horizontal)
            ax.axhline(
                r["analytical"], color=color, linestyle="--", linewidth=1.0, alpha=0.7
            )
        ax.axhline(0.0, color=COLORS["black"], linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Sample size $N$")
        ax.set_title(f"Jensen gap, {flabel}")
        ax.legend(loc="best", title="solid: MC, dashed: exact")
    axes[0].set_ylabel(r"$\widehat{E[\varphi(X)]} - \varphi(\bar X)$")

    fig_path = os.path.join(OUTPUT_DIR, "jensen_gap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table: analytical vs MC gap at the largest N, both convex phi ---
    tex_path = os.path.join(OUTPUT_DIR, "jensen_gap.tex")
    max_n = max(config["sample_sizes"])
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Monte Carlo Jensen gap $E[\\varphi(X)]-\\varphi(E[X])$ for $X\\sim "
            "N(0,\\sigma^2)$, at the largest sample size $N="
            + f"{max_n:,}".replace(",", "{,}")
            + "$, against the closed-form value. The plug-in estimator converges to the exact gap; "
            "for $\\varphi(x)=x^2$ and the small-$\\sigma$ exponential cases the two agree within a "
            "standard error, while the heavy-tailed $\\sigma\\geq 1.5$ exponential rows converge more "
            "slowly and still sit a few standard errors low at this $N$. Every estimate is "
            "nonnegative, as convexity requires. Mean $\\pm$ SE over "
            + str(config["n_seeds"])
            + " seeds.}\n"
        )
        f.write("\\label{tab:prelim_jensen}\n")
        f.write("\\begin{tabular}{llccc}\n\\hline\n")
        f.write(
            "$\\varphi$ & $\\sigma$ & Analytical gap & MC gap & $\\Pr(\\text{gap} \\geq 0)$ \\\\\n"
        )
        f.write("\\hline\n")
        tex_name = {"square": "$x^2$", "exp": "$e^x$"}
        for fname in ["square", "exp"]:
            for sigma in sigmas:
                r = results[fname][str(sigma)]
                f.write(
                    f"{tex_name[fname]} & {sigma} & {r['analytical']:.4f} & "
                    f"{r['gap_at_max']:.4f} $\\pm$ {r['se_at_max']:.4f} & "
                    f"{r['frac_nonneg']:.3f} \\\\\n"
                )
            f.write("\\hline\n")
        f.write("\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Jensen inequality convexity gap")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("JENSEN'S INEQUALITY: THE CONVEXITY GAP E[phi(X)] - phi(E[X]) >= 0")
    print("=" * 70)
    print()
    print("Estimator: plug-in gap (1/N) sum phi(X_i) - phi((1/N) sum X_i),")
    print("  which never uses the true mean. For convex phi the gap is >= 0 and")
    print("  converges to the closed-form value as N grows.")
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
