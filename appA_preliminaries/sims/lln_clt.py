# Law of Large Numbers and Central Limit Theorem
# Appendix A - Mathematical Preliminaries
# The sample mean converges to the population mean (LLN), and the rescaled
# fluctuation sqrt(n)(Xbar_n - mu) converges in distribution to N(0, sigma^2)
# (CLT), regardless of the base distribution. Both are checked against theory:
# LLN by the shrinking |Xbar_n - mu|, CLT by the variance of the rescaled mean
# matching sigma^2 and its shape approaching normal (KS distance falling as 1/sqrt(n)).

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE, BENCH_STYLE

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "lln_clt"
CONFIG = {
    # base distributions: (name, mu, sigma^2). None is normal.
    "distributions": ["exponential", "uniform", "bernoulli"],
    "bernoulli_p": 0.3,
    # LLN: running-mean trajectories for the primary distribution
    "primary": "exponential",
    "n_traj": 25,  # trajectories drawn (shown thinned)
    "lln_max_n": 100000,
    # sqrt(n)-rate grid for the LLN deviation and the CLT KS distance
    "n_grid": [10, 30, 100, 300, 1000, 3000, 10000, 30000],
    "n_dev_seeds": 2000,  # replicates for estimating E|Xbar_n - mu| and KS(n)
    # CLT: histogram of sqrt(n)(Xbar_n - mu) at a fixed large n
    "n_clt": 1000,
    "n_replicates": 40000,
    "seed_base": 20250,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)

DIST_COLORS = {
    "exponential": COLORS["blue"],
    "uniform": COLORS["orange"],
    "bernoulli": COLORS["green"],
}
DIST_LABEL = {
    "exponential": r"Exp$(1)$",
    "uniform": r"Unif$(0,1)$",
    "bernoulli": r"Bern$(0.3)$",
}

# ---------------------------------------------------------------------------
# Base distributions: sampler + analytical mean and variance.
# CLT holds for any of these even though none is normal.
# ---------------------------------------------------------------------------


def dist_params(name, p_bern):
    if name == "exponential":
        return 1.0, 1.0  # Exp(1): mu = 1, var = 1
    if name == "uniform":
        return 0.5, 1.0 / 12.0  # Unif(0,1): mu = 1/2, var = 1/12
    if name == "bernoulli":
        return p_bern, p_bern * (1.0 - p_bern)  # Bern(p): mu = p, var = p(1-p)
    raise ValueError(name)


def sample(name, rng, size, p_bern):
    if name == "exponential":
        return rng.exponential(1.0, size=size)
    if name == "uniform":
        return rng.uniform(0.0, 1.0, size=size)
    if name == "bernoulli":
        return (rng.uniform(0.0, 1.0, size=size) < p_bern).astype(float)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    dists = CONFIG["distributions"]
    p_bern = CONFIG["bernoulli_p"]
    primary = CONFIG["primary"]
    n_grid = CONFIG["n_grid"]
    n_dev_seeds = CONFIG["n_dev_seeds"]
    n_clt = CONFIG["n_clt"]
    n_rep = CONFIG["n_replicates"]
    seed_base = CONFIG["seed_base"]

    print("LLN + CLT: sample mean of non-normal draws")
    print(f"  distributions: {dists}")
    print(f"  CLT histogram at n={n_clt} over {n_rep} replicates")
    print(f"  rate grid n_grid={n_grid}, {n_dev_seeds} replicates per n")
    print()

    # --- LLN trajectories for the primary distribution -----------------------
    mu_p, var_p = dist_params(primary, p_bern)
    sd_p = np.sqrt(var_p)
    max_n = CONFIG["lln_max_n"]
    rng = np.random.RandomState(seed_base)
    traj = np.zeros((CONFIG["n_traj"], max_n))
    for i in range(CONFIG["n_traj"]):
        x = sample(primary, rng, max_n, p_bern)
        traj[i] = np.cumsum(x) / np.arange(1, max_n + 1)  # running mean
    # thin the columns to a log grid for storage/plotting
    show_idx = np.unique(np.round(np.logspace(0, np.log10(max_n), 200)).astype(int))
    show_idx = show_idx[show_idx <= max_n] - 1
    lln = {
        "mu": mu_p,
        "sd": sd_p,
        "n_show": (show_idx + 1),
        "traj_show": traj[:, show_idx],
    }
    print(
        f"  LLN ({primary}): mu={mu_p:.4f}, final running mean range over "
        f"{CONFIG['n_traj']} seeds at n={max_n}: "
        f"[{traj[:, -1].min():.4f}, {traj[:, -1].max():.4f}]"
    )

    # --- sqrt(n)-rate of the mean absolute deviation E|Xbar_n - mu| ----------
    # LLN says this -> 0; CLT predicts it ~ sigma * sqrt(2/pi) / sqrt(n).
    dev_mean = np.zeros(len(n_grid))
    ks_by_n = np.zeros(len(n_grid))
    rng = np.random.RandomState(seed_base + 1)
    for j, n in enumerate(n_grid):
        xbar = sample(primary, rng, (n_dev_seeds, n), p_bern).mean(axis=1)
        dev_mean[j] = np.mean(np.abs(xbar - mu_p))
        s = np.sqrt(n) * (xbar - mu_p)  # rescaled fluctuation, target N(0, var_p)
        ks_by_n[j] = stats.kstest(s, stats.norm(loc=0.0, scale=sd_p).cdf).statistic
    dev_pred = sd_p * np.sqrt(2.0 / np.pi) / np.sqrt(np.array(n_grid))
    print(
        f"  LLN deviation E|Xbar-mu| at n={n_grid[0]}: {dev_mean[0]:.4f} "
        f"(pred {dev_pred[0]:.4f}); at n={n_grid[-1]}: {dev_mean[-1]:.5f} "
        f"(pred {dev_pred[-1]:.5f})"
    )

    # --- CLT: distribution of sqrt(n)(Xbar_n - mu) at fixed n, per dist -------
    clt = {}
    for di, name in enumerate(dists):
        mu, var = dist_params(name, p_bern)
        sd = np.sqrt(var)
        # deterministic per-distribution seed (not hash(), which is process-randomized)
        rng = np.random.RandomState(seed_base + 100 + 17 * di)
        xbar = sample(name, rng, (n_rep, n_clt), p_bern).mean(axis=1)
        s = np.sqrt(n_clt) * (xbar - mu)  # ~ N(0, var)
        emp_var = float(np.var(s))
        ks = float(stats.kstest(s, stats.norm(loc=0.0, scale=sd).cdf).statistic)
        clt[name] = {
            "mu": mu,
            "var": var,
            "emp_var": emp_var,
            "var_ratio": emp_var / var,
            "skew": float(stats.skew(s)),
            "exkurt": float(stats.kurtosis(s)),  # excess kurtosis (0 for normal)
            "ks": ks,
            "samples": s
            if name == CONFIG["primary"]
            else None,  # store hist data for primary only
        }
        print(
            f"  CLT {name:11s}: mu={mu:.4f}, sigma^2={var:.4f}, "
            f"emp var(sqrt(n)(Xbar-mu))={emp_var:.4f} (ratio {emp_var / var:.4f}), "
            f"skew={clt[name]['skew']:+.4f}, exkurt={clt[name]['exkurt']:+.4f}, KS={ks:.4f}"
        )

    return {
        "config": CONFIG,
        "lln": lln,
        "n_grid": list(n_grid),
        "dev_mean": dev_mean,
        "dev_pred": dev_pred,
        "ks_by_n": ks_by_n,
        "clt": clt,
    }


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "lln_clt",
        CONFIG,
        _run_experiment,
        force=("lln_clt" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    config = data["config"]
    lln = data["lln"]
    clt = data["clt"]
    primary = config["primary"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: LLN running-mean trajectories converge to mu ---------------
    axA = axes[0]
    n_show = lln["n_show"]
    for i in range(lln["traj_show"].shape[0]):
        axA.semilogx(
            n_show, lln["traj_show"][i], color=COLORS["blue"], linewidth=0.6, alpha=0.35
        )
    axA.axhline(lln["mu"], **BENCH_STYLE, label=r"population mean $\mu$")
    axA.set_xlabel("Sample size $n$")
    axA.set_ylabel(r"running mean $\bar X_n$")
    axA.set_title(f"LLN: {DIST_LABEL[primary]} sample mean $\\to \\mu$")
    axA.legend(loc="upper right")

    # --- Panel B: CLT histogram of sqrt(n)(Xbar - mu) vs N(0, sigma^2) --------
    axB = axes[1]
    s = clt[primary]["samples"]
    sd = np.sqrt(clt[primary]["var"])
    axB.hist(
        s,
        bins=60,
        density=True,
        color=COLORS["blue"],
        alpha=0.55,
        label=r"$\sqrt{n}(\bar X_n - \mu)$",
    )
    grid = np.linspace(s.min(), s.max(), 400)
    axB.plot(
        grid,
        stats.norm(loc=0.0, scale=sd).pdf(grid),
        color=COLORS["red"],
        linewidth=1.8,
        linestyle="--",
        label=rf"$N(0,\sigma^2),\ \sigma^2={clt[primary]['var']:.2f}$",
    )
    axB.set_xlabel(r"$\sqrt{n}(\bar X_n - \mu)$")
    axB.set_ylabel("density")
    axB.set_title(f"CLT: {DIST_LABEL[primary]} at $n={config['n_clt']}$")
    axB.legend(loc="upper right")

    fig_path = os.path.join(OUTPUT_DIR, "lln_clt.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table: per distribution, CLT variance vs sigma^2 + shape ------
    tex_path = os.path.join(OUTPUT_DIR, "lln_clt.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Central limit theorem for the rescaled sample mean "
            "$\\sqrt{n}(\\bar X_n - \\mu)$ at $n="
            + str(config["n_clt"])
            + "$, over "
            + f"{config['n_replicates']:,}".replace(",", "{,}")
            + " replicates, for three non-normal base distributions. The empirical "
            "variance matches $\\sigma^2$ (ratio near one), and the skewness and excess "
            "kurtosis are near the normal values of zero, so the rescaled mean is "
            "approximately $N(0,\\sigma^2)$ despite the skewed or discrete source. "
            "$D_{\\mathrm{KS}}$ is the Kolmogorov-Smirnov distance to $N(0,\\sigma^2)$.}\n"
        )
        f.write("\\label{tab:prelim_lln_clt}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write(
            "Distribution & $\\sigma^2$ & Emp.\\ var / $\\sigma^2$ & Skew & "
            "Excess kurt.\\ & $D_{\\mathrm{KS}}$ \\\\\n"
        )
        f.write("\\hline\n")
        tex_name = {
            "exponential": "Exp$(1)$",
            "uniform": "Unif$(0,1)$",
            "bernoulli": "Bern$(0.3)$",
        }
        for name in config["distributions"]:
            c = clt[name]
            f.write(
                f"{tex_name[name]} & {c['var']:.4f} & {c['var_ratio']:.4f} & "
                f"{c['skew']:+.4f} & {c['exkurt']:+.4f} & {c['ks']:.4f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LLN and CLT for the sample mean")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("LAW OF LARGE NUMBERS AND CENTRAL LIMIT THEOREM")
    print("=" * 70)
    print()
    print("LLN: the sample mean of i.i.d. draws converges to the population mean.")
    print("CLT: sqrt(n)(Xbar_n - mu) converges in distribution to N(0, sigma^2),")
    print("  whatever the (finite-variance) base distribution.")
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
