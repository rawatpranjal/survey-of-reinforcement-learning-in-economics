# What a Discount Factor Near One Costs
# Appendix A - Mathematical Preliminaries
# Readers meet the effective horizon 1/(1-gamma) as a symbol and do not feel it until
# gamma = 0.99 needs a hundred times the work of gamma = 0.9. Three measurements on the
# same axis: iterations of value iteration to a fixed tolerance, the reward-to-value
# amplification of the policy-evaluation solve, and the terms of the Neumann series needed
# for the same accuracy. Merges what banach_contraction.py, neumann_series.py and the right
# panel of lipschitz_continuity.py each showed separately.

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
SCRIPT_NAME = "discount_cost"
OUTPUT_DIR = os.path.dirname(__file__)

CONFIG = {
    "gammas": [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999],
    "n_states": 50,
    "n_seeds": 30,
    "tolerance": 1e-3,
    "seed": 20260725,
    "reported_gammas": [0.5, 0.9, 0.99, 0.999],
    "version": 1,
}


def random_mrp(rng, n):
    """A random Markov reward process: row-stochastic P and a bounded reward vector."""
    P = rng.random((n, n))
    P = P / P.sum(axis=1, keepdims=True)
    r = rng.random(n)  # rewards in [0, 1], so ||r||_inf <= 1
    return P, r


def _measure():
    rng = np.random.default_rng(CONFIG["seed"])
    n, tol = CONFIG["n_states"], CONFIG["tolerance"]
    rows = []
    print(f"Random {n}-state Markov reward processes, {CONFIG['n_seeds']} seeds,")
    print(f"tolerance {tol} in the supremum norm, rewards drawn in [0, 1].")
    print()
    print(
        f"  {'gamma':>6s}  {'1/(1-g)':>9s}  {'VI iters':>9s}  {'se':>5s}  "
        f"{'Neumann M':>10s}  {'||(I-gP)^-1||':>14s}  {'||V||_inf':>10s}"
    )
    for gamma in CONFIG["gammas"]:
        vi_counts, neumann_counts, amps, vnorms = [], [], [], []
        for seed in range(CONFIG["n_seeds"]):
            P, r = random_mrp(np.random.default_rng(CONFIG["seed"] + seed), n)
            V_star = np.linalg.solve(np.eye(n) - gamma * P, r)

            # Value iteration on the policy-evaluation operator T V = r + gamma P V.
            V = np.zeros(n)
            k = 0
            while np.max(np.abs(V - V_star)) >= tol and k < 200000:
                V = r + gamma * P @ V
                k += 1
            vi_counts.append(k)

            # Neumann truncation to the same tolerance.
            partial = np.zeros(n)
            term = np.eye(n)
            M = -1
            while np.max(np.abs(partial - V_star)) >= tol and M < 200000:
                partial = partial + term @ r
                term = gamma * term @ P
                M += 1
            neumann_counts.append(M)

            # Reward-to-value amplification, the supremum operator norm of the resolvent.
            resolvent = np.linalg.inv(np.eye(n) - gamma * P)
            amps.append(float(np.abs(resolvent).sum(axis=1).max()))
            vnorms.append(float(np.max(np.abs(V_star))))

        row = {
            "gamma": gamma,
            "horizon": 1.0 / (1.0 - gamma),
            "vi_mean": float(np.mean(vi_counts)),
            "vi_se": float(np.std(vi_counts, ddof=1) / np.sqrt(CONFIG["n_seeds"])),
            "neumann_mean": float(np.mean(neumann_counts)),
            "amp_mean": float(np.mean(amps)),
            "v_norm": float(np.mean(vnorms)),
        }
        rows.append(row)
        print(
            f"  {gamma:6.3f}  {row['horizon']:9.1f}  {row['vi_mean']:9.1f}  "
            f"{row['vi_se']:5.2f}  {row['neumann_mean']:10.1f}  "
            f"{row['amp_mean']:14.2f}  {row['v_norm']:10.2f}"
        )
    print()

    # The resolvent's row sums are exactly 1/(1-gamma) for a row-stochastic P, so the
    # measured amplification must equal the effective horizon to solver precision.
    worst = max(abs(r["amp_mean"] - r["horizon"]) / r["horizon"] for r in rows)
    print(f"  amplification against 1/(1-gamma): worst relative deviation {worst:.2e}")
    assert worst < 1e-8, "the resolvent row sums do not equal the effective horizon"

    print(
        "  Value iteration from V_0 = 0 and Neumann truncation are the same computation"
    )
    print("  written two ways: after k backups the iterate is the partial sum through")
    print(
        "  m = k-1, so the two counters differ by exactly one at every discount factor."
    )
    offsets = {round(r["vi_mean"] - r["neumann_mean"], 9) for r in rows}
    print(
        f"  measured difference in counts, over all discount factors: {sorted(offsets)}"
    )
    assert offsets == {1.0}, "value iteration and Neumann truncation do not agree"
    print()
    print("  Iteration count against the effective horizon:")
    print(f"    {'gamma':>6s}  {'iters / (1/(1-g))':>18s}")
    for r in rows:
        print(f"    {r['gamma']:6.3f}  {r['vi_mean'] / r['horizon']:18.2f}")
    print("  The ratio drifts up rather than staying flat, because the count is")
    print("  log(||V*||/tol) / log(1/gamma) and the numerator grows with gamma too.")
    print()
    return {"rows": rows}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "measure", CONFIG, _measure, force=("measure" in force)
    )


def generate_outputs(data):
    rows = data["rows"]
    gammas = np.array([r["gamma"] for r in rows])
    horizon = np.array([r["horizon"] for r in rows])
    vi = np.array([r["vi_mean"] for r in rows])
    amp = np.array([r["amp_mean"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax = axes[0]
    ax.loglog(
        horizon,
        vi,
        "o-",
        color=COLORS["blue"],
        lw=1.6,
        ms=4,
        label="value-iteration steps",
    )
    ax.loglog(
        horizon, horizon, "--", color=COLORS["gray"], lw=1.2, label=r"$1/(1-\gamma)$"
    )
    for r in rows:
        if r["gamma"] in CONFIG["reported_gammas"]:
            ax.annotate(
                rf"$\gamma = {r['gamma']}$",
                (r["horizon"], r["vi_mean"]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7,
            )
    ax.set_xlabel(r"effective horizon $1/(1-\gamma)$")
    ax.set_ylabel(f"steps to sup-norm error below {CONFIG['tolerance']:g}")
    ax.set_title("the work grows with the effective horizon")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    ax.loglog(
        horizon,
        amp,
        "o",
        color=COLORS["red"],
        ms=5,
        label=r"$\|(I - \gamma P)^{-1}\|_\infty$",
    )
    ax.loglog(
        horizon, horizon, "--", color=COLORS["gray"], lw=1.2, label=r"$1/(1-\gamma)$"
    )
    ax.set_xlabel(r"effective horizon $1/(1-\gamma)$")
    ax.set_ylabel("reward-to-value amplification")
    ax.set_title("so does the error amplification")
    ax.legend(loc="upper left", fontsize=7)

    fig_path = os.path.join(OUTPUT_DIR, "discount_cost.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "discount_cost.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            f"\\caption{{What the discount factor costs, on random {CONFIG['n_states']}-state "
            f"Markov reward processes over {CONFIG['n_seeds']} seeds with rewards in $[0,1]$. "
            f"Steps are backups of $T^\\pi$ from $V_0 = 0$ to a supremum-norm error below "
            f"{CONFIG['tolerance']:g}. Amplification is $\\|(I - \\gamma P)^{{-1}}\\|_\\infty$, "
            "the factor by which an error in the reward is magnified in the value, which for a "
            "row-stochastic $P$ equals the effective horizon exactly.}\n"
        )
        f.write("\\label{tab:prelim_discount_cost}\n")
        f.write("\\begin{tabular}{rrrrr}\n\\hline\n")
        f.write(
            "$\\gamma$ & $1/(1-\\gamma)$ & steps & amplification & $\\|V^\\pi\\|_\\infty$ \\\\\n"
        )
        f.write("\\hline\n")
        for r in rows:
            f.write(
                f"{r['gamma']:.3f} & {r['horizon']:.1f} & {r['vi_mean']:.0f} & "
                f"{r['amp_mean']:.1f} & {r['v_norm']:.1f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="The cost of a discount factor near one"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("WHAT A DISCOUNT FACTOR NEAR ONE COSTS")
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
