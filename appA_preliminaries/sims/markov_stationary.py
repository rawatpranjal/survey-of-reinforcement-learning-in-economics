# Markov Stationary Distribution and Perron-Frobenius Mixing
# Appendix A - Mathematical Preliminaries
# An irreducible aperiodic Markov chain has a unique stationary distribution d* with d* P = d*.
# Power iteration d_k = d_0 P^k converges to d* at the rate set by the second-largest
# eigenvalue modulus |lambda_2| (the spectral gap 1 - |lambda_2| is the mixing rate).

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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "markov_stationary"
CONFIG = {
    "n_states": 8,
    # mixing controls: 'stickiness' s sets P = s*I + (1-s)*R. Larger s -> slower mixing
    # (second eigenvalue nearer 1), smaller s -> faster mixing.
    "chains": [("slow", 0.85), ("medium", 0.5), ("fast", 0.1)],
    "n_iters": 150,
    "n_seeds": 30,
    "version": 2,
}
OUTPUT_DIR = os.path.dirname(__file__)
CHAIN_COLORS = {
    "slow": COLORS["red"],
    "medium": COLORS["green"],
    "fast": COLORS["blue"],
}


def random_chain(n, s, seed):
    """Irreducible aperiodic transition matrix P = s*I + (1-s)*R, R a random stochastic
    matrix. R has strictly positive Dirichlet rows, so P is strictly positive, hence
    primitive (irreducible and aperiodic). Larger stickiness s pushes the second eigenvalue
    toward 1, slowing the mix to stationarity."""
    rng = np.random.RandomState(seed)
    R = rng.dirichlet(np.ones(n), size=n)
    return s * np.eye(n) + (1.0 - s) * R


def stationary(P):
    """Left eigenvector of P for eigenvalue 1, normalized to a probability vector."""
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    d = np.real(vecs[:, idx])
    return d / d.sum()


def second_eigmod(P):
    vals = np.linalg.eigvals(P)
    mods = np.sort(np.abs(vals))[::-1]
    return float(mods[1])  # second-largest modulus


def power_iterate(P, d_star, n_iters, seed):
    rng = np.random.RandomState(seed)
    d = rng.dirichlet(np.ones(len(d_star)))  # arbitrary start
    err = np.zeros(n_iters + 1)
    err[0] = 0.5 * np.sum(np.abs(d - d_star))  # total-variation distance
    for k in range(1, n_iters + 1):
        d = d @ P
        err[k] = 0.5 * np.sum(np.abs(d - d_star))
    return err


def _run():
    n = CONFIG["n_states"]
    chains = CONFIG["chains"]
    n_iters = CONFIG["n_iters"]
    n_seeds = CONFIG["n_seeds"]
    print("Markov stationary distribution: d* P = d*, power iteration d_k = d_0 P^k")
    print(f"  states: {n}, seeds: {n_seeds}, iters: {n_iters}\n")
    results = {}
    for name, eps in chains:
        errs = np.zeros((n_seeds, n_iters + 1))
        lam2s = np.zeros(n_seeds)
        resid = np.zeros(n_seeds)
        for si in range(n_seeds):
            P = random_chain(n, eps, 300 + si)
            d_star = stationary(P)
            resid[si] = np.max(np.abs(d_star @ P - d_star))  # check d* P = d*
            lam2s[si] = second_eigmod(P)
            errs[si] = power_iterate(P, d_star, n_iters, 900 + si)
        mean_err = errs.mean(axis=0)
        lam2 = float(lam2s.mean())
        below = np.where(mean_err < 1e-6)[0]
        k_to_tol = int(below[0]) if len(below) else -1
        results[name] = {
            "mean_err": mean_err,
            "lam2": lam2,
            "resid": float(resid.max()),
            "k_to_tol": k_to_tol,
        }
        print(
            f"  {name:7s}: |lambda_2|={lam2:.4f}, ||d*P - d*||={resid.max():.2e}, "
            f"iters to 1e-6 = {k_to_tol}"
        )
    return {"results": results, "config": CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "markov", CONFIG, _run, force=("markov" in force)
    )


def generate_outputs(data):
    results = data["results"]
    chains = data["config"]["chains"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name, _eps in chains:
        r = results[name]
        k = np.arange(len(r["mean_err"]))
        c = CHAIN_COLORS[name]
        ax.semilogy(
            k,
            r["mean_err"],
            color=c,
            linewidth=1.8,
            label=f"{name} ($|\\lambda_2| = {r['lam2']:.2f}$)",
        )
        ax.semilogy(
            k,
            r["mean_err"][0] * r["lam2"] ** k,
            color=c,
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"TV distance $\|d_k - d^\star\|_{TV}$")
    ax.set_title("Mixing to the stationary distribution at rate $|\\lambda_2|$")
    ax.set_ylim(1e-8, None)
    ax.legend(loc="upper right", title="solid: measured, dashed: $|\\lambda_2|^k$")
    fig_path = os.path.join(OUTPUT_DIR, "markov_stationary.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "markov_stationary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Convergence of the state distribution $d_k = d_0 P^k$ to the stationary "
            "distribution $d^\\star$ on random 8-state chains, at three mixing speeds. The total-"
            "variation distance falls at rate $|\\lambda_2|$, the second-largest eigenvalue "
            "modulus. The residual confirms $d^\\star P = d^\\star$. Means over 30 seeds.}\n"
        )
        f.write("\\label{tab:prelim_markov}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write(
            "Chain & $|\\lambda_2|$ & $\\|d^\\star P - d^\\star\\|_\\infty$ & Iters to $10^{-6}$ \\\\\n"
        )
        f.write("\\hline\n")
        for name, _eps in chains:
            r = results[name]
            f.write(
                f"{name} & {r['lam2']:.4f} & {r['resid']:.1e} & {r['k_to_tol']} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Markov stationary distribution")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("MARKOV STATIONARY DISTRIBUTION AND MIXING")
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
