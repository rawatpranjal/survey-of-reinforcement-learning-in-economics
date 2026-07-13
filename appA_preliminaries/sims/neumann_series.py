# Neumann Series / Resolvent: (I - gamma P)^{-1} = sum_m gamma^m P^m
# Appendix A - Mathematical Preliminaries
# The value function V = (I - gamma P)^{-1} r equals the discounted sum of future rewards,
# sum_m gamma^m P^m r. Truncating the sum at M terms leaves an error bounded by
# gamma^{M+1}/(1-gamma) times ||r||, so a finite horizon approximates the infinite one.

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
SCRIPT_NAME = "neumann_series"
CONFIG = {
    "n_states": 40,
    "gammas": [0.5, 0.8, 0.95],
    "n_terms": 400,
    "n_seeds": 30,
    "version": 1,
}
OUTPUT_DIR = os.path.dirname(__file__)
GAMMA_COLORS = {0.5: COLORS["blue"], 0.8: COLORS["green"], 0.95: COLORS["red"]}


def random_mdp(n, seed):
    rng = np.random.RandomState(seed)
    P = rng.dirichlet(np.ones(n), size=n)
    r = rng.uniform(-1.0, 1.0, size=n)
    return P, r


def truncation_errors(P, r, gamma, n_terms):
    """||V - V_M||_inf for M = 0..n_terms, where V_M = sum_{m=0}^{M} gamma^m P^m r."""
    n = len(r)
    V = np.linalg.solve(np.eye(n) - gamma * P, r)  # exact resolvent
    partial = np.zeros(n)
    term = r.copy()  # gamma^0 P^0 r
    err = np.zeros(n_terms + 1)
    for m in range(n_terms + 1):
        partial = partial + term
        err[m] = np.max(np.abs(V - partial))
        term = gamma * (P @ term)  # gamma^{m+1} P^{m+1} r
    return err


def _run():
    n = CONFIG["n_states"]
    gammas = CONFIG["gammas"]
    n_terms = CONFIG["n_terms"]
    n_seeds = CONFIG["n_seeds"]
    print("Neumann series: V = (I - gamma P)^-1 r = sum_m gamma^m P^m r")
    print(f"  states: {n}, seeds: {n_seeds}, terms: {n_terms}, gammas: {gammas}\n")
    results = {}
    for gamma in gammas:
        errs = np.zeros((n_seeds, n_terms + 1))
        rnorms = np.zeros(n_seeds)
        for si in range(n_seeds):
            P, r = random_mdp(n, 200 + si)
            errs[si] = truncation_errors(P, r, gamma, n_terms)
            rnorms[si] = np.max(np.abs(r))
        mean_err = errs.mean(axis=0)
        rbar = float(rnorms.mean())
        # theoretical bound gamma^{M+1}/(1-gamma) * ||r||
        M = np.arange(n_terms + 1)
        bound = rbar * gamma ** (M + 1) / (1.0 - gamma)
        # terms to reach 1e-6
        below = np.where(mean_err < 1e-6)[0]
        m_to_tol = int(below[0]) if len(below) else -1
        results[str(gamma)] = {
            "mean_err": mean_err,
            "bound": bound,
            "m_to_tol": m_to_tol,
            # invert the bound rbar*gamma^{M+1}/(1-gamma) = tol for M+1, then -1 to get the
            # term index M (matching the measured 'terms to tol' column's convention)
            "m_pred": float(np.log(1e-6 * (1 - gamma) / rbar) / np.log(gamma) - 1.0)
            if rbar > 0
            else float("nan"),
        }
        print(
            f"  gamma={gamma}: terms to 1e-6 = {m_to_tol} (predicted "
            f"{results[str(gamma)]['m_pred']:.0f})"
        )
    return {"results": results, "config": CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "neumann", CONFIG, _run, force=("neumann" in force)
    )


def generate_outputs(data):
    results = data["results"]
    gammas = data["config"]["gammas"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for gamma in gammas:
        r = results[str(gamma)]
        M = np.arange(len(r["mean_err"]))
        c = GAMMA_COLORS[gamma]
        ax.semilogy(
            M, r["mean_err"], color=c, linewidth=1.8, label=f"$\\gamma = {gamma}$"
        )
        ax.semilogy(M, r["bound"], color=c, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Terms kept $M$")
    ax.set_ylabel(r"$\|V - V_M\|_\infty$")
    ax.set_title("Neumann truncation error decays geometrically")
    ax.set_ylim(1e-10, None)
    ax.legend(loc="upper right", title="solid: error, dashed: bound")
    fig_path = os.path.join(OUTPUT_DIR, "neumann_series.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "neumann_series.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Truncating the Neumann series $V = \\sum_m \\gamma^m P^m r$ at $M$ terms, "
            "on random 40-state chains. The error falls at rate $\\gamma$, matching the bound "
            "$\\gamma^{M+1}\\|r\\|_\\infty/(1-\\gamma)$. Terms needed to reach $10^{-6}$, measured "
            "and predicted from the bound.}\n"
        )
        f.write("\\label{tab:prelim_neumann}\n")
        f.write("\\begin{tabular}{ccc}\n\\hline\n")
        f.write("$\\gamma$ & Terms to $10^{-6}$ & Predicted \\\\\n")
        f.write("\\hline\n")
        for gamma in gammas:
            r = results[str(gamma)]
            f.write(f"{gamma} & {r['m_to_tol']} & {r['m_pred']:.0f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Neumann series resolvent")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("NEUMANN SERIES / RESOLVENT")
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
