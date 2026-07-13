# Spectral Radius Governs Asymptotic Decay: rho(A) < 1 iff A^k -> 0
# Appendix A - Mathematical Preliminaries
# Powers of a matrix decay to zero exactly when its spectral radius is below 1. The decay
# rate is the spectral radius, not the norm: a non-normal matrix can have ||A|| > 1 yet
# still decay, after a transient. Illustrates Gelfand's formula ||A^k||^{1/k} -> rho(A).

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
SCRIPT_NAME = "spectral_radius"
CONFIG = {
    "rhos": [0.5, 0.8, 0.95],
    "shear": 2.0,  # off-diagonal that makes A non-normal (norm > rho)
    "n_iters": 200,
    "version": 2,
}
OUTPUT_DIR = os.path.dirname(__file__)

RHO_COLORS = {0.5: COLORS["blue"], 0.8: COLORS["green"], 0.95: COLORS["red"]}


def make_matrix(rho, shear):
    """2x2 upper-triangular matrix with both eigenvalues = rho (so spectral radius = rho)
    and a large off-diagonal 'shear' that makes the operator 2-norm exceed rho."""
    return np.array([[rho, shear], [0.0, rho]])


def power_norms(A, n_iters):
    norms = np.zeros(n_iters + 1)
    M = np.eye(A.shape[0])
    norms[0] = np.linalg.norm(M, 2)
    for k in range(1, n_iters + 1):
        M = M @ A
        norms[k] = np.linalg.norm(M, 2)  # spectral (operator 2-) norm
    return norms


def _run():
    rhos = CONFIG["rhos"]
    shear = CONFIG["shear"]
    n_iters = CONFIG["n_iters"]
    print(f"Spectral radius vs norm: A = [[rho, {shear}], [0, rho]] (non-normal)")
    print(f"  rhos: {rhos}, iters: {n_iters}\n")
    results = {}
    for rho in rhos:
        A = make_matrix(rho, shear)
        norms = power_norms(A, n_iters)
        A_norm = np.linalg.norm(A, 2)
        peak = float(norms.max())
        peak_k = int(norms.argmax())
        # tail per-step decay ratio ||A^{k+1}|| / ||A^k|| over the last 30 steps -> rho
        ratios = norms[-30:] / norms[-31:-1]
        tail_ratio = float(ratios.mean())
        results[str(rho)] = {
            "norms": norms,
            "A_norm": float(A_norm),
            "peak": peak,
            "peak_k": peak_k,
            "tail_ratio": tail_ratio,
        }
        print(
            f"  rho={rho}: ||A||_2={A_norm:.3f}, transient peak={peak:.2f} at k={peak_k}, "
            f"tail decay ratio={tail_ratio:.4f} (-> rho)"
        )
    return {"results": results, "config": CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "spectral", CONFIG, _run, force=("spectral" in force)
    )


def generate_outputs(data):
    results = data["results"]
    rhos = data["config"]["rhos"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for rho in rhos:
        norms = results[str(rho)]["norms"]
        k = np.arange(len(norms))
        c = RHO_COLORS[rho]
        ax.semilogy(k, norms, color=c, linewidth=1.8, label=f"$\\rho = {rho}$")
        ax.semilogy(k, rho**k, color=c, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Power $k$")
    ax.set_ylabel(r"$\|A^k\|_2$")
    ax.set_title("Spectral radius sets the decay rate, not the norm")
    ax.legend(loc="upper right", title="solid: $\\|A^k\\|_2$, dashed: $\\rho^k$")
    fig_path = os.path.join(OUTPUT_DIR, "spectral_radius.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "spectral_radius.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Powers of the non-normal matrix $A = \\left[\\begin{smallmatrix} "
            "\\rho & 2 \\\\ 0 & \\rho \\end{smallmatrix}\\right]$. The operator norm exceeds "
            "$\\rho$ and the powers first grow, yet they decay asymptotically at rate $\\rho$. "
            "The tail decay ratio $\\|A^{k+1}\\|/\\|A^k\\|$ approaches $\\rho$ from above, "
            "with a benign $+O(1/k)$ finite-sample bias.}\n"
        )
        f.write("\\label{tab:prelim_spectral_radius}\n")
        f.write("\\begin{tabular}{ccccc}\n\\hline\n")
        f.write(
            "$\\rho(A)$ & $\\|A\\|_2$ & Transient peak & at $k$ & Tail decay ratio \\\\\n"
        )
        f.write("\\hline\n")
        for rho in rhos:
            r = results[str(rho)]
            f.write(
                f"{rho} & {r['A_norm']:.3f} & {r['peak']:.2f} & {r['peak_k']} & {r['tail_ratio']:.4f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Spectral radius governs decay")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("SPECTRAL RADIUS GOVERNS ASYMPTOTIC DECAY")
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
