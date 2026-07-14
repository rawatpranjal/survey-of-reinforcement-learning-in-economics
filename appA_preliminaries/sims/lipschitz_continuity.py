# Lipschitz Continuity
# Appendix A - Mathematical Preliminaries
# A function is L-Lipschitz if |f(x)-f(y)| <= L||x-y||; for a differentiable f the
# smallest such L is the supremum of the gradient norm. Two checks. (i) The empirical
# Lipschitz constant (max difference quotient over sampled pairs) rises to the analytic
# L = sup|f'| from below and never exceeds it. (ii) The policy-evaluation Bellman
# operator T V = r + gamma P V is exactly gamma-Lipschitz in the sup norm, so its
# fixed-point solve map r -> (I - gamma P)^{-1} r amplifies a reward perturbation by
# the operator norm ||(I - gamma P)^{-1}||_inf = 1/(1 - gamma). This ties Lipschitz
# continuity to the Banach contraction and the Neumann-series bound in this appendix.

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
SCRIPT_NAME = "lipschitz_continuity"
CONFIG = {
    # (i) scalar functions with a known Lipschitz constant on [-3, 3]
    "domain": 3.0,
    "n_pairs_grid": [100, 1000, 10000, 100000, 1000000],
    "pair_seeds": 12,
    # (ii) Bellman operator on random MDPs
    "n_states": 40,
    "gammas": [0.5, 0.7, 0.9, 0.95, 0.99],
    "n_mdp_seeds": 20,
    "seed_base": 77000,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)

# scalar test functions: name -> (callable, analytic Lipschitz constant L)
FUNCS = {
    "0.5x": (lambda x: 0.5 * x, 0.5),
    "tanh": (lambda x: np.tanh(x), 1.0),  # sup|f'| = 1 at 0
    "|x|": (lambda x: np.abs(x), 1.0),  # Lipschitz, non-differentiable
    "sin(3x)": (lambda x: np.sin(3.0 * x), 3.0),  # sup|f'| = 3
}
FUNC_COLORS = {
    "0.5x": COLORS["blue"],
    "tanh": COLORS["orange"],
    "|x|": COLORS["green"],
    "sin(3x)": COLORS["red"],
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _empirical_lipschitz(fn, dom, n_pairs, rng):
    """max_{i} |f(x_i) - f(y_i)| / |x_i - y_i| over random pairs."""
    x = rng.uniform(-dom, dom, size=n_pairs)
    y = rng.uniform(-dom, dom, size=n_pairs)
    d = np.abs(x - y)
    keep = d > 1e-9
    slopes = np.abs(fn(x[keep]) - fn(y[keep])) / d[keep]
    return float(np.max(slopes))


def _run_experiment():
    dom = CONFIG["domain"]
    grid = CONFIG["n_pairs_grid"]
    pair_seeds = CONFIG["pair_seeds"]

    print("(i) Empirical Lipschitz constant of scalar functions on [-D, D]:")
    lip = {}
    for name, (fn, L) in FUNCS.items():
        curve = np.zeros((pair_seeds, len(grid)))
        for si in range(pair_seeds):
            rng = np.random.RandomState(CONFIG["seed_base"] + si)
            for j, n in enumerate(grid):
                curve[si, j] = _empirical_lipschitz(fn, dom, n, rng)
        mean_curve = curve.mean(axis=0)
        Lhat = float(mean_curve[-1])
        Lhat_se = float(curve[:, -1].std() / np.sqrt(pair_seeds))
        over = float(np.max(curve))  # any pair ever exceeded L?
        lip[name] = {
            "L": L,
            "mean_curve": mean_curve,
            "Lhat": Lhat,
            "Lhat_se": Lhat_se,
            "ratio": Lhat / L,
            "max_ever": over,
            "exceeds": bool(over > L + 1e-9),
        }
        print(
            f"    {name:8s}: analytic L={L:.3f}, measured L_hat={Lhat:.4f} +/- {Lhat_se:.1e} "
            f"(ratio {Lhat / L:.4f}), max over all pairs={over:.4f}, exceeds L={lip[name]['exceeds']}"
        )

    print()
    print("(ii) Bellman operator T V = r + gamma P V (policy evaluation):")
    ns = CONFIG["n_states"]
    gammas = CONFIG["gammas"]
    n_seeds = CONFIG["n_mdp_seeds"]
    op = {}
    for gamma in gammas:
        op_lip = np.zeros(n_seeds)  # ||gamma P||_inf, the operator Lipschitz constant
        amp = np.zeros(
            n_seeds
        )  # ||(I - gamma P)^{-1}||_inf  (reward -> value amplification)
        for si in range(n_seeds):
            rng = np.random.RandomState(CONFIG["seed_base"] + 500 + si)
            P = rng.dirichlet(np.ones(ns), size=ns)  # row-stochastic
            # Operator Lipschitz constant = induced sup-norm ||gamma P||_inf = max abs row
            # sum of gamma P = gamma (P row-stochastic). This is a supremum over V-pairs,
            # attained by a constant-sign perturbation; random V-pairs would only probe the
            # much smaller typical ratio, not the Lipschitz constant.
            op_lip[si] = float(np.max(np.sum(np.abs(gamma * P), axis=1)))
            # reward-to-value amplification = operator norm of the resolvent
            R = np.linalg.inv(np.eye(ns) - gamma * P)
            amp[si] = float(
                np.max(np.sum(np.abs(R), axis=1))
            )  # max abs row sum = sup-norm
        op[str(gamma)] = {
            "op_lip_mean": float(op_lip.mean()),
            "op_lip_theory": gamma,
            "amp_mean": float(amp.mean()),
            "amp_se": float(amp.std() / np.sqrt(n_seeds)),
            "amp_theory": 1.0 / (1.0 - gamma),
        }
        r = op[str(gamma)]
        print(
            f"    gamma={gamma:.2f}: operator Lipschitz={r['op_lip_mean']:.4f} "
            f"(theory {gamma:.2f}); reward->value amplification="
            f"{r['amp_mean']:.4f} (theory 1/(1-gamma)={1 / (1 - gamma):.4f})"
        )

    return {"config": CONFIG, "lip": lip, "op": op}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "lipschitz",
        CONFIG,
        _run_experiment,
        force=("lipschitz" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    config = data["config"]
    lip = data["lip"]
    op = data["op"]
    grid = config["n_pairs_grid"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: empirical Lipschitz constant rises to analytic L ------------
    axA = axes[0]
    for name in FUNCS:
        r = lip[name]
        axA.semilogx(
            grid,
            r["mean_curve"],
            color=FUNC_COLORS[name],
            linewidth=1.6,
            marker="o",
            markersize=3,
            label=f"{name} ($L={r['L']:.1f}$)",
        )
        axA.axhline(
            r["L"], color=FUNC_COLORS[name], linestyle="--", linewidth=1.0, alpha=0.6
        )
    axA.set_xlabel("Number of sampled pairs")
    axA.set_ylabel(r"empirical $\hat L = \max |\Delta f| / |\Delta x|$")
    axA.set_title("Difference quotient rises to $L = \\sup |f'|$")
    axA.legend(loc="center right", fontsize=8)

    # --- Panel B: Bellman error amplification = 1/(1-gamma) -------------------
    axB = axes[1]
    gammas = config["gammas"]
    amp = [op[str(g)]["amp_mean"] for g in gammas]
    gg = np.linspace(min(gammas), max(gammas), 200)
    axB.plot(
        gg,
        1.0 / (1.0 - gg),
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.4,
        label=r"$1/(1-\gamma)$",
    )
    # markers only: the exact points sit on the (convex) theory curve, so a connecting
    # line would bow above it between the sampled gammas and misread as "exceeds theory"
    axB.plot(
        gammas,
        amp,
        color=COLORS["blue"],
        linewidth=0,
        marker="o",
        markersize=7,
        label=r"measured $\|(I-\gamma P)^{-1}\|_\infty$",
    )
    axB.set_xlabel(r"discount factor $\gamma$")
    axB.set_ylabel("reward-to-value amplification")
    axB.set_title("Bellman solve amplifies error by $1/(1-\\gamma)$")
    axB.legend(loc="upper left")

    fig_path = os.path.join(OUTPUT_DIR, "lipschitz_continuity.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "lipschitz_continuity.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Lipschitz continuity, two checks. Top: the empirical Lipschitz "
            "constant $\\hat L$ (largest difference quotient over sampled pairs) of four scalar "
            "functions approaches the analytic $L = \\sup|f'|$ from below and never exceeds it, "
            "attaining $L$ exactly for the two piecewise-linear cases. "
            "Bottom: the policy-evaluation Bellman operator $TV = r + \\gamma P V$ is exactly "
            "$\\gamma$-Lipschitz in the sup norm, and its fixed-point solve amplifies a reward "
            "perturbation by $\\|(I-\\gamma P)^{-1}\\|_\\infty = 1/(1-\\gamma)$. Means $\\pm$ SE over "
            + str(config["pair_seeds"])
            + " sampling seeds and "
            + str(config["n_mdp_seeds"])
            + " random MDPs.}\n"
        )
        f.write("\\label{tab:prelim_lipschitz}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write(
            "Function & Analytic $L$ & Measured $\\hat L$ & $\\hat L / L$ \\\\\n\\hline\n"
        )
        # math-mode display names (the document has no T1 fontenc, so a bare "|x|"
        # in text mode would render the pipes as dashes)
        tex_name = {
            "0.5x": "$0.5x$",
            "tanh": "$\\tanh x$",
            "|x|": "$|x|$",
            "sin(3x)": "$\\sin(3x)$",
        }
        for name in FUNCS:
            r = lip[name]
            f.write(
                f"{tex_name[name]} & {r['L']:.2f} & {r['Lhat']:.4f} $\\pm$ {r['Lhat_se']:.1e} "
                f"& {r['ratio']:.4f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write(
            "$\\gamma$ & Operator Lip.\\ ($=\\gamma$) & Amplification & "
            "Theory $1/(1-\\gamma)$ \\\\\n\\hline\n"
        )
        for gamma in config["gammas"]:
            r = op[str(gamma)]
            f.write(
                f"{gamma} & {r['op_lip_mean']:.4f} & {r['amp_mean']:.4f} & "
                f"{r['amp_theory']:.4f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Lipschitz continuity")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("LIPSCHITZ CONTINUITY")
    print("=" * 70)
    print()
    print("|f(x) - f(y)| <= L ||x - y||. For differentiable f, the least L is sup|f'|.")
    print("The Bellman operator is gamma-Lipschitz; the resolvent solve amplifies")
    print("a reward error by 1/(1-gamma).")
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
