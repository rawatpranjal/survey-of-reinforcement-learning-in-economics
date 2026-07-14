# The Envelope (Danskin) Theorem
# Appendix A - Mathematical Preliminaries
# For a value function V(theta) = max_a f(a, theta), the derivative equals the partial
# of f at the maximizer, dV/dtheta = f_theta(a*(theta), theta): the dependence of a*
# on theta contributes nothing to first order. This is checked two ways. (i) On smooth
# conjugate families f(a,theta)=theta*a - a^p/p, a finite-difference derivative of the
# numerically-maximized V reproduces the partial f_theta(a*) = a*(theta) with no
# knowledge of a*. (ii) On a family of lines V(theta)=max_a (alpha_a + beta_a theta),
# the envelope is tangent to the active line everywhere and V'(theta)=beta_{a*} between
# kinks, where the maximizer switches (Danskin's directional-derivative case).

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
SCRIPT_NAME = "envelope_theorem"
CONFIG = {
    # smooth conjugate families f(a,theta) = theta*a - a^p/p; a*(theta)=theta^{1/(p-1)}
    "p_values": [2, 4, 6],
    "a_max": 3.5,
    "n_a": 20000,
    "theta_min": 0.2,
    "theta_max": 3.0,
    "n_theta": 240,
    # line family for the envelope geometry / Danskin kinks
    "n_lines": 8,
    "line_seeds": 15,
    "line_theta_min": 0.0,
    "line_theta_max": 2.0,
    "n_theta_line": 400,
    "seed_base": 66000,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)
P_COLORS = {2: COLORS["blue"], 4: COLORS["orange"], 6: COLORS["green"]}


# ---------------------------------------------------------------------------
# (i) Smooth family: envelope identity via finite-differencing a numerically
#     maximized value function.
# ---------------------------------------------------------------------------


def smooth_family(p):
    a = np.linspace(1e-6, CONFIG["a_max"], CONFIG["n_a"])
    theta = np.linspace(CONFIG["theta_min"], CONFIG["theta_max"], CONFIG["n_theta"])
    # f(a, theta) = theta*a - a^p/p, maximized over a on the grid for each theta
    fa = -(a**p) / p  # theta-independent part
    V = np.zeros_like(theta)
    a_star = np.zeros_like(theta)
    for i, th in enumerate(theta):
        f = th * a + fa
        j = np.argmax(f)
        V[i] = f[j]
        a_star[i] = a[j]
    # dV/dtheta by central differences (V knows nothing about a*)
    Vp = np.gradient(V, theta)
    # envelope theorem: dV/dtheta should equal f_theta(a*, theta) = a*(theta)
    partial = a_star  # f_theta = a
    resid = np.abs(Vp - partial)
    # analytic references
    a_star_analytic = theta ** (1.0 / (p - 1))
    V_analytic = ((p - 1) / p) * theta ** (p / (p - 1))
    return {
        "theta": theta,
        "V": V,
        "a_star": a_star,
        "Vp": Vp,
        "partial": partial,
        "resid": resid,
        "max_resid": float(np.max(resid[1:-1])),  # ignore the one-sided endpoints
        "max_V_err": float(np.max(np.abs(V - V_analytic))),
        "max_astar_err": float(np.max(np.abs(a_star - a_star_analytic))),
    }


# ---------------------------------------------------------------------------
# (ii) Line family: V(theta) = max_a (alpha_a + beta_a theta), envelope geometry.
# ---------------------------------------------------------------------------


def line_family(seed):
    rng = np.random.RandomState(seed)
    K = CONFIG["n_lines"]
    alpha = rng.normal(size=K)
    beta = rng.normal(size=K)
    theta = np.linspace(
        CONFIG["line_theta_min"], CONFIG["line_theta_max"], CONFIG["n_theta_line"]
    )
    lines = alpha[:, None] + beta[:, None] * theta[None, :]  # K x n_theta
    active = np.argmax(lines, axis=0)
    V = lines[active, np.arange(len(theta))]
    Vp = np.gradient(V, theta)
    beta_active = beta[active]  # V'(theta) = beta_{a*(theta)} away from kinks
    # kinks are where the active line switches
    switches = np.where(np.diff(active) != 0)[0]
    # identity holds where the active line is stable in a neighborhood (not a kink cell)
    is_kink = np.zeros(len(theta), dtype=bool)
    for s in switches:
        is_kink[max(s - 1, 0) : min(s + 2, len(theta))] = True
    resid = np.abs(Vp - beta_active)
    non_kink = ~is_kink
    frac_identity = float(np.mean(resid[non_kink] < 1e-6)) if non_kink.any() else 0.0
    return {
        "theta": theta,
        "lines": lines,
        "V": V,
        "alpha": alpha,
        "beta": beta,
        "n_kinks": int(len(switches)),
        "frac_identity": frac_identity,
    }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    print("Envelope (Danskin) theorem: dV/dtheta = f_theta(a*(theta), theta)")
    print()

    print("(i) Smooth conjugate families f(a,theta) = theta*a - a^p/p:")
    smooth = {}
    for p in CONFIG["p_values"]:
        s = smooth_family(p)
        smooth[str(p)] = s
        print(
            f"    p={p}: max envelope residual |V'-f_theta(a*)| = {s['max_resid']:.2e}, "
            f"max |V - V_analytic| = {s['max_V_err']:.2e}, "
            f"max |a* - a*_analytic| = {s['max_astar_err']:.2e}"
        )

    print()
    print("(ii) Line families V(theta) = max_a (alpha_a + beta_a theta):")
    fracs = np.zeros(CONFIG["line_seeds"])
    kinks = np.zeros(CONFIG["line_seeds"])
    example_line = None
    for si in range(CONFIG["line_seeds"]):
        lf = line_family(CONFIG["seed_base"] + si)
        fracs[si] = lf["frac_identity"]
        kinks[si] = lf["n_kinks"]
        if si == 0:
            example_line = lf
    print(
        f"    over {CONFIG['line_seeds']} seeds: mean identity-holds fraction "
        f"(off kinks) = {fracs.mean():.4f}, mean kinks per envelope = {kinks.mean():.1f}"
    )

    return {
        "config": CONFIG,
        "smooth": smooth,
        "example_line": example_line,
        "line_frac_mean": float(fracs.mean()),
        "line_frac_min": float(fracs.min()),
        "line_kinks_mean": float(kinks.mean()),
        "line_seeds": CONFIG["line_seeds"],
    }


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "envelope",
        CONFIG,
        _run_experiment,
        force=("envelope" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    smooth = data["smooth"]
    ex = data["example_line"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: line family and its upper envelope (the tangent picture) ----
    axA = axes[0]
    theta = np.array(ex["theta"])
    lines = np.array(ex["lines"])
    for a in range(lines.shape[0]):
        axA.plot(theta, lines[a], color=COLORS["gray"], linewidth=0.8, alpha=0.6)
    axA.plot(
        theta,
        ex["V"],
        color=COLORS["red"],
        linewidth=2.2,
        label=r"envelope $V(\theta)=\max_a f(a,\theta)$",
    )
    axA.set_xlabel(r"$\theta$")
    axA.set_ylabel("value")
    axA.set_title("Value is the upper envelope of the family")
    axA.legend(loc="upper left")

    # --- Panel B: envelope residual for the smooth families -------------------
    axB = axes[1]
    for p in data["config"]["p_values"]:
        s = smooth[str(p)]
        th = np.array(s["theta"])[1:-1]
        rr = np.array(s["resid"])[1:-1]
        axB.semilogy(
            th,
            np.maximum(rr, 1e-16),
            color=P_COLORS[p],
            linewidth=1.4,
            label=rf"$p={p}$",
        )
    axB.set_xlabel(r"$\theta$")
    axB.set_ylabel(r"$|\,dV/d\theta - f_\theta(a^\star,\theta)\,|$")
    axB.set_title("Envelope residual at the numerical floor")
    axB.legend(loc="best", title=r"$f=\theta a - a^p/p$")

    fig_path = os.path.join(OUTPUT_DIR, "envelope_theorem.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "envelope_theorem.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{The envelope theorem $dV/d\\theta = f_\\theta(a^\\star(\\theta),\\theta)$ "
            "checked on smooth conjugate families $f(a,\\theta)=\\theta a - a^p/p$, for which "
            "$a^\\star(\\theta)=\\theta^{1/(p-1)}$. A central-difference derivative of the "
            "numerically maximized $V$, which uses no knowledge of $a^\\star$, reproduces the "
            "partial $f_\\theta(a^\\star,\\theta)=a^\\star$ to the finite-difference floor; the "
            "numerical $V$ and $a^\\star$ match their closed forms. The last row reports the "
            "line-family envelope, where the identity holds at every $\\theta$ off the "
            "measure-zero kinks.}\n"
        )
        f.write("\\label{tab:prelim_envelope}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write(
            "Family & Max envelope residual & Max $|V-V_{\\mathrm{exact}}|$ & "
            "Max $|a^\\star - a^\\star_{\\mathrm{exact}}|$ \\\\\n\\hline\n"
        )
        for p in data["config"]["p_values"]:
            s = smooth[str(p)]
            f.write(
                f"$p={p}$ & {s['max_resid']:.2e} & {s['max_V_err']:.2e} & "
                f"{s['max_astar_err']:.2e} \\\\\n"
            )
        f.write("\\hline\n")
        f.write(
            f"Lines (off kinks) & identity holds on {data['line_frac_mean']:.3f} "
            f"of $\\theta$ & \\multicolumn{{2}}{{c}}{{"
            f"{data['line_kinks_mean']:.1f} kinks per envelope}} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Envelope (Danskin) theorem")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("THE ENVELOPE (DANSKIN) THEOREM")
    print("=" * 70)
    print()
    print(
        "V(theta) = max_a f(a, theta) has derivative dV/dtheta = f_theta(a*(theta), theta):"
    )
    print("the change in the maximizer contributes nothing to first order.")
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
