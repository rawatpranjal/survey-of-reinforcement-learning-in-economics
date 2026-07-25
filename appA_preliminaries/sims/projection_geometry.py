# Projection Geometry and the Deadly Triad
# Appendix A - Mathematical Preliminaries
# Where exactly does the deadly triad bite? Holds the Bellman operator, the features and the
# target policy fixed, and varies only the distribution that defines the projection. The
# composed operator Pi_d T^pi contracts under the target policy's own stationary law and
# expands under a logging law far from it. Uses the two-state running example, small enough
# that the modulus is a closed-form scalar and every claim is checkable by hand.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_TRIPLE

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The environment, the policies and the feature are all fixed by the running example, so
# this script never re-chooses them.
from running_example import (  # noqa: E402
    GOOD,
    WORN,
    KEEP,
    REPLACE,
    build_mdp,
    policy_matrices,
    stochastic_policy_matrices,
    stationary_distribution,
    exact_value,
    projected_modulus,
    compute_data as running_compute_data,
)

apply_style()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "projection_geometry"
OUTPUT_DIR = os.path.dirname(__file__)

CONFIG = {
    "n_alpha": 401,  # resolution of the weighting sweep
    "n_iters": 60,  # projected-Bellman iterations per weighting
    "trace_alphas": [0.0, 0.7048, 1.0],
    "version": 1,
}

ALPHA_COLORS = [COLORS["blue"], COLORS["gray"], COLORS["red"]]


def _setup():
    """Rebuild the running example's primitives and the two weightings."""
    params = running_compute_data()["params"]
    gamma = params["gamma"]
    P, r = build_mdp(
        params["r_keep_good"],
        params["r_keep_worn"],
        params["replace_cost"],
        params["degrade_prob"],
    )
    P_pi, r_pi = policy_matrices(P, r, [KEEP, REPLACE])
    P_mu, _, _ = stochastic_policy_matrices(P, r, params["behavior_keep_prob"])
    d_pi = stationary_distribution(P_pi)
    d_mu = stationary_distribution(P_mu)
    phi = np.array([1.0, params["feature_ratio"]])
    V_pi = exact_value(P_pi, r_pi, gamma)
    return {
        "params": params,
        "gamma": gamma,
        "P_pi": P_pi,
        "r_pi": r_pi,
        "d_pi": d_pi,
        "d_mu": d_mu,
        "phi": phi,
        "V_pi": V_pi,
    }


def projection_matrix(phi, d):
    Phi = phi.reshape(-1, 1)
    D = np.diag(d)
    return Phi @ np.linalg.inv(Phi.T @ D @ Phi) @ Phi.T @ D


def _sweep(setup):
    """Modulus of Pi_d T^pi as the weighting is tilted from d^pi to d^mu."""
    gamma, phi, P_pi = setup["gamma"], setup["phi"], setup["P_pi"]
    d_pi, d_mu = setup["d_pi"], setup["d_mu"]
    alphas = np.linspace(0.0, 1.0, CONFIG["n_alpha"])
    mods = np.array(
        [
            projected_modulus(phi, (1 - a) * d_pi + a * d_mu, P_pi, gamma)[0]
            for a in alphas
        ]
    )

    # Exact crossing by bisection, not by reading the grid.
    def excess(a):
        return projected_modulus(phi, (1 - a) * d_pi + a * d_mu, P_pi, gamma)[0] - 1.0

    lo, hi = 0.0, 1.0
    assert excess(lo) < 0.0 < excess(hi), "the sweep does not bracket the crossing"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    crossing = 0.5 * (lo + hi)

    print("Weighting sweep from the target law d^pi to the logging law d^mu")
    print(f"  {'alpha':>7s}  {'d(good)':>8s}  {'modulus':>8s}")
    for a in [0.0, 0.25, 0.5, 0.7048, 0.75, 1.0]:
        d = (1 - a) * d_pi + a * d_mu
        m = projected_modulus(phi, d, P_pi, gamma)[0]
        print(f"  {a:7.4f}  {d[GOOD]:8.4f}  {m:8.4f}")
    print(f"  modulus crosses one at alpha = {crossing:.6f}")
    print(f"  residual at the crossing     = {excess(crossing):.2e}")
    print()
    return {"alphas": alphas, "moduli": mods, "crossing": float(crossing)}


def _traces(setup):
    """Iterate the projected Bellman operator under three weightings."""
    gamma, phi, P_pi, r_pi = setup["gamma"], setup["phi"], setup["P_pi"], setup["r_pi"]
    d_pi, d_mu, V_pi = setup["d_pi"], setup["d_mu"], setup["V_pi"]
    n = CONFIG["n_iters"]
    out = {}
    print("Projected Bellman iteration from V_0 = 0, sup-norm distance to V^pi")
    print(
        f"  {'alpha':>7s}  {'modulus':>8s}  {'error at k=10':>14s}  {'error at k=60':>14s}"
    )
    for a in CONFIG["trace_alphas"]:
        d = (1 - a) * d_pi + a * d_mu
        Pi = projection_matrix(phi, d)
        mod = projected_modulus(phi, d, P_pi, gamma)[0]
        V = np.zeros(2)
        errs = [float(np.max(np.abs(V - V_pi)))]
        for _ in range(n):
            V = Pi @ (r_pi + gamma * P_pi @ V)
            errs.append(float(np.max(np.abs(V - V_pi))))
        out[str(a)] = {"errors": np.array(errs), "modulus": float(mod)}
        print(f"  {a:7.4f}  {mod:8.4f}  {errs[10]:14.4f}  {errs[n]:14.4e}")
    print()
    return out


def _geometry(setup):
    """The two projections of V^pi onto the feature line, and their residual angles."""
    phi, V_pi = setup["phi"], setup["V_pi"]
    d_pi, d_mu = setup["d_pi"], setup["d_mu"]
    gamma = setup["gamma"]
    rows = {}
    print("Projections of V^pi onto span(phi)")
    for name, d in [("on-policy", d_pi), ("off-policy", d_mu)]:
        Pi = projection_matrix(phi, d)
        proj = Pi @ V_pi
        resid = V_pi - proj
        # Angle between the residual and the feature line, measured in the TARGET
        # policy's geometry. Ninety degrees means orthogonal there; anything else is
        # oblique in the geometry that the Bellman contraction actually uses.
        D_pi = np.diag(d_pi)
        cos = (resid @ D_pi @ phi) / (
            np.sqrt(resid @ D_pi @ resid) * np.sqrt(phi @ D_pi @ phi)
        )
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        rows[name] = {
            "Pi": Pi,
            "proj": proj,
            "resid": resid,
            "angle_in_dpi": float(angle),
        }
        print(
            f"  {name:11s}: Pi V^pi = [{proj[GOOD]:7.4f}, {proj[WORN]:7.4f}], "
            f"angle of the residual to the feature line in the d^pi geometry = {angle:6.2f} deg"
        )
    # The 90 degrees is an identity of Pi_{d^pi}, not a measurement: it would print for any
    # MDP and any feature. The informative number is the other one, and the quantity that
    # actually decides whether the composition contracts is the operator norm of each
    # projection measured in the d^pi geometry, which is computed here rather than asserted.
    D_pi = np.diag(d_pi)
    L = np.linalg.cholesky(D_pi)
    for name, d in [("on-policy", d_pi), ("off-policy", d_mu)]:
        Pi = projection_matrix(phi, d)
        # ||Pi||_{d^pi} = largest singular value of L^T Pi L^{-T}, the same operator in
        # coordinates where the d^pi inner product is the ordinary Euclidean one.
        opnorm = float(np.linalg.norm(L.T @ Pi @ np.linalg.inv(L.T), 2))
        rows[name]["opnorm_dpi"] = opnorm
        approx_err = float(np.max(np.abs(rows[name]["proj"] - V_pi)))
        rows[name]["approx_error"] = approx_err
        print(
            f"  {name:11s}: ||Pi||_(d^pi) = {opnorm:.4f}, "
            f"best representable approximation error = {approx_err:.4f}"
        )
    print(
        "  a projection is nonexpansive in its own geometry, so the on-policy value is 1;"
    )
    print(
        f"  the off-policy projection exceeds 1/gamma = {1 / gamma:.4f} in the d^pi geometry,"
    )
    print("  which is what lets the composition with the gamma-contraction expand")
    print(
        "  (the 90 degrees above is an algebraic identity of the on-policy projection, not"
    )
    print("  a measurement; the informative number is the off-policy angle)")
    assert abs(rows["on-policy"]["opnorm_dpi"] - 1.0) < 1e-9
    assert rows["off-policy"]["opnorm_dpi"] > 1.0 / gamma
    print()
    return rows


def compute_data(force=None):
    force = force or set()
    # Every result here depends on the environment, the feature and the logging policy
    # chosen upstream by running_example.py. Those must be inside the cache key, or a
    # change to the environment leaves this script's cached figure and table untouched
    # and silently wrong. CONFIG alone does not mention them.
    upstream = running_compute_data()["params"]
    key = {**CONFIG, "upstream": upstream}
    setup = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, "setup", key, _setup, force=("setup" in force)
    )
    cascade = "setup" in force
    sweep = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "sweep",
        key,
        _sweep,
        setup,
        force=("sweep" in force or cascade),
    )
    traces = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "traces",
        key,
        _traces,
        setup,
        force=("traces" in force or cascade),
    )
    geometry = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "geometry",
        key,
        _geometry,
        setup,
        force=("geometry" in force or cascade),
    )
    return {"setup": setup, "sweep": sweep, "traces": traces, "geometry": geometry}


def generate_outputs(data):
    setup, sweep, traces, geom = (
        data["setup"],
        data["sweep"],
        data["traces"],
        data["geometry"],
    )
    phi, V_pi = setup["phi"], setup["V_pi"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_TRIPLE[0], FIG_TRIPLE[1]))

    # Panel (a): the feature line, V^pi, and the two projections.
    ax = axes[0]
    t = np.linspace(0, 5.2, 2)
    ax.plot(
        t,
        t * phi[WORN] / phi[GOOD],
        color=COLORS["black"],
        lw=1.4,
        label=r"$\mathrm{span}(\Phi)$",
    )
    ax.plot(*V_pi, "o", color=COLORS["green"], ms=7, zorder=5)
    ax.annotate(
        r"$V^{\pi^\star}$",
        V_pi,
        textcoords="offset points",
        xytext=(8, -10),
        color=COLORS["green"],
    )
    for name, color in [("on-policy", COLORS["blue"]), ("off-policy", COLORS["red"])]:
        p = geom[name]["proj"]
        ax.plot(*p, "o", color=color, ms=6, zorder=5)
        ax.plot(
            [V_pi[GOOD], p[GOOD]],
            [V_pi[WORN], p[WORN]],
            color=color,
            lw=1.2,
            ls="--",
            label=f"{name} residual, {geom[name]['angle_in_dpi']:.0f}$^\\circ$",
        )
    ax.set_xlabel(r"$V(\mathrm{good})$")
    ax.set_ylabel(r"$V(\mathrm{worn})$")
    ax.set_title("(a) two projections onto one line")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 11.0)

    # Panel (b): the modulus against the weighting.
    ax = axes[1]
    ax.plot(sweep["alphas"], sweep["moduli"], color=COLORS["black"], lw=1.6)
    ax.axhline(1.0, color=COLORS["red"], lw=1.0, ls="--")
    ax.axvline(sweep["crossing"], color=COLORS["red"], lw=1.0, ls=":")
    ax.annotate(
        "crosses one at\n" + r"$\alpha = $" + f"{sweep['crossing']:.4f}",
        (sweep["crossing"], 1.0),
        textcoords="offset points",
        xytext=(-92, 22),
        color=COLORS["red"],
        fontsize=7,
    )
    ax.set_xlabel(r"$\alpha$, weighting tilted from $d^{\pi^\star}$ to $d^\mu$")
    ax.set_ylabel(r"modulus of $\Pi_d T^{\pi^\star}$ on $\mathrm{span}(\Phi)$")
    ax.set_title("(b) where the contraction is lost")

    # Panel (c): the iterates.
    ax = axes[2]
    for color, a in zip(ALPHA_COLORS, CONFIG["trace_alphas"]):
        res = traces[str(a)]
        ax.semilogy(
            res["errors"],
            color=color,
            lw=1.5,
            label=r"$\alpha = $" + f"{a:g}, modulus {res['modulus']:.3f}",
        )
    ax.set_xlabel("projected Bellman iteration $k$")
    ax.set_ylabel(r"$\|V_k - V^{\pi^\star}\|_\infty$")
    ax.set_title("(c) the iterates")
    ax.legend(loc="upper left", fontsize=7)

    fig_path = os.path.join(OUTPUT_DIR, "projection_geometry.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tex_path = os.path.join(OUTPUT_DIR, "projection_geometry.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{The projected Bellman operator on the running example under three "
            "weightings, from the target policy's own stationary law ($\\alpha = 0$) to the "
            "logging law ($\\alpha = 1$). The operator, the features and the target policy "
            "are identical across rows; only the distribution defining the projection "
            "changes. Errors are supremum-norm distances from $V^{\\pi^\\star}$ after the "
            "stated number of projected Bellman iterations from $V_0 = 0$.}\n"
        )
        f.write("\\label{tab:prelim_projection_geometry}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write(
            "$\\alpha$ & $d(\\mathrm{good})$ & modulus & error at $k = 10$ & error at $k = 60$ \\\\\n"
        )
        f.write("\\hline\n")
        d_pi, d_mu = setup["d_pi"], setup["d_mu"]
        for a in CONFIG["trace_alphas"]:
            res = traces[str(a)]
            d = (1 - a) * d_pi + a * d_mu
            e10, e60 = res["errors"][10], res["errors"][CONFIG["n_iters"]]
            f.write(
                f"{a:g} & {d[GOOD]:.4f} & {res['modulus']:.4f} & "
                f"{e10:.4f} & {e60:.4g} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Projection geometry and the deadly triad"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("PROJECTION GEOMETRY AND THE DEADLY TRIAD")
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
