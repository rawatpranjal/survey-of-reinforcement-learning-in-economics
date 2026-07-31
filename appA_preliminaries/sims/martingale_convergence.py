# Martingale Convergence
# Appendix A - Mathematical Preliminaries
# Doob's theorem: an L1-bounded martingale converges almost surely. The Polya urn
# red-ball fraction is a martingale bounded in [0,1], so every trajectory settles
# on a (random) limit. This sim checks (i) the martingale property E[M_{n+1}|F_n]=M_n
# numerically, (ii) that each path converges (tail oscillation -> 0), and (iii) that
# the limits follow the Beta(a0,b0) law the theory predicts (Uniform for a 1,1 start).

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
SCRIPT_NAME = "martingale_convergence"
CONFIG = {
    "start_red": 1,
    "start_black": 1,  # (1,1) start => limit fraction is Beta(1,1) = Uniform(0,1)
    "traj_seeds": 40,
    "traj_n": 50000,  # steps for the displayed trajectories
    "limit_seeds": 6000,
    "limit_n": 4000,  # steps before recording the (near-)limit fraction
    "tail_frac": 0.1,  # last 10% of a trajectory defines its "tail"
    "tail_eps": 0.02,  # a path has "settled" if its tail oscillation is below this
    "seed_base": 31000,
    "version": 1,
}

OUTPUT_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Polya urn, vectorized across seeds. Draw a ball, return it with one more of the
# same color. M_n = red / total is a bounded martingale.
# ---------------------------------------------------------------------------


def run_urn(
    n_seeds,
    n_steps,
    start_red,
    start_black,
    seed,
    record_idx=None,
    track_increment=False,
):
    """Simulate n_seeds independent urns for n_steps. Returns final fractions and,
    optionally, the fraction at each index in record_idx (shape n_seeds x len(record_idx))
    and increment statistics (sum, sumsq, count) of one-step changes M_{n+1}-M_n."""
    rng = np.random.RandomState(seed)
    red = np.full(n_seeds, float(start_red))
    total = np.full(n_seeds, float(start_red + start_black))
    frac = red / total
    rec = None
    if record_idx is not None:
        record_set = set(int(i) for i in record_idx)
        rec = np.zeros((n_seeds, len(record_idx)))
        idx_order = {int(i): j for j, i in enumerate(record_idx)}
    inc_sum = 0.0
    inc_sqsum = 0.0
    inc_count = 0
    for step in range(n_steps):
        u = rng.uniform(size=n_seeds)
        drew_red = u < frac  # probability = current red fraction
        red = red + drew_red.astype(float)
        total = total + 1.0
        new_frac = red / total
        if track_increment:
            d = new_frac - frac
            inc_sum += d.sum()
            inc_sqsum += (d * d).sum()
            inc_count += n_seeds
        frac = new_frac
        if rec is not None and (step + 1) in record_set:
            rec[:, idx_order[step + 1]] = frac
    stats_inc = (inc_sum, inc_sqsum, inc_count) if track_increment else None
    return frac, rec, stats_inc


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _run_experiment():
    sr, sb = CONFIG["start_red"], CONFIG["start_black"]
    traj_seeds, traj_n = CONFIG["traj_seeds"], CONFIG["traj_n"]
    limit_seeds, limit_n = CONFIG["limit_seeds"], CONFIG["limit_n"]
    seed_base = CONFIG["seed_base"]

    print(f"Martingale convergence: Polya urn, start (red,black)=({sr},{sb})")
    print(f"  trajectories: {traj_seeds} seeds x {traj_n} steps")
    print(f"  limit ensemble: {limit_seeds} seeds x {limit_n} steps")
    print()

    # --- trajectories on a log-spaced record grid --------------------------
    rec_idx = np.unique(np.round(np.logspace(0, np.log10(traj_n), 250)).astype(int))
    rec_idx = rec_idx[(rec_idx >= 1) & (rec_idx <= traj_n)]
    frac_final, traj_rec, _ = run_urn(
        traj_seeds, traj_n, sr, sb, seed_base, record_idx=rec_idx
    )

    # tail oscillation per trajectory: max |M_n - M_last| over the tail window
    tail_start = int((1 - CONFIG["tail_frac"]) * len(rec_idx))
    tail_block = traj_rec[:, tail_start:]
    tail_osc = np.max(np.abs(tail_block - traj_rec[:, -1:]), axis=1)
    frac_settled = float(np.mean(tail_osc < CONFIG["tail_eps"]))
    print(
        f"  tail oscillation (last {int(CONFIG['tail_frac'] * 100)}% of the log-spaced "
        f"record grid): mean {tail_osc.mean():.4f}, max {tail_osc.max():.4f}; "
        f"fraction of paths settled (< {CONFIG['tail_eps']}): {frac_settled:.3f}"
    )

    # --- limit ensemble: martingale property + limit law -------------------
    limits, _, inc_stats = run_urn(
        limit_seeds, limit_n, sr, sb, seed_base + 1, track_increment=True
    )
    inc_sum, inc_sqsum, inc_count = inc_stats
    inc_mean = inc_sum / inc_count
    inc_sd = np.sqrt(max(inc_sqsum / inc_count - inc_mean**2, 0.0))
    inc_se = inc_sd / np.sqrt(inc_count)
    print(
        f"  martingale increment E[M_(n+1)-M_n]: {inc_mean:+.2e} "
        f"+/- {inc_se:.2e} over {inc_count:,} increments (theory 0)"
    )

    # limit law: Beta(sr, sb); for (1,1) this is Uniform(0,1)
    beta = stats.beta(sr, sb)
    ks = float(stats.kstest(limits, beta.cdf).statistic)
    print(
        f"  limit fractions: mean {limits.mean():.4f} (theory {sr / (sr + sb):.4f}), "
        f"var {limits.var():.4f} "
        f"(theory {sr * sb / ((sr + sb) ** 2 * (sr + sb + 1)):.4f}), "
        f"KS to Beta({sr},{sb}) {ks:.4f}"
    )

    return {
        "config": CONFIG,
        "rec_idx": rec_idx,
        "traj_rec": traj_rec,
        "tail_osc": tail_osc,
        "frac_settled": frac_settled,
        "limits": limits,
        "inc_mean": inc_mean,
        "inc_se": inc_se,
        "inc_count": inc_count,
        "limit_mean": float(limits.mean()),
        "limit_var": float(limits.var()),
        "limit_mean_theory": sr / (sr + sb),
        "limit_var_theory": sr * sb / ((sr + sb) ** 2 * (sr + sb + 1)),
        "ks_limit": ks,
    }


def compute_data(force=None):
    force = force or set()
    return compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "martingale",
        CONFIG,
        _run_experiment,
        force=("martingale" in force),
    )


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_outputs(data):
    config = data["config"]
    rec_idx = data["rec_idx"]
    traj_rec = data["traj_rec"]
    limits = data["limits"]
    sr, sb = config["start_red"], config["start_black"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Panel A: bounded martingale trajectories each settle on a random limit
    axA = axes[0]
    for i in range(traj_rec.shape[0]):
        axA.semilogx(
            rec_idx, traj_rec[i], color=COLORS["blue"], linewidth=0.7, alpha=0.4
        )
    axA.axhline(
        config["start_red"] / (sr + sb), **BENCH_STYLE, label=r"start fraction $M_0$"
    )
    axA.set_ylim(-0.02, 1.02)
    axA.set_xlabel("Step $n$")
    axA.set_ylabel(r"red fraction $M_n$")
    axA.set_title("Bounded martingale: each path converges")
    axA.legend(loc="upper right")

    # --- Panel B: the random limits follow the Beta(sr,sb) law ---------------
    axB = axes[1]
    axB.hist(
        limits,
        bins=40,
        density=True,
        color=COLORS["blue"],
        alpha=0.55,
        label=r"limit $M_\infty$",
    )
    grid = np.linspace(0, 1, 400)
    axB.plot(
        grid,
        stats.beta(sr, sb).pdf(grid),
        color=COLORS["red"],
        linewidth=1.8,
        linestyle="--",
        label=rf"Beta$({sr},{sb})$",
    )
    axB.set_xlabel(r"limit fraction $M_\infty$")
    axB.set_ylabel("density")
    axB.set_title(r"Limits spread across $[0,1]$")
    axB.legend(loc="upper center")

    fig_path = os.path.join(OUTPUT_DIR, "martingale_convergence.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "martingale_convergence.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Almost-sure convergence of the P\\'olya-urn martingale "
            "$M_n = (\\text{red})/(\\text{total})$ from a $("
            + str(sr)
            + ","
            + str(sb)
            + ")$ start. The one-step increment has mean zero, consistent with the "
            "martingale property (the theorem's conditional statement is proved analytically), "
            "every path settles (tail oscillation below "
            + f"{config['tail_eps']}"
            + " for the reported fraction of paths), and the "
            "random limits match the Beta$("
            + str(sr)
            + ","
            + str(sb)
            + ")$ law in mean, variance, and distribution ($D_{\\mathrm{KS}}$). "
            "Limit ensemble of "
            + f"{config['limit_seeds']:,}".replace(",", "{,}")
            + " urns.}\n"
        )
        f.write("\\label{tab:prelim_martingale}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        f.write("Quantity & Simulated & Theory \\\\\n\\hline\n")
        f.write(
            f"Increment $E[M_{{n+1}}-M_n]$ & ${data['inc_mean']:+.2e}$ "
            f"$\\pm$ {data['inc_se']:.1e} & $0$ \\\\\n"
        )
        f.write(
            f"Paths settled (tail $< {config['tail_eps']}$) & "
            f"{data['frac_settled']:.3f} & $1$ \\\\\n"
        )
        f.write(
            f"Limit mean & {data['limit_mean']:.4f} & "
            f"{data['limit_mean_theory']:.4f} \\\\\n"
        )
        f.write(
            f"Limit variance & {data['limit_var']:.4f} & "
            f"{data['limit_var_theory']:.4f} \\\\\n"
        )
        f.write(
            f"$D_{{\\mathrm{{KS}}}}$ to Beta$({sr},{sb})$ & "
            f"{data['ks_limit']:.4f} & $0$ \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Martingale convergence (Polya urn)")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print("=" * 70)
    print("MARTINGALE CONVERGENCE: DOOB'S THEOREM VIA THE POLYA URN")
    print("=" * 70)
    print()
    print("An L1-bounded martingale converges almost surely. The Polya-urn red")
    print("fraction is bounded in [0,1], hence converges on every path to a random")
    print("limit; the limit law is Beta(start_red, start_black).")
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
