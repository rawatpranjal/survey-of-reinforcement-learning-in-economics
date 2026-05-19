# Durable-Goods Monopoly: The Coase Conjecture --- Chapter 6, RL in Games
#
# Numerical demonstration of the Coase conjecture via dynamic programming on a
# finite-horizon, durable-goods monopoly with a continuum of buyers, uniform
# valuations on [0, 1], and zero seller cost. Two regimes are solved on the
# same (T, delta) grid:
#
#   1. Commitment.       The seller commits at t=1 to the entire price path
#                        {p_t}. Forward-looking buyers best-respond. For
#                        uniform F and c=0 the optimal commitment is the
#                        static monopoly price p* = 1/2 every period; the top
#                        half of valuations buys in period 1; nobody buys
#                        later. Profit = 1/4, independent of (T, delta).
#
#   2. No-commitment.    Markov-perfect equilibrium (skimming). State at t is
#                        the remaining-buyer cutoff v_t (valuations are
#                        uniform on [0, v_t], mass = v_t). The seller picks
#                        next-period cutoff w in [0, v_t]; the marginal
#                        buyer w is indifferent between accepting p_t now
#                        and waiting for p_{t+1}(w) at t+1. The Coase
#                        conjecture: as T -> infty with delta -> 1, the
#                        no-commitment price path collapses to marginal cost
#                        (here, zero).
#
# Scale-invariant backward induction. With uniform F on [0, v] and c=0, the
# value, equilibrium price, and equilibrium next-cutoff functions are
# homogeneous in v:
#
#   V_t(v) = beta_t * v^2,    p_t(v) = mu_t * v,    w_t^*(v) = lambda_t * v.
#
# Terminal period (t = T): single-shot monopoly on [0, v_T]. Optimal price
# v_T / 2, cutoff v_T / 2, value v_T^2 / 4. So
#     mu_T = 1/2,   lambda_T = 1/2,   beta_T = 1/4.
#
# Recursion (t = T-1 down to 1). The seller posts a price determined by the
# marginal-buyer indifference at the chosen cutoff:
#     p_t(w) = (1 - delta) w + delta * p_{t+1}(w) = A_{t+1} * w,
#     where  A_{t+1} = 1 - delta + delta * mu_{t+1}.
# Revenue plus discounted continuation as a function of cutoff w with state v:
#     R_t(w; v) = A_{t+1} w (v - w) + delta * beta_{t+1} w^2.
# FOC in w gives a quadratic with interior solution
#     lambda_t = A_{t+1} / (2 (A_{t+1} - delta * beta_{t+1})),
# clipped to [0, 1]. Then
#     mu_t   = lambda_t * A_{t+1},
#     beta_t = lambda_t (1 - lambda_t) A_{t+1} + delta * beta_{t+1} * lambda_t^2.
#
# Forward simulation from v_1 = 1 gives the equilibrium price path
# p_t = mu_t * v_t and cutoff trajectory v_{t+1} = lambda_t * v_t.
#
# Cross-check: solving the system at the stationary fixed point of these
# recursions (T -> infty) reproduces the closed-form stationary MPE.
#
# References: Coase 1972; Bulow 1982; Stokey 1981; Gul-Sonnenschein-Wilson 1986;
# Ausubel-Deneckere 1989; Ausubel-Cramton-Deneckere 2002.
#
# Outputs:
#   durable_goods_coase_price_paths.png  -- p_t vs t for representative (T, delta).
#   durable_goods_coase_collapse.png     -- p_T and p_1 vs T on log-T scale,
#                                            parameterized by delta.
#   durable_goods_coase_results.tex      -- commitment vs no-commitment value
#                                            and ratio table across (T, delta).

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import load_results, save_results, add_cache_args
from sims.plot_style import (apply_style, COLORS, BENCH_STYLE,
                             FIG_SINGLE, FIG_DOUBLE)

apply_style()

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'durable_goods_coase'

CONFIG = {
    # Buyer valuations uniform on [0, V_MAX]; cost c = 0.
    'V_MAX': 1.0,
    'COST':  0.0,
    # Horizon and discount sweeps.
    'T_GRID':     [2, 5, 10, 20, 50, 100, 200],
    'DELTA_GRID': [0.5, 0.75, 0.9, 0.95, 0.99],
    'version': 3,
}

# ---------------------------------------------------------------------------
# Scale-invariant DP for the no-commitment regime.
# ---------------------------------------------------------------------------

def solve_no_commitment_analytic(T, delta):
    """Backward induction for the no-commitment durable-goods monopolist
    using the scale-invariant representation V_t(v) = beta_t v^2,
    p_t(v) = mu_t v, w_t*(v) = lambda_t v.

    Returns
    -------
    dict with keys 'mu' (T,), 'lam' (T,), 'beta' (T,), 'price_path' (T,),
    'cutoff_path' (T+1,), 'value' (scalar V_1 at v_1 = 1).
    """
    mu  = np.zeros(T)
    lam = np.zeros(T)
    beta = np.zeros(T)

    # Terminal period (index T-1).
    mu[T - 1]   = 0.5
    lam[T - 1]  = 0.5
    beta[T - 1] = 0.25

    for t in range(T - 2, -1, -1):
        A = 1.0 - delta + delta * mu[t + 1]
        denom = 2.0 * (A - delta * beta[t + 1])
        if denom <= 1e-14:
            # Degenerate; the unconstrained FOC blows up. The seller's marginal
            # benefit of raising the cutoff is unbounded, so push cutoff to 0
            # (sell to everyone immediately) and price to 0.
            lam_t = 0.0
        else:
            lam_t = A / denom
        # Clip to [0, 1]; in this model lam is always in (0, 1) at an interior FOC.
        lam_t = float(np.clip(lam_t, 0.0, 1.0))

        mu_t   = lam_t * A
        beta_t = lam_t * (1.0 - lam_t) * A + delta * beta[t + 1] * lam_t**2

        mu[t]   = mu_t
        lam[t]  = lam_t
        beta[t] = beta_t

    # Forward sim from v_1 = 1.
    v = 1.0
    price_path = np.zeros(T)
    cutoff_path = np.zeros(T + 1)
    cutoff_path[0] = v
    for t in range(T):
        price_path[t] = mu[t] * v
        v = lam[t] * v
        cutoff_path[t + 1] = v

    return {
        'mu': mu, 'lam': lam, 'beta': beta,
        'price_path': price_path, 'cutoff_path': cutoff_path,
        'value': float(beta[0]),
    }


# ---------------------------------------------------------------------------
# Commitment benchmark.
# ---------------------------------------------------------------------------

def solve_commitment(T, delta, v_max=1.0):
    """Optimal pre-committed price path for the durable-goods monopolist.

    With forward-looking buyers and uniform F on [0, v_max] (c=0), the
    optimal commitment is the static-monopoly price v_max/2 posted forever.
    Buyers with v >= v_max/2 buy in period 1; nobody else buys (any later
    discount would induce strategic waiting and lower expected revenue). The
    seller earns p * (v_max - p) = v_max^2 / 4, independent of (T, delta).
    """
    p_star = v_max / 2.0
    value = p_star * (v_max - p_star)  # v_max^2 / 4
    price_path = np.full(T, p_star)
    cutoff_path = np.zeros(T + 1)
    cutoff_path[0] = v_max
    cutoff_path[1:] = p_star
    return {
        'value': value,
        'price_path': price_path,
        'cutoff_path': cutoff_path,
    }


# ---------------------------------------------------------------------------
# Stationary MPE (T -> infty) fixed-point solution. Used as a cross-check
# against the finite-T backward induction.
# ---------------------------------------------------------------------------

def solve_stationary_mpe(delta, n_iter=5000, tol=1e-12):
    """Fixed-point iteration on (lambda, mu, beta) for the stationary MPE.

    Self-consistent equations:
        mu     = lambda * A
        A      = 1 - delta + delta * mu
        beta   = lambda (1 - lambda) A + delta * beta * lambda^2
        lambda = A / (2 (A - delta * beta))
    """
    lam = 0.5
    for _ in range(n_iter):
        mu = lam * (1.0 - delta) / (1.0 - delta * lam)
        A = 1.0 - delta + delta * mu
        beta = lam * (1.0 - lam) * A / (1.0 - delta * lam**2)
        lam_new = A / (2.0 * (A - delta * beta))
        lam_new = float(np.clip(lam_new, 0.0, 1.0))
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = 0.5 * lam + 0.5 * lam_new
    return {'lambda': lam, 'mu': mu, 'beta': beta, 'A': A}


# ---------------------------------------------------------------------------
# Sweep over (T, delta).
# ---------------------------------------------------------------------------

def run_sweep(T_grid, delta_grid, v_max=1.0):
    """Solve commitment and no-commitment on the (T, delta) Cartesian grid."""
    results = {}
    for T in T_grid:
        for delta in delta_grid:
            t0 = time.time()
            nc = solve_no_commitment_analytic(T, delta)
            elapsed = time.time() - t0
            cm = solve_commitment(T, delta, v_max=v_max)
            # The scale-invariant solver runs on v_max = 1; rescale.
            scale = v_max
            value_nc = nc['value'] * scale**2
            price_path_nc = nc['price_path'] * scale
            cutoff_path_nc = nc['cutoff_path'] * scale
            results[(T, delta)] = {
                'no_commit_value': value_nc,
                'commit_value':    cm['value'],
                'no_commit_price_path':  price_path_nc.tolist(),
                'commit_price_path':     cm['price_path'].tolist(),
                'no_commit_cutoff_path': cutoff_path_nc.tolist(),
                'no_commit_p1':  float(price_path_nc[0]),
                'no_commit_pT':  float(price_path_nc[-1]),
                'no_commit_mu':  nc['mu'].tolist(),
                'no_commit_lam': nc['lam'].tolist(),
                'no_commit_beta': nc['beta'].tolist(),
                'elapsed_sec':   elapsed,
            }
            print(f"T={T:>4d} delta={delta:.2f}  "
                  f"V_nocom={value_nc:.4f}  V_com={cm['value']:.4f}  "
                  f"ratio={value_nc/cm['value']:.3f}  "
                  f"p_1={price_path_nc[0]:.4f}  "
                  f"p_T={price_path_nc[-1]:.4f}  "
                  f"({elapsed*1000:.1f} ms)")
    return results


# ---------------------------------------------------------------------------
# compute_data() and generate_outputs() per CLAUDE.md simulation standards.
# ---------------------------------------------------------------------------

def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    v_max = CONFIG['V_MAX']

    print("=" * 70)
    print("DURABLE GOODS MONOPOLY: COASE CONJECTURE VIA DP")
    print(f"Buyer valuations uniform on [0, {v_max}]; seller cost c = 0.")
    print("=" * 70)
    print(f"T grid: {CONFIG['T_GRID']}")
    print(f"delta grid: {CONFIG['DELTA_GRID']}")
    print()

    # Sanity check #1: T=1 (single shot). Seller charges static monopoly v_max/2,
    # value = v_max^2 / 4 = 0.25 (at v_max=1). No discounting matters.
    print("Sanity check #1: T=1, delta=0.9 (single-shot static monopoly).")
    s1 = solve_no_commitment_analytic(1, 0.9)
    print(f"  V = {s1['value']:.5f}  (target 0.25)")
    print(f"  p_1 = {s1['price_path'][0]:.5f}  (target 0.5)")
    print(f"  mu_T = {s1['mu'][-1]:.5f}, lambda_T = {s1['lam'][-1]:.5f}  (targets 0.5, 0.5)")
    print()

    # Sanity check #2: T=2, delta=0. With zero patience, period 2 is irrelevant
    # to buyers; the seller still treats each period's residual mass as a fresh
    # single-shot. Period 1 sells to top half at p=0.5, getting revenue 0.25;
    # period 2 starts with state 0.5, sells to top half at p=0.25, revenue 0.0625
    # but discounted to delta * 0.0625 = 0. Total: 0.25.
    print("Sanity check #2: T=2, delta=0 (zero patience).")
    s2 = solve_no_commitment_analytic(2, 0.0)
    print(f"  V = {s2['value']:.5f}  (target 0.25)")
    print(f"  p_1 = {s2['price_path'][0]:.5f}  (target 0.5)")
    print(f"  p_T = {s2['price_path'][-1]:.5f}  (target 0.25 = static monopoly on [0, 0.5])")
    print()

    # Sanity check #3: large-T no-commitment value approaches stationary MPE.
    print("Sanity check #3: T=200 finite-horizon vs stationary MPE.")
    print(f"  {'delta':>6} {'V_T=200':>10} {'V_stationary':>12} {'p_1_T=200':>11} {'p_1_stationary':>15}")
    for delta in CONFIG['DELTA_GRID']:
        nc = solve_no_commitment_analytic(200, delta)
        st = solve_stationary_mpe(delta)
        print(f"  {delta:>6.2f} {nc['value']:>10.5f} {st['beta']:>12.5f} "
              f"{nc['price_path'][0]:>11.5f} {st['mu']:>15.5f}")
    print()

    print("Main sweep (T, delta):")
    print("-" * 70)
    sweep_results = run_sweep(
        T_grid=CONFIG['T_GRID'],
        delta_grid=CONFIG['DELTA_GRID'],
        v_max=v_max,
    )

    # Stationary MPE values for the delta grid (reference).
    stationary = {}
    for delta in CONFIG['DELTA_GRID']:
        st = solve_stationary_mpe(delta)
        stationary[delta] = {
            'lambda': st['lambda'], 'mu': st['mu'], 'beta': st['beta'],
        }

    data = {
        'sweep': {f"{T}_{delta}": v for (T, delta), v in sweep_results.items()},
        'sweep_keys': list(sweep_results.keys()),
        'stationary': stationary,
        'sanity': {
            'T1_delta09_value': s1['value'], 'T1_delta09_p1': float(s1['price_path'][0]),
            'T2_delta00_value': s2['value'], 'T2_delta00_p1': float(s2['price_path'][0]),
            'T2_delta00_pT':    float(s2['price_path'][-1]),
        },
        'config': CONFIG,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def _get_sweep(data, T, delta):
    return data['sweep'][f"{T}_{delta}"]


def plot_price_paths(data, save_path):
    """Equilibrium no-commitment price paths p_t vs t for several (T, delta).
    Each subplot fixes delta and overlays T in {10, 50, 200}. Reference lines:
    commitment price p = 0.5; marginal cost p = 0.
    """
    deltas = data['config']['DELTA_GRID']
    Ts_to_show = [10, 50, 200]
    v_max = data['config']['V_MAX']

    fig, axes = plt.subplots(1, len(deltas), figsize=(3.4 * len(deltas), 4.0),
                             sharey=True)
    if len(deltas) == 1:
        axes = [axes]

    cmap = plt.get_cmap('viridis')
    n_T = len(Ts_to_show)

    for ax, delta in zip(axes, deltas):
        for idx, T in enumerate(Ts_to_show):
            s = _get_sweep(data, T, delta)
            prices = np.asarray(s['no_commit_price_path'])
            t_axis = (np.arange(T) + 1) / T
            col = cmap(0.15 + 0.7 * idx / max(1, n_T - 1))
            ax.plot(t_axis, prices, '-', color=col, linewidth=1.6,
                    label=f"$T = {T}$")
        # Commitment baseline at p = v_max / 2.
        ax.axhline(v_max / 2.0, **BENCH_STYLE)
        # Marginal cost reference.
        ax.axhline(0.0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
        ax.set_title(rf"$\delta = {delta}$", fontsize=11)
        ax.set_xlabel(r"$t / T$ (normalized time)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 0.55 * v_max)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Equilibrium price $p_t$")
    axes[-1].legend(fontsize=8, loc='upper right')
    axes[0].text(0.02, 0.51 * v_max, r"Commitment $p = 1/2$",
                 fontsize=8, color=COLORS['black'])
    axes[0].text(0.02, 0.02, r"$c = 0$ (marginal cost)",
                 fontsize=8, color=COLORS['gray'])

    fig.suptitle(
        "No-commitment price paths: collapse toward marginal cost grows with $T$ and $\\delta$",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_collapse(data, save_path):
    """Two-panel collapse plot: (a) final-period price p_T versus T on log-T
    scale, parameterized by delta; (b) opening price p_1 versus T. Both show
    Coase-style decay toward marginal cost as T grows and delta grows.
    """
    Ts = data['config']['T_GRID']
    deltas = data['config']['DELTA_GRID']
    v_max = data['config']['V_MAX']
    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    ax_pT, ax_p1 = axes

    for i_d, delta in enumerate(deltas):
        col = cmap(0.15 + 0.7 * i_d / max(1, len(deltas) - 1))
        pT_vec = np.array([_get_sweep(data, T, delta)['no_commit_pT'] for T in Ts])
        p1_vec = np.array([_get_sweep(data, T, delta)['no_commit_p1'] for T in Ts])
        ax_pT.plot(Ts, pT_vec, 'o-', color=col, linewidth=1.6,
                   label=rf"$\delta = {delta}$", markersize=5)
        ax_p1.plot(Ts, p1_vec, 'o-', color=col, linewidth=1.6,
                   label=rf"$\delta = {delta}$", markersize=5)

    for ax, ylabel, title in [
        (ax_pT, r"Terminal-period price $p_T$",
         r"Coase collapse: $p_T \to c = 0$ as $T$ grows"),
        (ax_p1, r"Opening price $p_1$",
         r"Opening-price erosion as $T$ grows"),
    ]:
        ax.set_xscale('log')
        ax.set_xlabel(r"Horizon $T$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.axhline(0.0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
        ax.axhline(v_max / 2.0, **BENCH_STYLE)
        ax.set_ylim(-0.02, 0.55 * v_max)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='upper right')

    ax_pT.text(Ts[0] * 1.05, 0.51 * v_max, r"Commitment $p = 1/2$",
               fontsize=8, color=COLORS['black'])
    ax_p1.text(Ts[0] * 1.05, 0.51 * v_max, r"Commitment $p = 1/2$",
               fontsize=8, color=COLORS['black'])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_results_table(data, save_path):
    """LaTeX table: commitment value, no-commitment value, ratio, p_1, p_T
    across (T, delta).
    """
    Ts = data['config']['T_GRID']
    deltas = data['config']['DELTA_GRID']

    lines = []
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"$T$ & $\delta$ & $V^{\mathrm{com}}$ & $V^{\mathrm{nc}}$ "
                 r"& $V^{\mathrm{nc}}/V^{\mathrm{com}}$ & $p_1$ & $p_T$ \\")
    lines.append(r"\midrule")
    for T in Ts:
        for delta in deltas:
            s = _get_sweep(data, T, delta)
            ratio = s['no_commit_value'] / s['commit_value']
            lines.append(
                f"{T} & {delta:.2f} & {s['commit_value']:.4f} & "
                f"{s['no_commit_value']:.4f} & {ratio:.3f} & "
                f"{s['no_commit_p1']:.4f} & {s['no_commit_pT']:.4f} \\\\"
            )
        if T != Ts[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {save_path}")


def generate_outputs(data):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_price_paths(data,  os.path.join(out_dir, 'durable_goods_coase_price_paths.png'))
    plot_collapse(data,     os.path.join(out_dir, 'durable_goods_coase_collapse.png'))
    generate_results_table(data, os.path.join(out_dir, 'durable_goods_coase_results.tex'))

    Ts = data['config']['T_GRID']
    deltas = data['config']['DELTA_GRID']

    print()
    print("=" * 78)
    print("RESULTS SUMMARY: COMMITMENT VS NO-COMMITMENT")
    print("=" * 78)
    print(f"{'T':>4} {'delta':>6}  {'V_com':>8} {'V_nc':>8} {'ratio':>7} "
          f"{'p_1':>8} {'p_T':>8}")
    print("-" * 78)
    for T in Ts:
        for delta in deltas:
            s = _get_sweep(data, T, delta)
            ratio = s['no_commit_value'] / s['commit_value']
            print(f"{T:>4d} {delta:>6.2f}  {s['commit_value']:>8.4f} "
                  f"{s['no_commit_value']:>8.4f} {ratio:>7.3f} "
                  f"{s['no_commit_p1']:>8.4f} {s['no_commit_pT']:>8.4f}")
        print("-" * 78)

    print()
    print(f"Coase asymptotic check at T = {max(Ts)}:")
    for delta in deltas:
        s = _get_sweep(data, max(Ts), delta)
        st = data['stationary'][delta]
        print(f"  delta = {delta:.2f}: p_T = {s['no_commit_pT']:.5f}, "
              f"p_1 = {s['no_commit_p1']:.5f}, "
              f"V_nc / V_com = {s['no_commit_value'] / s['commit_value']:.4f}  "
              f"|  stationary MPE p_1 = {st['mu']:.5f}, V = {st['beta']:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
