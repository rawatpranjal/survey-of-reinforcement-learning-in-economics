# Uninformative optimal price: why p* reveals nothing about demand
# Chapter 7 (Bandits), conceptual diagram
# Shows that multiple demand models agree at p*, so playing p* cannot distinguish them.

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────
p_star = 5.0
r_star = 25.0
k_values = [0.5, 1.0, 2.0, 3.5]
p_grid = np.linspace(1, 9, 500)
exploration_half = 0.7  # exploration zone half-width

# ── Revenue curves ──────────────────────────────────────────────────────────
curve_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]


def revenue(p, k):
    return np.clip(r_star - k * (p - p_star) ** 2, 0, None)


def generate_outputs():
    # ── Stdout verification ────────────────────────────────────────────────────
    print("Uninformative Optimal Price Diagram")
    print("=" * 50)
    print(f"Optimal price p* = {p_star}")
    print(f"Optimal revenue r* = {r_star}")
    print()

    print("Revenue at p* for each k:")
    for k in k_values:
        r_at_star = revenue(p_star, k)
        print(f"  k = {k:.1f}:  r({p_star}) = {r_at_star:.4f}")

    print()
    print("Revenue separation at exploration boundaries:")
    p_lo = p_star - exploration_half
    p_hi = p_star + exploration_half
    print(f"  p_lo = {p_lo},  p_hi = {p_hi}")
    print(f"  {'k':>5s}  {'r(p_lo)':>10s}  {'r(p_hi)':>10s}")
    for k in k_values:
        r_lo = revenue(p_lo, k)
        r_hi = revenue(p_hi, k)
        print(f"  {k:5.1f}  {r_lo:10.4f}  {r_hi:10.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Exploration zone
    ax.axvspan(p_star - exploration_half, p_star + exploration_half,
               color=COLORS['gray'], alpha=0.15, label='Exploration zone')

    # Revenue curves
    for k, color in zip(k_values, curve_colors):
        r = revenue(p_grid, k)
        ax.plot(p_grid, r, color=color, label=f'$k = {k}$')

    # Vertical dashed line at p*
    ax.axvline(p_star, color=COLORS['gray'], linestyle='--', linewidth=1.0, alpha=0.7)

    # Dot at peak
    ax.plot(p_star, r_star, 'ko', markersize=6, zorder=5)

    # Annotation arrow
    ax.annotate(
        r'All demand models agree at $p^*$',
        xy=(p_star, r_star),
        xytext=(p_star + 1.8, r_star + 2.5),
        fontsize=10,
        ha='left',
        arrowprops=dict(arrowstyle='->', color=COLORS['black'], lw=1.2),
    )

    ax.set_xlabel(r'Price $p$')
    ax.set_ylabel(r'Revenue $r(p)$')
    ax.set_xlim(1, 9)
    ax.set_ylim(0, 30)
    ax.legend(loc='lower left', framealpha=0.9)

    out_path = os.path.join(os.path.dirname(__file__), 'uninformative_price.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print()
    print(f"Output file: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
