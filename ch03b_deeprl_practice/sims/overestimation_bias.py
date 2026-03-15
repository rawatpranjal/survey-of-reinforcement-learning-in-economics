"""
Overestimation bias in Q-learning via Jensen's inequality.
Chapter 3b — Deep RL in Practice.
Shows that E[max Q_i] >= max E[Q_i] for noisy Q-value estimates.
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# ── Parameters ────────────────────────────────────────────────────────────────
mu = 2.0
sigma = 1.0
OUTPUT = os.path.join(os.path.dirname(__file__), 'overestimation_bias.png')


def generate_outputs():
    # ── Analytical results ────────────────────────────────────────────────────────
    # For n=2 iid N(mu, sigma^2), E[max] = mu + sigma / sqrt(pi)
    E_max_analytical = mu + sigma / np.sqrt(np.pi)

    # ── Define PDFs ───────────────────────────────────────────────────────────────
    x = np.linspace(mu - 4 * sigma, mu + 5 * sigma, 1000)

    pdf_individual = norm.pdf(x, mu, sigma)

    # Max of 2 iid normals: f_max(x) = 2 * phi(x) * Phi(x)
    pdf_max = 2 * norm.pdf(x, mu, sigma) * norm.cdf(x, mu, sigma)

    # ── Numerical verification ────────────────────────────────────────────────────
    def max_pdf(t):
        return 2 * norm.pdf(t, mu, sigma) * norm.cdf(t, mu, sigma)

    integral_check, _ = quad(max_pdf, mu - 10 * sigma, mu + 10 * sigma)

    def max_pdf_times_x(t):
        return t * max_pdf(t)

    E_max_numerical, _ = quad(max_pdf_times_x, mu - 10 * sigma, mu + 10 * sigma)

    print("=" * 60)
    print("Overestimation Bias: Jensen's Inequality in Q-Learning")
    print("=" * 60)
    print()
    print(f"Parameters: mu = {mu}, sigma = {sigma}")
    print(f"Number of actions (for plot): n = 2")
    print()
    print("Verification (n=2 iid normals):")
    print(f"  Integral of max PDF     = {integral_check:.10f}  (should be 1.0)")
    print(f"  E[max] analytical       = mu + sigma/sqrt(pi) = {E_max_analytical:.6f}")
    print(f"  E[max] numerical (quad) = {E_max_numerical:.6f}")
    print(f"  Overestimation bias     = {E_max_analytical - mu:.6f}")
    print()

    # ── Scaling table: E[max] - mu for n actions ──────────────────────────────────
    # For n iid N(mu, sigma^2), E[max] = mu + sigma * E[max of n standard normals]
    # We compute E[max of n iid N(0,1)] numerically.
    print("-" * 50)
    print("Overestimation bias scaling with number of actions")
    print("-" * 50)
    print(f"  mu = {mu}, sigma = {sigma}")
    print()
    print(f"  {'n':>5s}  {'E[max]-mu':>12s}  {'E[max]':>10s}")
    print(f"  {'-----':>5s}  {'------------':>12s}  {'----------':>10s}")

    n_values = [2, 5, 10, 20, 50, 100]
    for n in n_values:
        # CDF of max of n iid standard normals: Phi(z)^n
        # PDF: n * phi(z) * Phi(z)^{n-1}
        def max_n_pdf(z, n=n):
            return n * norm.pdf(z) * norm.cdf(z) ** (n - 1)

        def max_n_mean(z, n=n):
            return z * max_n_pdf(z, n)

        E_max_std, _ = quad(max_n_mean, -10, 10)
        E_max_n = mu + sigma * E_max_std
        bias_n = E_max_n - mu
        print(f"  {n:5d}  {bias_n:12.6f}  {E_max_n:10.6f}")

    print()

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Individual Q-hat distribution
    ax.plot(x, pdf_individual, color=COLORS['blue'], linewidth=2,
            label=r'Individual $\hat{Q}(s,a_i)$')
    ax.fill_between(x, pdf_individual, alpha=0.15, color=COLORS['blue'])

    # Max distribution
    ax.plot(x, pdf_max, color=COLORS['red'], linewidth=2,
            label=r'$\max_i \hat{Q}(s,a_i)$')
    ax.fill_between(x, pdf_max, alpha=0.25, color=COLORS['red'])

    # True value vertical line
    ymax_plot = max(pdf_individual.max(), pdf_max.max()) * 1.05
    ax.axvline(mu, color=COLORS['black'], linewidth=1.5, linestyle='-',
               zorder=5)
    ax.text(mu - 0.05, ymax_plot * 0.96, r'$\mathbb{E}[\hat{Q}_i] = \mu$',
            ha='right', fontsize=11, color=COLORS['black'],
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    # E[max] vertical line
    ax.axvline(E_max_analytical, color=COLORS['red'], linewidth=1.5,
               linestyle='--', zorder=5)
    ax.text(E_max_analytical + 0.05, ymax_plot * 0.80,
            r'$\mathbb{E}[\max_i \hat{Q}_i]$',
            ha='left', fontsize=11, color=COLORS['red'],
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    # Shaded bias gap between the two vertical lines
    ax.axvspan(mu, E_max_analytical, alpha=0.15, color=COLORS['red'],
               zorder=1, label='_nolegend_')

    # Bias annotation arrow
    bias_y = ymax_plot * 0.62
    ax.annotate('', xy=(E_max_analytical, bias_y), xytext=(mu, bias_y),
                arrowprops=dict(arrowstyle='<->', color=COLORS['red'],
                                lw=1.5))
    ax.text((mu + E_max_analytical) / 2, bias_y + 0.012,
            f'bias $= \\sigma/\\sqrt{{\\pi}} \\approx {E_max_analytical - mu:.3f}$',
            ha='center', fontsize=10, color=COLORS['red'])

    # Jensen's inequality text box
    textstr = r"$\mathbb{E}[\max_i \hat{Q}_i] \geq \max_i \mathbb{E}[\hat{Q}_i]$"
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor=COLORS['gray'], alpha=0.9)
    ax.text(0.97, 0.55, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_xlabel('Q-value')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(mu - 3.5 * sigma, mu + 4.5 * sigma)
    ax.set_ylim(0, ymax_plot)

    fig.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Output: {OUTPUT}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true',
                        help='No computation to cache (diagram-only script)')
    parser.add_argument('--plots-only', action='store_true',
                        help='Runs normally (same as no flags)')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
