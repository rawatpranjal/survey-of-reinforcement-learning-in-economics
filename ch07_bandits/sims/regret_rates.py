# regret_rates.py
# Chapter 7: Dynamic Pricing
# Plot theoretical regret rate functions to companion Table 1 (Per 10K column).

import argparse
import sys
sys.path.insert(0, 'sims')

import numpy as np
import matplotlib.pyplot as plt
from plot_style import apply_style, COLORS

apply_style()


def generate_outputs():
    # T range and d
    T = np.logspace(2, np.log10(200_000), 500)
    d = 5

    # Regret rate functions (constants = 1)
    # Legend labels name the source paper for each rate so a reader skimming
    # the legend can map the curve to the chapter table without cross-reference.
    rates = {
        r'$T$ (linear, strategic-naive, Liu 2024)':       T,
        r'$d\sqrt{T}$, $d=5$ (corrected, Liu 2024)':      d * np.sqrt(T),
        r'$T^{2/3}$ (Lipschitz noise, Tullii 2024)':      T ** (2/3),
        r'$\sqrt{T}$ (Kleinberg 2003 / Broder 2012)':     np.sqrt(T),
        r'$d\log T$, $d=5$ (contextual, Xu 2021)':        d * np.log(T),
        r'$s_0 \log d \log T$, $s_0=5$ (Javanmard 2019)$^\dagger$': 5 * np.log(d) * np.log(T),
        r'$s_0 \log d \log T$, $s_0=1$ (Javanmard 2019)': 1 * np.log(d) * np.log(T),
        r'$\log T$ (well-sep., Broder 2012 / Misra 2019)': np.log(T),
    }

    # Ordered from top to bottom (worst to best) for legend clarity
    order = [
        r'$T$ (linear, strategic-naive, Liu 2024)',
        r'$d\sqrt{T}$, $d=5$ (corrected, Liu 2024)',
        r'$T^{2/3}$ (Lipschitz noise, Tullii 2024)',
        r'$\sqrt{T}$ (Kleinberg 2003 / Broder 2012)',
        r'$d\log T$, $d=5$ (contextual, Xu 2021)',
        r'$s_0 \log d \log T$, $s_0=5$ (Javanmard 2019)$^\dagger$',
        r'$s_0 \log d \log T$, $s_0=1$ (Javanmard 2019)',
        r'$\log T$ (well-sep., Broder 2012 / Misra 2019)',
    ]

    colors = [
        COLORS['red'],
        COLORS['orange'],
        COLORS['brown'],
        COLORS['blue'],
        COLORS['purple'],
        COLORS['cyan'],
        COLORS['olive'],
        COLORS['green'],
    ]

    linestyles = ['-', '-', '-', '-', '-', '--', '--', '-']

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, color, ls in zip(order, colors, linestyles):
        ax.plot(T, rates[label], label=label, color=color, linestyle=ls)

    # Vertical line at T = 10,000
    ax.axvline(x=10_000, color=COLORS['black'], linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(10_000 * 1.08, 2.5, 'Per 10K', fontsize=8,
            color=COLORS['black'], va='bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of customers $T$')
    ax.set_ylabel('Cumulative regret')
    ax.set_title('Theoretical regret rates (constants $= 1$, $d = 5$)')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

    # In-figure footnote explaining the visually counterintuitive ordering
    # of the s_0=5 sparse curve relative to d log T. At constants=1 the rate
    # s_0 log d log T exceeds d log T iff s_0 log d > d, which holds at
    # s_0=5, d=5 (log d ~ 1.61, so s_0 log d ~ 8.05 > d = 5). In typical
    # sparse-recovery regimes s_0 << d, so the sparse rate dominates.
    fig.text(
        0.5, -0.02,
        r'$^\dagger$ At $s_0 = d = 5$, $s_0 \log d \approx 8.0 > d = 5$, so the sparse '
        r'rate sits above $d \log T$. Sparse dominates when $s_0 \ll d$.',
        ha='center', va='top', fontsize=7, color=COLORS['black'])

    fig.tight_layout()
    fig.savefig('ch07_bandits/sims/regret_rates.png', dpi=300, bbox_inches='tight')
    print('Saved: ch07_bandits/sims/regret_rates.png')

    # Verify Per 10K values
    T0 = 10_000
    s0_vals = [1, 5]
    print('\nPer 10K (d=5) verification:')
    print(f'  sqrt(T)        = {np.sqrt(T0):.1f}   (table: ~100)')
    print(f'  log(T)         = {np.log(T0):.2f}   (table: ~9)')
    print(f'  T^(2/3)        = {T0**(2/3):.1f}  (table: ~464)')
    print(f'  d*log(T)       = {d*np.log(T0):.1f}   (table: ~46)')
    print(f'  d*sqrt(T)      = {d*np.sqrt(T0):.1f}  (table: ~500)')
    print(f'  T              = {T0}  (table: never improves)')
    print(f'  s0=1 log(d)*log(T) = {1*np.log(d)*np.log(T0):.1f}')
    print(f'  s0=5 log(d)*log(T) = {5*np.log(d)*np.log(T0):.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
