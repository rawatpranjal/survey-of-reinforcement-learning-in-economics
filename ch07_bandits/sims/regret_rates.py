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
    rates = {
        r'$T$ (linear)':                T,
        r'$d\sqrt{T}$, $d=5$':         d * np.sqrt(T),
        r'$T^{2/3}$':                   T ** (2/3),
        r'$\sqrt{T}$':                  np.sqrt(T),
        r'$d\log T$, $d=5$':           d * np.log(T),
        r'$s_0 \log d \log T$, $s_0=5$': 5 * np.log(d) * np.log(T),
        r'$s_0 \log d \log T$, $s_0=1$': 1 * np.log(d) * np.log(T),
        r'$\log T$':                    np.log(T),
    }

    # Ordered from top to bottom (worst to best) for legend clarity
    order = [
        r'$T$ (linear)',
        r'$d\sqrt{T}$, $d=5$',
        r'$T^{2/3}$',
        r'$\sqrt{T}$',
        r'$d\log T$, $d=5$',
        r'$s_0 \log d \log T$, $s_0=5$',
        r'$s_0 \log d \log T$, $s_0=1$',
        r'$\log T$',
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
