# Audit: ch07_bandits/sims/regret_rates.py

**Date:** 2026-05-19
**Diagram-only:** YES. The script does no Monte Carlo, no algorithm runs, and no environment simulation. It plots seven closed-form regret rate functions of T on log-log axes and prints per-10K verification numbers to stdout. Under CLAUDE.md's diagram-only cap, the maximum bullshit score is 25% unless the picture visually contradicts its caption.
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch07_bandits/tex/dynamic_pricing.tex` (Section 2.6 "Comparison of Regret Rates", Table 1 `tab:regret_comparison` at lines 113-136, Figure `fig:regret_rates` at lines 138-143, prose paragraph at line 111).
**Cited paper PDFs read:** none re-read for this audit; cross-checked rate formulas against the chapter's own table and the chapter prose, which cite `kleinberg2003_demand_curve_value.pdf`, `broder2012_dynamic_pricing_parametric.pdf`, `javanmard2019_dynamic_pricing_high_dim.pdf`, `Xu2021_logarithmic_regret_dynamic_pricing.pdf`, `tullii2024_contextual_dynamic_pricing.pdf`, `Misra-DynamicOnlinePricing-2019.pdf`, `liu2024_strategic_buyers_pricing.pdf`. All present in `ch07_bandits/papers/`.

## 1. Algorithm Identity

Not applicable in the usual sense -- no algorithms are run. The "objects" being plotted are seven analytical regret-rate functions:

- $T$ (linear, strategic-naive)
- $d\sqrt{T}$ with $d=5$ (Liu2024strategic corrected)
- $T^{2/3}$ (Tullii2024, Lipschitz noise)
- $\sqrt{T}$ (Kleinberg2003 nonparametric, Broder2012 parametric)
- $d\log T$ with $d=5$ (Xu2021 contextual, known noise)
- $s_0\log d \log T$ with $s_0=5$ and $s_0=1$ (Javanmard2019 sparse)
- $\log T$ (Broder2012 well-separated, Misra2019 WARP)

These are the same seven entries the table claims, with the strategic-uncorrected case mapped to "$T$ (linear)". The functional forms match the analytical rates as stated in the chapter and as commonly cited from the named papers (problem-dependent vs problem-independent bounds in the Lai-Robbins / Auer tradition).

One narrow nit: the legend label "$T$ (linear)" is the naive Liu2024strategic rate. A hostile reviewer would prefer the legend to name the source of the linear regret (strategic-naive) rather than only its functional form, because the same $\Theta(T)$ rate is shared by other failure modes mentioned in the chapter (reference effects, fairness-violating policies). Not a math error, a labeling weakness.

## 2. Environment / MDP Fidelity

No environment. The script's "world" is $T \in [10^2, 2\times 10^5]$ on 500 log-spaced points and a fixed $d=5$. The horizon range covers the table's headline anchor $T = 10{,}000$ at the center of the plot. The vertical line at $T = 10{,}000$ exactly matches the "Per 10K" table column.

The choice $d = 5$ is a hardcoded constant that propagates to three curves ($d\sqrt T$, $d\log T$, $\log d$ in the sparse rate). It matches what the table caption announces ("Per 10K ($d=5$)") so the figure is consistent with the table, but the chapter never justifies $d = 5$ as anything other than a small illustrative value. A hostile reviewer would ask why not $d = 10$ or $d = 50$, and the honest answer is "this is illustrative." That's acceptable for a comparison plot but should be flagged in the caption (it is, line 141: "constants equal to 1, $d = 5$").

## 3. Data Integrity

There is no Monte Carlo so there is nothing to cache. `compute_data` is replaced by `generate_outputs`, which is consistent with the diagram-only convention in CLAUDE.md (caching skipped, `--data-only` exits with a message, `--plots-only` runs the plot path). Numbers printed to `regret_rates_stdout.txt` match what I recompute independently:

| Rate | Script print | Table claim | Independent check |
|---|---|---|---|
| $\sqrt{T}$ | 100.0 | $\sim$100 lost | 100.0 |
| $\log T$ | 9.21 | $\sim$9 lost | 9.21 |
| $T^{2/3}$ | 464.2 | $\sim$464 lost | 464.16 |
| $d\log T$ | 46.1 | $\sim$46 lost | 46.05 |
| $d\sqrt{T}$ | 500.0 | $\sim$500 lost | 500.0 |
| $s_0 = 1$ sparse | 14.8 | (caption: $\approx 15$) | 14.82 |
| $s_0 = 5$ sparse | 74.1 | (caption: $\approx 74$) | 74.12 |
| $T$ linear | 10000 | "never improves" | 10000 |

Every printed number reproduces. Captions also match.

## 4. Comparison Fairness

Not applicable. There is nothing to compare horizon-by-horizon. Constants are uniformly set to 1 as the title and caption disclose, which is the standard convention for visualizing asymptotic rates. A hostile reviewer might note that setting all constants to 1 is precisely what hides the genuinely interesting cross-over behavior between, say, $\sqrt{T}$ and $\log T$ at finite horizons, since the implicit constant in front of $\log T$ is typically much larger than that in front of $\sqrt{T}$ in actual pricing problems. The chapter is honest about this: the caption explicitly says "constants $=1$" and the prose at line 111 frames the conclusion correctly ("the gap between $\log T$ and $\sqrt{T}$ grows without bound, so the distinction is not just a constant factor"). The plot makes a qualitative point, not a quantitative one, and the prose treats it that way.

## 5. Theoretical Sanity Checks

This is the heart of the audit because the figure's purpose is to show asymptotic ordering. Slopes on log-log axes correspond to power-law exponents; logarithmic curves bend toward horizontal.

Visual / analytical predictions (slope = $d \log y / d \log T$):
- $T$ linear: slope 1
- $T^{2/3}$: slope 2/3
- $\sqrt{T}$ and $d\sqrt{T}$: slope 1/2 (parallel)
- $d\log T$, $s_0 \log d \log T$, $\log T$: slope $\to 0$ asymptotically; on log-log axes they show as concave-down curves bending toward horizontal as $T$ grows

The script plots exactly these. The ordering at $T = 10^5$ from steepest to flattest is: $T$, $T^{2/3}$, $d\sqrt{T}$, $\sqrt{T}$, $d\log T$, $s_0 = 5$ sparse, $s_0 = 1$ sparse, $\log T$. This is the same order the legend uses (top to bottom). At $T = 10{,}000$ the per-10K values are correctly computed and ordered.

Two minor reviewer flags:
- The $s_0 = 5$ curve (label $s_0 \log d \log T$ with $s_0 = 5$) plots at $5 \log d \log T = 5 \times 1.609 \times \log T \approx 8.05 \log T$, which is steeper at every $T$ than $d \log T = 5 \log T$. That ordering -- where the "sparse" rate is worse than the contextual rate at the chosen constants -- is correct given the formulas, but visually the sparse-$s_0=5$ curve sits above $d\log T$ in the plot, which a reader skimming the legend (sparse rates are typically advertised as faster) might find confusing. A hostile reviewer would note that the formula $s_0 \log d \log T$ becomes faster than $d \log T$ only when $s_0 \log d < d$, i.e., $s_0 < d / \log d \approx 3.1$ at $d=5$. The script does plot both $s_0 = 1$ (which IS faster than $d\log T$) and $s_0 = 5$ (which is NOT), and the caption discloses both values. Reader-confusion risk, not a math error.
- The $\log T$ baseline is the analytical Lai-Robbins lower-bound rate. At the chosen constants ($C=1$) it sits below every other curve. That is the qualitative point of the figure and it lands.

Theoretical rate predictions match plotted rates. No slope deception.

## 6. Information Leakage

Not applicable -- no agents, no rewards, no learning. Vacuously pass.

## 7. Seed & Reproducibility

No seeds needed (deterministic plot). The script is deterministic given numpy and matplotlib versions. Independent re-execution would produce a bit-identical PNG (modulo font rendering). No standard errors to report. Diagram caching is correctly skipped. The argparse flags `--data-only` (exits with message) and `--plots-only` (runs normally) follow the project convention for diagram-only scripts in CLAUDE.md.

## Hostile-Reviewer Summary

The figure is a closed-form rate plot that supports a table -- not a simulation. Every printed number reproduces, the per-10K column matches the table to displayed precision, the log-log slopes match the named asymptotic rates, and the caption discloses the constants-equal-to-1 convention. The figure is doing exactly what it claims.

Reviewer 2's real complaints are presentational, not substantive:
1. The "$T$ (linear)" legend entry doesn't name its source (strategic-naive Liu2024).
2. The $s_0 = 5$ sparse curve sits above $d \log T$ at the chosen constants, which is correct from the formula but visually counterintuitive for readers who expect "sparse = faster."
3. Setting all constants to 1 erases finite-horizon ordering that practitioners actually care about (the $\log T$ constant in real pricing problems can dwarf the $\sqrt{T}$ constant). The chapter addresses this in prose, but a hostile reviewer would still note that the figure as drawn is a stylized asymptotic claim, not a finite-horizon comparison.

None of these threaten the substance. The picture is honest about being asymptotic. Diagram-only cap applies; no visual contradiction with the caption.

**Bullshit score: 15%** -- A hostile reviewer writes one snippy comment about the $s_0 = 5$ curve ordering and the legend label for the linear-regret strategic case, but the math, the per-10K numbers, the log-log slopes, and the caption all hold. Diagram-only cap of 25% is not reached.
