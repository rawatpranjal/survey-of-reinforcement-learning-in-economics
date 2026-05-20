# Polish: ch07_bandits/sims/regret_rates.py

**Date:** 2026-05-20
**Prior audit:** `ch07_bandits__regret_rates_2026-05-19.md` (15%)
**Diagram-only:** YES. Cap at 25% applies; this polish targets ≤10%.
**Scope:** Three cosmetic fixes addressing the hostile-reviewer flags in §"Hostile-Reviewer Summary" of the prior audit. No algorithm change, no MDP change, no recomputation.

## Fixes applied

### 1. Legend now names source paper per curve
Reviewer flag: "$T$ (linear)" did not say where the linear regret comes from. Resolved by relabeling every curve to include the source paper name. The eight legend entries are now:

| Curve | New label |
|---|---|
| $T$ | $T$ (linear, strategic-naive, Liu 2024) |
| $d\sqrt{T}$ | $d\sqrt{T}$, $d=5$ (corrected, Liu 2024) |
| $T^{2/3}$ | $T^{2/3}$ (Lipschitz noise, Tullii 2024) |
| $\sqrt{T}$ | $\sqrt{T}$ (Kleinberg 2003 / Broder 2012) |
| $d\log T$ | $d\log T$, $d=5$ (contextual, Xu 2021) |
| $s_0\log d\log T$ ($s_0=5$) | $s_0 \log d \log T$, $s_0=5$ (Javanmard 2019)$^\dagger$ |
| $s_0\log d\log T$ ($s_0=1$) | $s_0 \log d \log T$, $s_0=1$ (Javanmard 2019) |
| $\log T$ | $\log T$ (well-sep., Broder 2012 / Misra 2019) |

BibTeX key `Liu2024strategic` verified in `docs/refs.bib:2344`. The legend labels match the citations used in `tab:regret_comparison`.

### 2. In-figure footnote explains the $s_0=5$ inversion
Reviewer flag: the $s_0=5$ sparse curve sits above $d \log T$ despite being advertised as "sparse," which is counterintuitive on first read.

Added a dagger marker ($^\dagger$) on the $s_0=5$ legend entry and a footnote below the axes:

> $^\dagger$ At $s_0 = d = 5$, $s_0 \log d \approx 8.0 > d = 5$, so the sparse rate sits above $d \log T$. Sparse dominates when $s_0 \ll d$.

The math justifies the ordering: $s_0 \log d \log T > d \log T$ iff $s_0 \log d > d$, i.e., $s_0 > d / \log d \approx 3.1$ at $d = 5$. The footnote tells a reader skimming the figure why $s_0=5$ does not look "fast," and reassures them that the typical sparse-recovery regime ($s_0 \ll d$) restores the expected ordering.

The figure caption in `dynamic_pricing.tex` was also updated to repeat this caveat for readers who never look at the figure footnote.

### 3. Tex prose sharpened on the constants-equal-to-1 caveat
Reviewer flag: setting all constants to 1 hides the finite-horizon ordering practitioners care about. The chapter prose at line 111 acknowledged the asymptotic point ("the gap between $\log T$ and $\sqrt{T}$ grows without bound, so the distinction is not just a constant factor") but did not flag the finite-horizon honesty issue. Tightened to:

> The figure makes a qualitative asymptotic point; in practice the implicit constants in front of $\log T$ can dwarf those in front of $\sqrt{T}$ at modest horizons, so the visual ordering near $T = 10{,}000$ is illustrative rather than predictive of finite-sample behavior.

This pre-empts the hostile reviewer's "your figure overstates the practical advantage of $\log T$ rates" complaint by saying it first.

## Verification

- Re-ran `python3 ch07_bandits/sims/regret_rates.py > ch07_bandits/sims/regret_rates_stdout.txt 2>&1`. Exit 0. Per-10K verification numbers unchanged (the rate functions were not edited).
- Inspected the regenerated PNG. Legend reads source paper per row, dagger marker is visible on the $s_0=5$ entry, footnote sits below the x-axis label without colliding with the legend.
- Recompiled `docs/ch07_bandits.pdf` (16 pages, 1,465,256 bytes). LaTeX run is clean modulo unrelated hyperref `Hfootnote` warnings present before the polish.

## What did NOT change

- The eight rate functions, the $T$ range, the $d = 5$ choice, the vertical line at $T = 10{,}000$, the color palette, and the figure size are unchanged.
- The stdout numbers reproduce bit-identically (deterministic plot).
- No new science. No new claim. Pure presentational cleanup.

## Audit checklist (all carried over from 2026-05-19 audit)

1. Algorithm identity: vacuous (no algorithms run). Legend now names papers explicitly. PASS.
2. Environment fidelity: $T \in [10^2, 2 \times 10^5]$, $d = 5$. Unchanged. PASS.
3. Data integrity: numbers reproduce, table per-10K column matches. PASS.
4. Comparison fairness: constants-equal-to-1 convention disclosed in title, caption, and now in the surrounding prose. PASS.
5. Theoretical sanity: log-log slopes match named rates; $s_0 = 5$ ordering disclosed as a function of the formula and the $s_0 \geq d$ regime via the footnote. PASS.
6. Information leakage: vacuous. PASS.
7. Seeds and reproducibility: deterministic. PASS.

## Hostile-reviewer pass

The three flagged complaints are gone:
- Legend labels source per curve (was: unattributed "$T$ (linear)").
- $s_0 = 5$ sparse-above-$d \log T$ inversion is annotated in the figure and the caption (was: silent).
- Constants-equal-to-1 caveat is acknowledged in the prose, not just the caption (was: prose claimed "not just a constant factor" without finite-horizon hedge).

Remaining nits the worst reader could raise: the figure is still asymptotic-only, the constants are still 1, the source-paper attribution in the legend uses author-year strings rather than `\cite{}` commands (PNGs cannot render bibtex). None of these threaten substance, and the prose now flags the finite-horizon caveat before the reader does.

**Bullshit score: 5%** — Reviewer 2 might quibble that the legend uses "Liu 2024" rather than "Liu et al. (2024)" or that the footnote is small print, but the substance, the math, the per-10K table cross-check, and the asymptotic ordering all hold, and the three prior flags are addressed in both the figure and the surrounding prose.
