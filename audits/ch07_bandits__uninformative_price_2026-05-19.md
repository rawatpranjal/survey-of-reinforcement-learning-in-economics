# Audit: ch07_bandits/sims/uninformative_price.py

**Date:** 2026-05-19
**Diagram-only:** YES (rubric cap 25% unless diagram contradicts caption)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch07_bandits/tex/dynamic_pricing.tex` (Section 1.2 "Parametric Demand", `\label{sec:broder}`, figure label `fig:uninformative_price`, lines 22-27)
**Cited paper PDFs read:** Broder & Rusmevichientong 2012 referenced in tex but `broder2012_dynamic_pricing_parametric.pdf` was not opened in this audit; the script's claim is sourced from the tex narrative, not from a re-derivation of Broder's Theorem 3.1.

## 1. Algorithm Identity

There is no algorithm. The script defines a single closed-form function `revenue(p, k) = max(r* - k*(p - p*)^2, 0)` (line 26-27) and plots four curves for `k ∈ {0.5, 1.0, 2.0, 3.5}`. No greedy MLE, no UCB, no Thompson sampling, no learner of any kind. The header declares the script "conceptual diagram" (line 2), and the tex figure is captioned as "Revenue curves ..." (line 25). Identity matches what is claimed.

Caveat: a hostile reviewer would note the diagram parameterizes directly in revenue space (quadratic peaks at $p^*$ for any $k$ by construction). It does not exhibit the underlying *demand-curve* family from Broder 2012 (their lower-bound family is linear in price: $d(p; z) = a(z) - b(z) p$ with curves crossing at $p^* = 1$). The fact that "all revenue curves agree at $p^*$" is forced by the parameterization $r(p) = r^* - k(p-p^*)^2$ — every member peaks at $(p^*, r^*)$ tautologically. The diagram thus illustrates the *consequence* of Broder's construction (revenue equality at the peak), not the *mechanism* (demand-curve identification failure). The tex caption is honest about this — it says "Revenue curves ... all models agree at $p^*$", consistent with the figure — but the prose at line 18 frames Broder's lower bound in terms of demand curves passing through the same point at $p^*(z_0)$, not revenue curves. A careful reader can connect the two; a fast skimmer will conflate them.

## 2. Environment / MDP Fidelity

There is no environment. The figure is a static pedagogical illustration. The functional form $r(p) = r^* - k(p - p^*)^2$ is a quadratic concave revenue curve with maximum $r^*$ at $p^*$, clipped at 0. There is no MDP, no transition, no noise, no buyer behavior, no time horizon $T$. The tex section discusses Kleinberg 2003 ($\Theta(\sqrt{T})$), Broder 2012 ($\Theta(\sqrt{T})$), and Javanmard 2019 ($O(s_0 \log d \cdot \log T)$) regret rates, none of which appear in this script. Consistent with diagram-only status.

Quadratic revenue curves are not Broder's family. Broder uses *linear* demand $d(p; z)$; the implied revenue $p \cdot d(p; z)$ is also a quadratic in $p$, but the peak location moves with $z$ unless the family is constructed so that all curves cross at a single price (the lower-bound family of Theorem 3.1). The script hardcodes all peaks at $(p^*, r^*)$ — equivalent to projecting Broder's family onto its lower-bound case, not the general case. The tex prose ("all demand curves pass through the same point at the optimal price $p^*(z_0)$") is the correct framing; the figure does the visual reduction.

## 3. Data Integrity

`compute_data()` does not exist (declared diagram-only on line 102: "No computation to cache"). `generate_outputs()` computes revenue values inline from the closed-form expression and prints them to stdout. Reported numbers (e.g., $r(4.3) = 24.755$ for $k=0.5$) are reproducible from $r^* - k(p_{lo} - p^*)^2 = 25 - 0.5(0.49) = 24.755$. Verified.

The stdout table also confirms by inspection that all four curves yield $r(5.0) = 25.0000$, the central claim of the figure. No hardcoded "expected" values masquerading as results.

## 4. Comparison Fairness

No comparison to make. The figure compares four members of one parametric family, all with the same $p^*$ and $r^*$, evaluated on the same price grid `np.linspace(1, 9, 500)`. There is no algorithm-vs-algorithm or oracle-vs-learner comparison.

## 5. Theoretical Sanity Checks

This is where a hostile reviewer would push hardest. The diagram is intended to motivate the *incomplete-learning failure* (Broder 2012 Theorem 3.1; den Boer & Zwart 2013; Harrison-Keskin-Zeevi 2012). The claim is: a greedy/certainty-equivalent seller posts $p^*(\hat{z}_t)$, observes only purchase behavior at that price, and because all candidate models predict identical demand at $p^*$, the estimator never gets enough Fisher information to distinguish them — the seller stalls.

What the figure shows:
- All four curves equal $r^* = 25$ at $p^* = 5$ (verified in stdout).
- Within the "exploration zone" $[4.3, 5.7]$, revenue separation is small but nonzero (stdout: separation grows from $0.255$ at $k=0.5$ to $1.715$ at $k=3.5$ at the exploration boundary).
- Outside the zone, curves diverge sharply.

What the figure does NOT show:
- No demand curves are plotted. The phrase "All demand models agree at $p^*$" is asserted via annotation, but the figure plots only revenue curves. A hostile reviewer would say: "Of course they agree — you parameterized them to share a peak. Show me the underlying demand curves and that observing purchase behavior at $p^*$ is uninformative."
- No simulation of greedy stalling. The diagram cannot demonstrate that a greedy MLE seller actually gets stuck; that would require a learner.
- No comparison to UCB/TS escape. The chapter narrative claims well-separated cases get $\log T$ regret; the diagram cannot show this either.

The diagram is essentially a textbook illustration, not a demonstration. This is OK for a conceptual figure if the caption and surrounding prose are precise — which they are (caption explicitly says "revenue curves", prose at line 18 explains the demand-curve mechanism separately). The diagram-only cap therefore holds.

One nitpick: the "exploration zone" half-width 0.7 is arbitrary. Stdout shows separations $\{0.255, 0.51, 1.02, 1.715\}$ at the zone boundary for $k \in \{0.5, 1.0, 2.0, 3.5\}$. Whether 1.7 revenue units of separation at the $k=3.5$ boundary is "nearly indistinguishable" (caption claim) depends on the noise scale, which is unspecified. A reviewer could legitimately ask: "Indistinguishable relative to what variance?" The figure has no error bars or noise model to anchor this.

## 6. Information Leakage

Not applicable. No agent, no observations, no policy. Trivially clean.

## 7. Seed & Reproducibility

No randomness — fully deterministic closed-form plot. The script seeds nothing (correctly, since nothing is random). The figure is reproducible bit-for-bit on any platform that runs matplotlib. Stdout disclosure of parameters at the top is adequate.

## Hostile-Reviewer Summary

The diagram does what the caption says — plots four revenue curves that share a peak at $(p^*, r^*)$ to illustrate why playing near $p^*$ provides little information about the demand parameter. It's a pedagogical figure, not a simulation, and its stdout values check out arithmetically.

The substantive complaint a hostile reviewer would lodge: the figure conflates two distinct visualizations. Broder's lower-bound construction is about *demand* curves crossing at the optimal price, which forces *revenue* curves to share the same peak value. The script visualizes only the latter and asserts the former via annotation text. The tex prose handles this carefully at line 18, but the figure caption (line 25) and the annotation arrow ("All demand models agree at $p^*$") elide the distinction. A reader who only looks at the figure could conclude that the construction is just "draw four quadratics that share a peak" — which would miss the point entirely (any parameterization $r(p) = r^* - k(p-p^*)^2$ tautologically shares a peak, regardless of the underlying demand structure).

A secondary nitpick: the "exploration zone" width is arbitrary and the caption's "nearly indistinguishable" claim is ungrounded without a noise scale. Stdout shows up to 1.7 revenue units of separation at the boundary for $k=3.5$, which may or may not be small depending on Bernoulli/Gaussian noise variance.

Neither complaint kills the diagram's pedagogical purpose. The substance survives. Caption-figure consistency is good enough that the rubric's hostile-revision threshold is not crossed.

**Bullshit score: 20%** — Diagram-only cap (25%) applies; the figure illustrates a consequence of Broder's construction (shared revenue peak) rather than the mechanism (demand-curve identification failure), and the annotation text "demand models agree at $p^*$" overloads a plot of revenue curves. Reviewer 2 writes a snarky comment about visualizing the wrong object, the prose at line 18 saves it, the figure stays.

Path: `/Users/pranjal/Code/rl/audits/ch07_bandits__uninformative_price_2026-05-19.md`
