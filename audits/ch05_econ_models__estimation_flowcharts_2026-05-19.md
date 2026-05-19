# Audit: ch05_econ_models/sims/estimation_flowcharts.py

**Date:** 2026-05-19
**Diagram-only:** yes (cap 25%)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch05_econ_models/tex/rl_in_se.tex` (figure inserted at lines 17–22; caption attributes NFXP to Rust 1987).
**Cited paper PDFs read:** none read in full for this audit. Relevant references in `ch05_econ_models/papers/`: `AdusumilliEckardt2022_td_learning_ddc.pdf`, `HuYang2025_policy_gradient_ddc.pdf`, `Estimation_of_Dynamic_Discrete_Choice_Models_with_Differenti.pdf` (the AVI/TD and policy-gradient DDC papers cited in the surrounding section). No PDF of Rust 1987 in `papers/`, but its NFXP structure is canonical and unambiguous.

## 1. Algorithm Identity

The figure shows two stylized templates, not five named methods (NFXP / CCP / MPEC / FS / EE).

*Left panel — NFXP.*
- Outer dashed container labeled "Outer loop: MLE over $\theta$" — correct, NFXP's outer loop is BHHH/Newton on the log-likelihood.
- Inner solid container "Inner loop: Bellman equation" with box $V_{k+1} = T_\theta V_k$ and self-loop "VI iters" — correct. Rust 1987's NFXP repeatedly applies the contraction operator $T_\theta$ (or Newton-Kantorovich polishing near the fixed point) until $V$ converges, for each candidate $\theta$.
- Output $V^*(\theta) \to \mathcal{L}(\theta)$ — correct: the converged value function induces choice probabilities used in the likelihood.
- Complexity tag $\mathcal{O}(|\mathcal{S}|^2 \cdot N_{\mathrm{VI}})$ per $\theta$-evaluation — order-of-magnitude reasonable for the inner VI step on a tabular state space; a hostile reviewer could quibble that NFXP in Rust 1987 typically uses contraction iterations plus Newton polishing, and the per-iteration cost depends on the implementation of $T_\theta$. But as a stylized "per $\theta$-eval" annotation this passes.

*Right panel — RL-based single-loop.*
- "Single-loop stochastic approximation" with two coupled sub-boxes "Update $\theta$ (structural parameters)" and "Update $\omega$ (value/policy weights)", bidirectional arrows, output $(\hat\theta, \hat\omega)$, and "Two-timescale SA: $\mathcal{O}(1)$ per gradient step" — this is a faithful generic schematic of the family of methods reviewed in the section (Adusumilli–Eckardt TD-CCP, Hu–Yang policy-gradient SMM, etc.) where the inner Bellman fixed-point is replaced by stochastic updates on auxiliary weights $\omega$ run on a faster timescale than the structural-parameter updates on $\theta$.
- The "Two-timescale SA" label is the right name for this class (Borkar) and matches the canonical formulation those papers cite.

The figure deliberately collapses the heterogeneous RL-based methods into a single template — that is the diagram's whole point (Rust's nested loops versus a single loop). The caption is consistent with that scope ("NFXP versus RL-based structural estimation"). Reviewer 2 may grumble that CCP, MPEC, EE are not shown as separate flowcharts, but the caption and section context do not promise five-method coverage.

## 2. Environment / MDP Fidelity (N/A)

No specific MDP is depicted; the figure is method-template only.

## 3. Data Integrity (N/A)

No data computed; the figure is fully programmatic.

## 4. Comparison Fairness

Both panels share:
- same axis limits (`xlim=(-2.2, 2.8)`, `ylim=(-2.6, 3.2)`)
- same vertical placement of data input, body container, and output
- same level of granularity: one outer process + a body of containers/sub-boxes + a labeled output + a one-line complexity annotation

The NFXP side has two nested containers (outer MLE + inner Bellman) while the RL side has one container with two sub-boxes. This is the substantive difference the figure is built to communicate (nested vs single-loop), not an unfairness. The visual weight and font sizes are matched.

Minor cosmetic asymmetries:
- left panel has a self-loop "VI iters" with a small label; right panel's iteration is implied by the two-timescale label rather than a loop glyph. Defensible (the RL update is one step per data batch, not an inner iteration to convergence) but a hostile reviewer could ask for parallel self-loops on both sub-boxes.
- The "VI iters" label is placed slightly to the right of the Bellman rectangle and overlaps the right edge of the box at typical render sizes (visible in the rendered PNG). Cosmetic but real.

## 5. Theoretical Sanity Checks (N/A)

No numerical results.

## 6. Information Leakage (N/A)

No estimation.

## 7. Seed & Reproducibility (N/A)

Deterministic matplotlib drawing. No RNG calls in the script (only `import numpy as np`, used for `np.hypot`).

## Hostile-Reviewer Summary

The figure is a clean two-panel schematic that correctly contrasts Rust 1987's nested-fixed-point structure against the generic single-loop two-timescale stochastic-approximation template used by the recent papers reviewed in the section. NFXP's outer MLE / inner Bellman / output likelihood structure is faithfully drawn, and the single-loop side's bidirectional $\theta \leftrightarrow \omega$ coupling labelled "Two-timescale SA" is the right umbrella for what Adusumilli–Eckardt 2022 and Hu–Yang 2025 actually do. Caption and figure match.

Quibbles a hostile reviewer would raise but not die on:
- "VI iters" label overlaps the Bellman box edge at default render.
- Right panel has no explicit self-loop while left panel does; iteration symmetry is broken visually even if technically defensible.
- The complexity annotation on the left ($\mathcal{O}(|\mathcal{S}|^2 \cdot N_\mathrm{VI})$) is fine for tabular VI but glosses over Rust's Newton-Kantorovich polishing step. The right-panel "$\mathcal{O}(1)$ per gradient step" hides batch size and per-update cost. Both are stylized but defensible.
- Title "RL-Based Estimation (Single Loop)" is more aspirational than literal: some RL-based DDC estimators (e.g., the AVI variant in Adusumilli–Eckardt 2022) still iterate the regression to convergence, which would be a nested-ish loop. Generic enough that the caption survives.

The diagram does not visually contradict the caption; the cap holds. None of the quibbles invalidates the contrast the figure is making.

**Bullshit score: 10%** — Diagram-only cap 25% applies. Figure is faithful to NFXP and to the single-loop two-timescale family it claims to depict; reviewer 2 catches the "VI iters" label overlap and the asymmetric self-loop treatment, but the substance is sound.
