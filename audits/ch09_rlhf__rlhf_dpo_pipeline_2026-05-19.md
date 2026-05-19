# Audit: ch09_rlhf/sims/rlhf_dpo_pipeline.py

**Date:** 2026-05-19
**Diagram-only:** yes (matplotlib FancyBboxPatch + annotate arrows; no Monte Carlo, no training, no preference data)
**Cited tex file(s):** `ch09_rlhf/tex/rlhf.tex` (Figure ref `fig:rlhf_dpo_pipeline`, included at L50 inside Section "The RLHF Pipeline and Direct Optimization")
**Cited paper PDFs read (against figure content):** `ouyang2022training.pdf` (header reads .md summary), `rafailov2023direct.pdf` (.md summary), `christiano:2017.pdf` (.md summary), `ziegler2019fine.pdf`, `stiennon2020learning.pdf` — all present in `ch09_rlhf/papers/`. Korbak2022 (Bayesian RLHF) also present and referenced by adjacent tex.

The script renders a two-row pipeline diagram. Top row (RLHF): `SFT π_ref → Reward Model r_φ → PPO Fine-tuning`, with a "Human Preferences" arrow into the reward model and a curved feedback loop above the PPO box. Bottom row (DPO): `SFT π_ref → [ghost reward model] → Direct Optimization L_DPO`, with a bypass arrow from SFT to DPO and a "Human Preferences" arrow into the DPO box. Between the rows is the implicit-reward formula `r(x,y) = β log(π_θ(y|x) / π_ref(y|x))`. No equations beyond that; no learning curves; no numeric results.

## 1. Algorithm Identity

Conceptual content present in the figure:

- RLHF row: SFT → Reward Model → PPO. This matches the InstructGPT three-stage pipeline (Ouyang 2022, §3.1) and the prior \citet{ziegler2019fine}, \citet{stiennon2020learning} construction. The reward model is shown as separate from the SFT policy and downstream of human preferences, consistent with BT MLE training.
- The PPO box has an inner annotation: `π_θ generates → r_φ scores → λ_KL penalty → update`. This is a correct narrative sketch of the RL fine-tuning loop in Eq. 1.37 of the chapter (the KL-regularised reward objective). The figure does NOT show the PPO clipped-ratio surrogate explicitly, but a single-box diagram is not the place for that; the chapter cross-references §actor-critic where PPO is defined. Acceptable for a pipeline overview.
- DPO row: SFT → Direct Optimization (ghost reward model). Correct depiction of Rafailov 2023's key claim: the reward model is reparameterized away. The ghost dashed box is a reasonable visual.
- Implicit reward formula between rows: `r(x,y) = β log(π_θ(y|x) / π_ref(y|x))`. Compare against Rafailov 2023 Eq. 5: `r(x,y) = β log(π_r(y|x)/π_ref(y|x)) + β log Z(x)`. The figure omits the `+ β log Z(x)` partition-function term. For a pipeline schematic this omission is conventional (the term cancels in pairwise preference loss anyway and is explicitly discussed in the tex at Eq. 1.38 and the cancellation argument). Defensible.

Notation slippage with the chapter:
- Figure uses **β** in the implicit reward formula. Chapter Eq. 1.36's footnote explicitly says: "$\lambda_{KL}$ denotes the KL penalty weight, reserving $\beta$ for model parameters and $\gamma$ for discount factors. The standard RLHF literature, including \citet{rafailov2023direct}, uses $\beta$ for this parameter." So the chapter chose `λ_KL` and the figure undoes that choice mid-figure. The PPO inner-loop annotation in the same figure uses `λ_KL`. So the diagram is internally inconsistent: top row says `λ_KL`, between-row formula says `β`. The reader has to do the translation that the chapter prose explicitly tried to avoid.
- Figure uses `π_θ` for the trained/optimized policy; chapter Eq. 1.38 and 1.39 use `π_φ` (chapter Eq. 1.38: `r(s,y) = λ_{KL} log(π*(y|s)/π^{SFT}(y|s)) + λ_{KL} log Z(s)`; Eq. 1.39 has the DPO loss on `π_φ`). Same problem: the figure undoes the chapter's notation choice.
- Figure uses `π_ref` for the reference; chapter uses `π^{SFT}` in the equations and `π_{ref}` informally. The figure's SFT boxes are labelled `π_ref` and that's consistent with itself, but the chapter equations are `π^{SFT}`. Minor.

These are presentation issues, not algorithm-identity violations. The conceptual content is correct.

## 2. Environment / MDP Fidelity

Not applicable. There is no environment, no MDP, no preference data in this script. It is a static schematic with hand-placed boxes and arrows.

The figure does NOT mislabel a method as having an MDP that it doesn't, so this is genuinely N/A rather than a covered-up problem.

## 3. Data Integrity

No data is computed. No table of numbers is produced. The only output is `rlhf_dpo_pipeline.png`. Nothing in the figure claims to be empirical, so there is no data-integrity violation to detect.

However: `rlhf_dpo_pipeline_stdout.txt` reads literally:

```
Saved: /Users/pranjal/Code/rl/ch08_rlhf/sims/rlhf_dpo_pipeline.png
```

The path `ch08_rlhf` is stale (chapter has been renamed to `ch09_rlhf`; see `tex/rlhf.tex` and the includegraphics path `../ch09_rlhf/sims/rlhf_dpo_pipeline.png`). The script header docstring also opens with "RLHF vs DPO pipeline comparison diagram for Chapter 8." So the script was written when this was Chapter 8 and the stdout was never regenerated after the rename. Not a Bullshit-Score-relevant error but visible repo drift. Per the project memory `feedback_update_stdout.md`, stdout should be regenerated after script changes.

## 4. Comparison Fairness

Not applicable in any quantitative sense (no comparison numbers are produced).

Visually, the diagram does treat RLHF and DPO symmetrically: both rows start from the same SFT box, both receive human preferences from a labelled arrow, both terminate at a final-optimization box of the same dimensions. The ghost reward model in the DPO row makes the bypass visible without distorting the layout. The "Human Preferences" label feeds into the reward model on the RLHF row and into the DPO loss directly on the DPO row, which is the correct contrast.

One subtle asymmetry: the RLHF row has a *curved red feedback loop* over the PPO box (suggesting iteration), while the DPO row has no such loop. This is accurate (DPO is a single supervised pass over the preference dataset; RLHF PPO is a sampling-update loop), so it is a fair visual representation of the two algorithms, not unfair colouring.

## 5. Theoretical Sanity Checks

There are no quantitative results. The qualitative content matches Rafailov 2023 Theorem 1 in that DPO recovers the same optimal policy class as KL-regularised RLHF under the BT model and infinite preference data. The figure does not over-claim this equivalence in any caption.

The figure caption (chapter L51) says: "Top row: the three-stage RLHF pipeline trains a reward model from human preferences, then uses PPO to fine-tune the policy with a KL penalty. Bottom row: DPO collapses the pipeline into a single supervised learning objective over preference pairs, eliminating the explicit reward model (ghosted box)." This is an accurate one-line summary of both methods.

## 6. Information Leakage

Not applicable. No learner, no eval, no reward function in the script.

## 7. Seed & Reproducibility

Not applicable. The figure is deterministic in matplotlib (no randomness). The script accepts `--data-only` (exits cleanly) and `--plots-only` (renders) flags per the project's diagram-only convention. The script is reproducible: running it twice produces the same PNG bit-for-bit modulo matplotlib build-info.

## Hostile-Reviewer Summary

Reviewer 2 picks up the figure, opens the chapter PDF, and looks at the figure-and-equations side by side. The figure says `β` for the KL weight; the chapter's footnote at Eq. 1.36 explicitly bargains with the reader for the right to call that thing `λ_KL` "reserving β for model parameters" and *then* the figure uses `β` anyway. The figure says `π_θ` for the optimized policy; chapter equations 1.38–1.39 use `π_φ`. None of this is wrong, but it is exactly the kind of presentation slip a hostile reviewer marks: "the authors cannot maintain a notation across one figure and the surrounding two pages."

The implicit-reward formula in the figure also drops the partition-function `+ β log Z(x)` term that appears in Rafailov 2023 Eq. 5 and in the chapter at Eq. 1.38 (`+ λ_KL log Z(s)`). For a pipeline schematic that's fine, since the term cancels in pairwise comparisons, but Reviewer 2 still mutters about it.

The stdout file logs a stale `ch08_rlhf` path; the script header still says "for Chapter 8." This is a self-inflicted credibility wound on what is otherwise an inoffensive diagram. Not a substance issue but does drift the audit upward by one bracket.

Diagram-only cap is 25%. The notation drift and the stale stdout/header path push to the cap. The diagram does not visually contradict the caption (the cap-removal trigger), so I hold the cap.

**Bullshit score: 25%** — Reviewer 2 catches that the figure uses `β` and `π_θ` while the surrounding equations use `λ_KL` and `π_φ`, plus a stale `ch08_rlhf` stdout/header path. Pipeline content itself is correct.
