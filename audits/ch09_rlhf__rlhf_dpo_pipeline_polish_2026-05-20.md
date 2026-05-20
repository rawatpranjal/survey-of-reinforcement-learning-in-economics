# Polish Report: ch09_rlhf/sims/rlhf_dpo_pipeline.py

**Date:** 2026-05-20
**Prior audit:** `ch09_rlhf__rlhf_dpo_pipeline_2026-05-19.md` (Bullshit score: 25%)
**Diagram-only:** yes (matplotlib FancyBboxPatch + annotate arrows; no Monte Carlo)

## Option Picked

**Option A.** Rewrote the figure's matplotlib text strings to match the surrounding tex notation, instead of patching tex with a translation footnote. The change was small (six string edits in two callsites), so the cleaner option was free.

## Fixes Applied

All changes in `ch09_rlhf/sims/rlhf_dpo_pipeline.py`. Tex was not touched.

| # | Where | Before | After | Reason |
|---|-------|--------|-------|--------|
| 1 | SFT box label (top row, L138) | `SFT Model $\pi_{\mathrm{ref}}$` | `SFT Model $\pi^{\mathrm{SFT}}$` | Match tex Eq. 1.34/1.36/1.38 reference-policy notation |
| 2 | SFT box label (bot row, L190) | `SFT Model $\pi_{\mathrm{ref}}$` | `SFT Model $\pi^{\mathrm{SFT}}$` | Same |
| 3 | Reward model box (L142) | `Reward Model $r_\varphi$` | `Reward Model $r_\theta$` | Match tex Eq. 1.34/1.37: $r_\theta$ throughout chapter |
| 4 | PPO inner annotation (L167–169) | `$\pi_\theta$ generates $\to r_\varphi$ scores $\to \lambda_{\mathrm{KL}}$ penalty $\to$ update` | `$\pi_\phi$ generates $\to r_\theta$ scores $\to \lambda_{\mathrm{KL}}$ penalty $\to$ update` | Match tex Eq. 1.37 ($J(\phi)$ optimizes $\pi_\phi$ against $r_\theta$ with $\lambda_{KL}$ KL weight). Inner annotation now internally consistent with the between-row formula |
| 5 | Implicit-reward formula (L225–227) | `$r(x,y) = \beta \log(\pi_\theta(y\mid x) / \pi_{\mathrm{ref}}(y\mid x))$` | `$r(s,y) = \lambda_{\mathrm{KL}} \log(\pi_\phi(y\mid s) / \pi^{\mathrm{SFT}}(y\mid s)) + \lambda_{\mathrm{KL}} \log Z(s)$` | Match tex Eq. 1.38 verbatim. Resolves three audit issues at once: (i) $\beta \to \lambda_{\mathrm{KL}}$, (ii) $\pi_\theta \to \pi_\phi$ and $\pi_{\mathrm{ref}} \to \pi^{\mathrm{SFT}}$, (iii) restores the dropped partition term $+ \lambda_{\mathrm{KL}} \log Z(s)$ (Rafailov 2023 Eq. 5, tex Eq. 1.38) |
| 6 | Script docstring header (L1) | `RLHF vs DPO pipeline comparison diagram for Chapter 8` | `RLHF vs DPO pipeline comparison diagram for Chapter 9` plus a notation key | Chapter rename drift |

The notation key added under the docstring states: reference policy `pi^{SFT}`, optimized policy `pi_phi`, reward model `r_theta`, KL weight `lambda_{KL}`, state `s`, output `y`. This future-proofs against another notation drift if the figure is edited in isolation.

## Verification

1. Re-ran the script:

   ```
   cd /Users/pranjal/Code/rl && python3 ch09_rlhf/sims/rlhf_dpo_pipeline.py > ch09_rlhf/sims/rlhf_dpo_pipeline_stdout.txt 2>&1
   ```

   Stdout is now `Saved: /Users/pranjal/Code/rl/ch09_rlhf/sims/rlhf_dpo_pipeline.png` (path no longer says `ch08_rlhf`).

2. Rendered figure inspected. Confirmed labels render as `π^SFT`, `r_θ`, `π_φ`, `λ_KL`, and the implicit-reward formula reads `r(s,y) = λ_KL log(π_φ(y|s) / π^SFT(y|s)) + λ_KL log Z(s)`. No LaTeX render glitches.

3. Recompiled chapter PDF:

   ```
   cd docs && pdflatex -shell-escape -jobname=ch09_rlhf "\def\chapterfile{../ch09_rlhf/tex/rlhf}\input{compile_chapter}" && bibtex ch09_rlhf && pdflatex -shell-escape -jobname=ch09_rlhf "..." && pdflatex -shell-escape -jobname=ch09_rlhf "..."
   ```

   Output: `/Users/pranjal/Code/rl/docs/ch09_rlhf.pdf` (16 pages, 890961 bytes). No LaTeX errors. Figure 1.36 (`fig:rlhf_dpo_pipeline`) now renders next to Eqs. 1.36–1.39 with matching notation.

## Residuals

None of the three nicks from the prior audit survive:

- Notation drift β vs λ_KL between figure and tex — **resolved** (figure now uses λ_KL everywhere).
- π_θ / π_ref vs π_φ / π^SFT — **resolved** (figure now uses π_φ and π^SFT).
- Stale `ch08_rlhf` in stdout/header — **resolved** (stdout regenerated, header updated to Chapter 9).
- Dropped partition term in implicit reward — **resolved** (`+ λ_KL log Z(s)` restored, matching tex Eq. 1.38).

One micro-residual: the script's inline comment reference `cf. ch09 identification_dags.py` still points to `ch09`, but that script actually now lives in `ch10_causal/` after the causal split (per `claude.md` 2026-05-12 note). Updated to `cf. ch10 identification_dags.py` in the docstring. Verified.

## Hostile-Reviewer Re-read

Reviewer 2 picks up the figure, opens the chapter PDF, and looks at the figure-and-equations side by side. The figure now uses the same symbols as the surrounding two pages: `π^SFT` for the reference, `π_φ` for the trained policy, `r_θ` for the reward model, `λ_KL` for the KL weight. The implicit-reward formula matches Eq. 1.38 character-for-character, including the partition-function term. The state variable is `s` (consistent with the rest of Section 1) rather than `x` (LLM convention). The chapter prose explicitly bargained for `λ_KL` notation in the Eq. 1.36 footnote, and the figure now honors that bargain.

The diagram is still a pipeline schematic — it does not draw the PPO clipped-ratio surrogate or the Bradley-Terry preference loss explicitly. Both are correctly cross-referenced in the prose. The ghost reward model in the DPO row remains the central visual contrast. The asymmetric feedback loop on PPO (curved red arrow) versus none on DPO accurately reflects the iterative-vs-supervised structural difference.

Nothing left to mark. The diagram is now flush with the tex.

**Bullshit score: 5%** — Reviewer 2 reads the figure and the equations side by side, finds nothing to mark, moves on. The 5% buffer reflects the inherent looseness of any pipeline schematic relative to the underlying math (PPO clipped surrogate not drawn, BT preference loss not drawn, deterministic-vs-stochastic distinction not visualized) — none of which can be fixed without making the diagram busy and none of which a reviewer would actually flag.

Diagram-only cap is 25%. New score 5% is well under the cap.
