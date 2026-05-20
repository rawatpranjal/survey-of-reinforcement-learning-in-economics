# Polish audit: ch05_econ_models/sims/estimation_flowcharts.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch05_econ_models__estimation_flowcharts_2026-05-19.md` (score 10%)
**Diagram-only:** yes (cap 25%)
**Cited tex file(s):** `ch05_econ_models/tex/rl_in_se.tex` (figure inserted at lines 17–22). No tex changes in this pass.
**Cited paper PDFs read for this polish pass:** none. The polish pass is purely cosmetic; algorithm-identity and template-fidelity claims established in the 2026-05-19 audit are unchanged.

## What changed

Two pure-cosmetic edits to `estimation_flowcharts.py`:

1. **"VI iters" label clearance.** The self-loop label on the left-panel Bellman box previously sat at `edge_x + 0.55`, which placed it close enough to the rounded-rectangle edge that the multi-line "VI / iters" overlapped the box outline at default render. Added a `label_offset` keyword to `draw_self_loop` (default `0.85`) and passed `label_offset=0.85` at the call site. Label now clears the box edge with visible whitespace and still sits inside the outer dashed container (right edge of container at `x = 2.0`; label at `x ≈ 1.80`).
2. **Right-panel self-loop for symmetry.** Added a `draw_self_loop(..., side='right', label='SA\nsteps', label_offset=0.85)` on the `Update ω` sub-box. This mirrors the left panel's VI loop and gives the fast-timescale stochastic-approximation step an explicit iteration glyph. The choice of the `ω` box (not the `θ` box) for the loop is principled: in the two-timescale-SA family being depicted (Borkar, Adusumilli–Eckardt 2022, Hu–Yang 2025), the value/policy-weight update is the fast loop run many steps per slow `θ` step. The label "SA steps" is generic enough to cover both the AVI variant (which iterates a regression toward a fixed point) and the policy-gradient variant (which runs many gradient steps on $\omega$ per outer SA step on $\theta$).

No changes to: data nodes, container boxes, complexity annotations, bidirectional coupling arrows, axis limits, titles, or captions.

## Verification

- Script ran without error. Stdout: `Output file: /Users/pranjal/Code/rl/ch05_econ_models/sims/estimation_flowcharts.png` + `Estimation flowcharts diagram generated.`
- Stdout file regenerated: `ch05_econ_models/sims/estimation_flowcharts_stdout.txt`.
- Rendered PNG inspected. Both fixes visible:
  - Left panel: "VI iters" label sits clearly to the right of the Bellman box outline, no overlap.
  - Right panel: matching self-loop arc on the `Update ω` sub-box with "SA steps" label, visually mirroring the left panel's loop and label placement.
- Chapter PDF recompiled (three pdflatex passes + bibtex). Output: `/Users/pranjal/Code/rl/docs/ch05_econ_models.pdf` (770 KB). Compile exit codes all 0.

## Hostile-reviewer revisit

- "VI iters" / Bellman-box overlap: fixed.
- Asymmetric self-loop (left panel had one, right did not): fixed; right panel now has a matching loop on the fast-timescale `ω` update.
- Remaining quibbles from the 2026-05-19 audit:
  - The complexity annotations are still stylized ($\mathcal{O}(|\mathcal{S}|^2 N_{\mathrm{VI}})$ on the left, $\mathcal{O}(1)$ per gradient step on the right). These are structural claims about the algorithms, not cosmetic, and remain defensible at this level of stylization. Out of scope for a cosmetic polish pass.
  - The right-panel title still says "Single Loop" even though the new `SA steps` self-loop shows the inner fast-timescale iteration explicitly. A pedantic reviewer could argue the figure now visually depicts a nested loop on the right too. Defensible: the "single loop" claim refers to the absence of an *outer* MLE wrapper, not the absence of any iteration; two-timescale SA is one unified update rule with two step-size sequences, drawn as one container with one self-loop. Caption and section text already frame it that way. No prose change needed.

The diagram does not visually contradict the caption. The two remaining quibbles flagged in the 2026-05-19 audit are both resolved; no new defects introduced.

**Bullshit score: 5%** — Diagram-only cap 25% applies. The two cosmetic nicks from the prior audit are fixed; reviewer 2 would have nothing left to write a snarky comment about beyond the stylized big-O annotations, which are an editorial choice and survive a cosmetic pass.
