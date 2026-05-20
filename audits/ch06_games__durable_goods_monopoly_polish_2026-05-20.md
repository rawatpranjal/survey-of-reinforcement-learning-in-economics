# Polish Report: ch06_games / durable_goods_monopoly + durable_goods_coase

**Date:** 2026-05-20
**Phase 0 (audit):** 65%
**Phase 1 (fix, rescope to screening vs pooling):** 20%
**Phase 2 (recovery, new DP-based Coase sweep):** 10-15%
**Phase 3 (this polish):** ~10%
**Scope:** Prose-flow polish only. No re-run of either sim.

## What this pass did

Light prose-flow edits to `/Users/pranjal/Code/rl/ch06_games/tex/rl_in_games.tex` to remove redundancy between the Coase subsection's two forward pointers and the screening subsection's opening line, and to make the bidirectional cross-references between the two subsections terser and more accurate.

## Edits

Three small edits, all in `rl_in_games.tex`. No content changes; no claims added or removed.

1. **Line 155 (intro paragraph of the Coase subsection, forward pointer).** Trimmed the trailing "the Coase conjecture proper is delivered by the asymptotic sweep here" clause (which restates the subsection title) and rewrote the forward pointer to name the *method* used by the companion subsection rather than just its topic:
   - Before: "A separate two-period simulation in Section~\ref{subsec:durable_goods_screening} illustrates the screening-versus-pooling equilibrium structure that underlies the durable-goods game; the Coase conjecture proper is delivered by the asymptotic sweep here."
   - After: "Section~\ref{subsec:durable_goods_screening} below presents a companion two-period exercise that uses CFR on the extensive-form game tree to recover the screening-versus-pooling equilibrium structure underlying the durable-goods game."

2. **Line 204 (closing paragraph of the Coase subsection, end-of-section bridge).** Tightened. Made the bridge sentence describe what changes in the companion subsection (two periods, two-point types, CFR on the tree) so the reader knows exactly what to expect rather than reading "two-period precursor" twice:
   - Before: "Section~\ref{subsec:durable_goods_screening} below presents a two-period precursor that focuses on the screening-versus-pooling equilibrium structure."
   - After: "Section~\ref{subsec:durable_goods_screening} restricts the same model to two periods and a two-point type distribution, where it reduces to a screening-versus-pooling problem and CFR can solve the extensive-form game tree directly."

3. **Line 209 (opening paragraph of the screening subsection, backward pointer).** Replaced "serves as a methodological precursor to the dynamic programming sweep above" (which is not quite right — CFR on the extensive-form tree is not methodologically prior to backward induction on a continuum) with a backward pointer that frames the screening exercise as a restricted version of the Coase model, and adds a one-line caveat that the asymptotic price-collapse mechanism does not operate at $T = 2$:
   - Before: "...it is small enough to admit equilibrium computation via counterfactual regret minimization (CFR) and serves as a methodological precursor to the dynamic programming sweep above."
   - After: "...small enough to admit equilibrium computation via counterfactual regret minimization (CFR). [...] The horizon is fixed at $T = 2$ throughout this subsection: the asymptotic price-collapse mechanism documented in Section~\ref{subsec:coase} does not operate here."

After these edits, the bidirectional cross-references are clean:

- Subsection 1 (Coase, $\S$\ref{subsec:coase}) introduction (line 155) points forward to the screening subsection naming its method (CFR on the extensive-form tree).
- Subsection 1 (Coase, $\S$\ref{subsec:coase}) conclusion (line 204) points forward to the screening subsection naming what changes (two periods, two-point types).
- Subsection 2 (Screening, $\S$\ref{subsec:durable_goods_screening}) opening (line 209) points backward to the Coase subsection and disclaims the asymptotic Coase mechanism does not operate at $T = 2$.

## Residual issues from the recovery report

- **$\delta = 0.99$ finite-horizon premium at $T = 200$.** Already documented in the table footnote at line 200; no edit needed. The reader sees the value reach within $\sim 25\%$ of the asymptotic stationary level rather than coincide exactly, and the footnote attributes this correctly to the finite $T$ relative to the effective duration $T(1 - \delta) = 2$ at this $\delta$.
- **CFR NashConv of 4-24 in the screening sim.** Already reported honestly in tex at line 242 with the explicit utility-share anchor ($12\%$ of max payoff). No edit needed.
- **Orphan PNG files from the old screening sim** (`durable_goods_coase.png`, `durable_goods_delta_sweep.png`, `durable_goods_nashconv.png`, `durable_goods_strategies.png`) are still in `ch06_games/sims/`. They are not referenced by the chapter tex (verified via `grep`); the screening subsection includes only the results table, not the figures. The file name `durable_goods_coase.png` is now confusing because the new Coase sim's figures live at `durable_goods_coase_price_paths.png` and `durable_goods_coase_collapse.png`. Renaming the old screening-sim outputs would require touching `durable_goods_monopoly.py` and rerunning; explicitly out of scope per the polish brief. Not addressed.

## Recompile

```
cd docs && pdflatex -shell-escape -jobname=ch06_games "\def\chapterfile{../ch06_games/tex/rl_in_games}\input{compile_chapter}" \
  && bibtex ch06_games \
  && pdflatex -shell-escape -jobname=ch06_games "\def\chapterfile{../ch06_games/tex/rl_in_games}\input{compile_chapter}" \
  && pdflatex -shell-escape -jobname=ch06_games "\def\chapterfile{../ch06_games/tex/rl_in_games}\input{compile_chapter}"
```

Exit 0 on all three pdflatex passes and on bibtex. Output: `/Users/pranjal/Code/rl/docs/ch06_games.pdf` (18 pages, 1{,}060{,}926 bytes). No undefined citations and no undefined references in the final pass log (verified by `grep -E "Undefined|undefined" /tmp/ch06_pass3.log` returning empty).

## Bullshit detector axis check (delta from recovery)

The polish is prose-only. The seven axes (Algorithm Identity, MDP Fidelity, Data Integrity, Comparison Fairness, Theoretical Sanity, Information Leakage, Seed and Reproducibility) were assessed at 10-15% by Phase 2 with no code changes in this pass; that assessment carries forward unchanged. The polish reduces score only via the "prose internal consistency" component of axis 4 (Comparison Fairness, broadly construed to include framing/cross-reference consistency).

## Bullshit score

**Bullshit score: 10%** --- The two-subsection structure is now internally consistent: forward and backward pointers are in place, each is short, and neither is redundant with the other. Reviewer 2's remaining concern is the residual $\delta = 0.99$ finite-horizon premium ($V = 0.058$ at $T = 200$ vs stationary $V = 0.045$) which the table footnote already documents; the reviewer could also note that the screening sim's CFR NashConv of 4-24 is large in absolute terms, but the tex already anchors this at $12\%$ of max payoff in linear-scale prose, and the multi-seed SE columns expose the genuine near-threshold mixing. The Coase conjecture is demonstrated cleanly by the $(T, \delta)$ sweep; the screening exercise is now positioned as a complementary CFR illustration of equilibrium structure at $T = 2$, not as a Coase demonstration.
