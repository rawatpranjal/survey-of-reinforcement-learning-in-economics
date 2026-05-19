# Audit fix: ch08_offline_rl/sims/offline_rl_pricing.py

**Date:** 2026-05-19
**Original score:** 50%
**Strategy:** Mixed — relabel mis-named algorithms (IQL → IQL-argmax, BCQ → BCQ-D), own the four-way `169.27` collapse in prose, populate `papers/`. NO substantive reimplementation of continuous BCQ, advantage-weighted IQL, or three-token DT.

## Summary of changes

### Code: `ch08_offline_rl/sims/offline_rl_pricing.py`

* Updated header comment to flag the three identity drifts (IQL-argmax policy step, BCQ-D variant, fused-token DT). Internal cache/registry keys remain bare `IQL` and `BCQ` so existing pickled results stay valid; published labels are applied at output time.
* Added `DISPLAY_NAMES = {'IQL': 'IQL-argmax', 'BCQ': 'BCQ-D'}` and a `_label(name)` helper.
* Routed the table writer, coverage figure legend, and stdout summary through `_label(...)` so every published artifact shows the qualified names. Tightened the stdout `Method` column width to fit `IQL-argmax`.
* No retraining required. Re-ran with `--plots-only`; all 15 cache files (`shared`, `DP_Oracle`, 7 main methods, 7 coverage sweeps) hit on first try.

### LaTeX: `ch08_offline_rl/tex/offline_rl.tex`

* IQL subsection: added footnote disclosing the argmax-over-Q policy extraction in place of advantage-weighted regression.
* BCQ subsection: added footnote distinguishing continuous BCQ (\citet{Fujimoto2019}, VAE + perturbation) from discrete BCQ-D (\citet{Fujimoto2019b}, threshold on $G_\omega(a|s)$). Stated that the table reports BCQ-D.
* Decision Transformer subsection: added footnote disclosing the fused-token simplification (one summed embedding per timestep instead of three separate tokens) and that the architecture coincides with the form used by \citet{Emmons2022}.
* Simulation introduction: corrected "four offline RL algorithms" to "seven offline learners" and listed which family each belongs to. Updated method names in the introduction to match the qualified labels.
* Results paragraph: renamed IQL → IQL-argmax and BCQ → BCQ-D inline. Removed the prior single-sentence BCQ-only acknowledgment.
* Added a new paragraph ("Four of the trained methods, BC, BCQ-D, DT, and RvS, all report $169.27 \pm 0.60$, identical to four decimal places...") explaining the four-way collapse: under behavioral mass concentrated 85% on $p=10$, BC, BCQ-D, DT, and RvS each reduce to the deterministic policy $\hat\pi(s) = 10$ for distinct mechanism-specific reasons (BC: cross-entropy mode; BCQ-D: threshold admits only the modal action; DT/RvS: high $R^\star$ is an extrapolation request that defaults to the modal training action). Cited \citet{Levine2020} and \citet{Fujimoto2019} for the general failure mode.
* Coverage paragraph + figure caption: renamed IQL, BCQ, and added explicit DT/RvS coverage behavior at $\epsilon_b = 0.9$.

### Bibliography: `docs/refs.bib`

* Added `@article{Fujimoto2019b, ...}` for "Benchmarking Batch Deep Reinforcement Learning Algorithms" (arXiv:1910.01708), which introduces BCQ-D. Verified no undefined citations in chapter compile.

### Papers: `ch08_offline_rl/papers/`

Previously empty. Now populated with seven PDFs verified by `mdls`:

| File | Pages | Source |
|---|---|---|
| `Ernst2005_FQI.pdf` | 54 | JMLR 6:503–556 |
| `Kumar2020_CQL.pdf` | 31 | arXiv:2006.04779 |
| `Kostrikov2022_IQL.pdf` | 13 | arXiv:2110.06169 |
| `Fujimoto2019_BCQ_continuous.pdf` | 23 | arXiv:1812.02900 (continuous BCQ, ICML 2019) |
| `Fujimoto2019b_BCQ_discrete_benchmark.pdf` | 13 | arXiv:1910.01708 (BCQ-D, benchmark paper) |
| `Chen2021_DT.pdf` | 21 | arXiv:2106.01345 |
| `Emmons2022_RvS.pdf` | 14 | arXiv:2112.10751 |

All seven cited papers in the chapter are now on disk for future audit verification.

## Verification

* `python3 ch08_offline_rl/sims/offline_rl_pricing.py --plots-only` exits 0. All 15 cache lookups hit. New stdout reports `IQL-argmax` and `BCQ-D` correctly.
* Re-rendered table `ch08_offline_rl/sims/offline_rl_pricing_results.tex` shows `IQL-argmax` and `BCQ-D` in the row labels; numerical values unchanged (no retraining).
* Coverage figure `ch08_offline_rl/sims/offline_rl_pricing_coverage.png` regenerated with the qualified legend labels.
* Chapter PDF compiles cleanly: `docs/ch08_offline_rl.pdf` (13 pages, 602,401 bytes). Three pdflatex passes + bibtex pass. Zero undefined citations. Pre-existing cross-chapter reference warnings (`section:rl_algorithms`) are unchanged.
* No code changes touched the training functions, so the audit's section-1 PASS findings (CQL identity, IQL expectile-V, MDP fidelity, no leakage, seed reproducibility) continue to hold.

## Residual issues not addressed

The mixed strategy explicitly excludes substantive reimplementation, so the following remain as-is:

* IQL-argmax's policy extraction is still argmax, not advantage-weighted regression. The footnote discloses this.
* BCQ-D is still the discrete variant; the continuous VAE+perturbation BCQ is not implemented. The footnote discloses this.
* DT is still fused-token; the strict three-token-per-timestep autoregressive form of \citet{Chen2021DT} is not implemented. The footnote discloses this.
* The $169.27 \pm 0.60$ four-way collapse is unchanged numerically. The new paragraph owns it as expected behavior of supervised-conditioning offline methods on concentrated behavioral data.
* DT/RvS sensitivity to the choice of $R^\star$ remains undisclosed in tex prose; only the choice itself ($R^\star = V^\ast(s_0) \approx 184$) is now mentioned in the four-way-collapse paragraph.

## New score

Under the hostile-reviewer rubric:

* The BCQ misattribution is no longer a misattribution: the chapter now names BCQ-D and cites \citet{Fujimoto2019b}, with a footnote stating which variant is implemented and why.
* The IQL policy-extraction simplification is now disclosed in a footnote and the method is renamed IQL-argmax in the table, figure, and prose.
* The DT fused-token form is now disclosed as a footnote, framed as the same architecture \citet{Emmons2022} use; the prose no longer claims it is the Chen2021 DT verbatim.
* The four-way `169.27` collapse is now explained in dedicated prose; a hostile reviewer can still ask "then why bother reporting four numerically identical rows?" but the answer is in the paragraph (the four mechanisms reduce to the same policy on concentrated behavioral data, and the experiment as designed cannot tell them apart). This is acknowledged rather than hidden.
* Residual mechanical drifts (IQL policy step, fused DT) remain but are footnoted; a reviewer who reads the footnotes will not write "the authors apparently did not understand the algorithm."

The remaining hostile-reviewer hooks are: (i) the experiment as designed cannot distinguish the four supervised-conditioning rows, which the new paragraph admits but does not fix; (ii) DT/RvS $R^\star$ sensitivity is undisclosed. Both are 25%-grade.

**Bullshit score: 25%** — Reviewer 2 catches that the experiment cannot tell the four collapsed rows apart and that the supervised-conditioning rows of the table carry no information beyond BC. The substance survives revision because the prose owns the collapse, the algorithm-identity drifts are disclosed in footnotes with corrected names, and the cited variant matches the implemented variant. Minor revise.
