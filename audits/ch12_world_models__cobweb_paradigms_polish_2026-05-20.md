# Polish pass: ch12_world_models/sims/cobweb_paradigms.py

**Date:** 2026-05-20
**Source audit:** `audits/ch12_world_models__cobweb_paradigms_2026-05-19.md` (25%)
**Goal:** address the two hostile-reviewer nicks from the source audit; target <=15%.

## Changes applied

### Nick 1 — "MBPO" oversold its implementation (Option A: rename)

**Source-audit framing.** The implementation is bootstrap-ensemble linear-Gaussian dynamics + a two-parameter Gaussian linear policy trained by REINFORCE with a moving-average baseline. Janner et al. 2019's MBPO uses dropout-disagreement neural dynamics, a SAC actor-critic with entropy regularization, and rollout-disagreement reweighting for composing model rollouts with real transitions. The label "MBPO" was doing more work than the implementation supports.

**Fix.** The display name and registry key were renamed from `'MBPO'` to `'MB-LG-REINFORCE'` (model-based linear-Gaussian REINFORCE). Concretely:

- `MBPOPolicy.name` attribute: `'MBPO'` -> `'MB-LG-REINFORCE'`. The class name `MBPOPolicy` is retained for cache-key stability and test compatibility; the file-header docstring, the class docstring, and the section banner all now use the new label and disclose the three departures from Janner 2019 (linear-Gaussian dynamics, two-parameter Gaussian policy, no dropout-disagreement weighting).
- `PARADIGM_REGISTRY`: key `'MBPO'` -> `'MB-LG-REINFORCE'`; configuration constant `MBPO_CONFIG` -> `MB_LG_REINFORCE_CONFIG`.
- `PARADIGM_ORDER`, `PARADIGM_COLORS`, the `make_paradigm` dispatch, the `_plot_param_recovery` paradigm list, the `_write_table_recovery` paradigm list, the `_print_summary` paradigm list, and the param-recovery figure suptitle all updated to use the new label.
- Stdout-print column widths widened from 13 to 17 characters so the 15-character label fits without overflow.

**Test update.** `sims/tests/test_mbpo_real.py::test_real_mbpo_name` was rewritten to assert `mbpo.name == 'MB-LG-REINFORCE'` and given a docstring explaining the rename rationale. The other three tests in the file (`test_parametric_lq_learner_exists`, `test_real_mbpo_has_learnable_policy_and_ensemble`, `test_real_mbpo_policy_moves_with_training`) pass unchanged because they check the class name `MBPOPolicy` (retained) and structural attributes (`K0`, `Kq`, `ensemble`), not the display label.

**Cache update.** The old cache file `cache/cobweb_paradigms__MBPO.pkl` was deleted because its config-key is no longer in the registry. The new file `cache/cobweb_paradigms__MB-LG-REINFORCE.pkl` was written by the re-run; numerical results are identical to the prior run because the internal RNG offset (`seed + 13579`) is unchanged.

**Tex propagation.** All MBPO references in the cobweb panel of `tex/s09_dual_sim.tex` that named the implemented method (lines 4, 15, 18, 20, 22, 24, 43, 70) were renamed to "MB-LG-REINFORCE" with the qualifier "in the spirit of \\citet{janner2019model}'s MBPO" on first mention. The only remaining "MBPO" occurrences in this section are inside the new disclosure footnote (Nick 1) and the broader chapter introduction footnote that explains the rename. Other chapter sections that discuss MBPO conceptually (`s03_dyna_q.tex`, `s07_mbpo_ensembles.tex`) were not touched because they describe the actual Janner 2019 algorithm in a literature-review context.

### Nick 2 — asymmetric structural prior between RLS and other learners (Option B: disclose)

**Source-audit framing.** Recursive least squares is given $(c, \phi)$ a priori; the two model-based learners must estimate all four of $(a, b, c, \phi)$. The conclusion "RLS wins regret because it has correct functional form and known cost parameters" entangles two factors. A reviewer would ask for a fourth panel where RLS does not get $(c, \phi)$ for free.

**Fix.** A new footnote was added to the Models paragraph of `tex/s09_dual_sim.tex` (attached to the recursive-least-squares description) that explicitly:

1. flags the asymmetric prior as deliberate but acknowledges it mixes two factors;
2. lists what each learner is given versus what it must estimate;
3. acknowledges that a fourth panel (RLS without known cost parameters) would disentangle the two and is left for follow-up work; and
4. argues why the qualitative inductive-bias-frontier ordering is robust to the asymmetry under the present parameters (the model-based LQ learner's transient cost-parameter information cost vanishes within the first hundred environment steps).

This is the cheaper Option B from the polish prompt; the more expensive Option A would require running the fourth panel.

## Files changed

- `/Users/pranjal/Code/rl/ch12_world_models/sims/cobweb_paradigms.py` (rename across header docstring, config constant, class banner/docstring, factory dispatch, registry, plotting code, table-writer code, stdout summary; widened stdout column widths from 13 to 17 chars)
- `/Users/pranjal/Code/rl/ch12_world_models/sims/tests/test_mbpo_real.py` (`test_real_mbpo_name` updated to new label, with docstring explaining the rename)
- `/Users/pranjal/Code/rl/ch12_world_models/tex/s09_dual_sim.tex` (introduction paragraph footnote disclosing the three MBPO departures; Models-paragraph footnote disclosing the asymmetric-prior confound; eight MBPO -> MB-LG-REINFORCE renames in chapter intro, Models, Results, recovery, policy-distance, verdict paragraphs, the recovery-figure caption, and the fishery-section's omission note)

## Re-run

```bash
python3 ch12_world_models/sims/cobweb_paradigms.py > ch12_world_models/sims/cobweb_paradigms_stdout.txt 2>&1
```

Re-run was needed because the cache key changed (registry key `'MBPO'` -> `'MB-LG-REINFORCE'`). Numerical results are identical to the prior cached run (RNG offset for MBPOPolicy unchanged at `seed + 13579`). Verified by spot-checking the regret table: RLS stable 5.89 +- 1.04, MB-LG-REINFORCE stable 656.60 +- 185.41, MB-LG-REINFORCE unstable 48.87 +- 3.47 -- all unchanged from the source-audit run. Param-recovery values also unchanged (0.000 +- 0.005, 0.005 +- 0.006, 0.000 +- 0.005 for the three regimes).

## Tests

Ran `pytest tests/test_mbpo_real.py tests/test_cobweb_ga_no_param_leak.py -v`. Seven tests passed; the updated `test_real_mbpo_name` confirms the new display label, the three structural MBPO tests confirm the class API is unchanged, and the three Arifovic-GA no-leak tests confirm the existing guardrail still fires.

## Compile

```bash
cd /Users/pranjal/Code/rl/docs && pdflatex -shell-escape -jobname=ch12_world_models "\\def\\chapterfile{../ch12_world_models/tex/world_models}\\input{compile_chapter}" && bibtex ch12_world_models && pdflatex -shell-escape -jobname=ch12_world_models "\\def\\chapterfile{../ch12_world_models/tex/world_models}\\input{compile_chapter}" && pdflatex -shell-escape -jobname=ch12_world_models "\\def\\chapterfile{../ch12_world_models/tex/world_models}\\input{compile_chapter}"
```

Compiled to `/Users/pranjal/Code/rl/docs/ch12_world_models.pdf` (2.57 MB). Only warnings are cosmetic hyperref `Hfootnote.N` destination-name warnings from the new footnotes; no undefined references and no missing citations.

## Reviewer reaction (revised)

Both source-audit nicks have been addressed at the artifact level. The MBPO label no longer overshoots the implementation -- it is renamed to MB-LG-REINFORCE everywhere a reader would see it (legend, table rows, stdout, figure suptitle), and a footnote on first mention enumerates the three specific departures from Janner 2019. The asymmetric-prior issue is now flagged in a footnote that names the confound, names the missing fourth panel, and argues why the qualitative frontier ordering survives. A reviewer can still mark the footnote as a deferral -- "the right fix is to run the fourth panel" -- but the prose no longer claims more than the experiment supports.

**Bullshit score: 12%** -- A determined Reviewer 2 still asks for the fourth-panel ablation that the footnote defers, but no longer has the MBPO-naming issue to write the snarky comment about. The qualitative inductive-bias-frontier story is intact and the artifact-level claims now match the implementation.
