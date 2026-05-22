# spec.md — arXiv Survey Final Push (v3, audit-grounded)

**Date:** 2026-05-21
**Branch:** humanize-pass (68 ahead main)
**Scope:** Lock arxiv content → fix 30% sims → polish → submit. Thesis done. Journals deferred.
**Horizon:** Multi-session. Strict numbered order; each task resumable.

---

## Context

v1 used surface signals (line counts, file existence). v2 ran Explore subagents → big reversals. v3 read 30%-audit files end-to-end → concrete fix lists.

Reversals vs v1:
- ch10b RL-for-CI: all 5 content sections + sims + discussion drafted (376 lines, 25 bib keys, 2,347 LOC sims). Only chapter intro deferred.
- ch09 RLHF: DPO derived (lines 34-43), Bradley-Terry present, McFadden bridge present (line 73), GARP mentioned (line 66). Gaps: Constitutional AI / RLAIF / scalable oversight not named.
- ch12: all 10 sections drafted, compressed style. User wants consolidation.
- ch10/ch10b sims: all 6 audited 2026-05-20.
- GARP+RLHF lit: gap is real. 5 adjacent candidates surfaced.
- lqc_fvi 30%: 4 documented fixes.
- fishery 30%: 6 documented fixes.

**Overall arxiv: ~85%.**

---

## Status snapshot

| Cluster | Chapters | % |
|---|---|---|
| Strong (ship as-is) | ch01,02,03,04,05,06_macro,06_games,07,08,10,ch10b,11 | 85-95% |
| Light gaps | ch00 intro, ch03b, ch12, ch09 RLHF | 65-75% |
| Real gap | ch99 conclusion | 20% |
| 30% audit drift | lqc_fvi_fqi, fishery_paradigms | needs fix |

---

## Decisions (locked 2026-05-21)

1. **ch09 GARP:** Keep diagnostic mention (line 66). Add research-gap paragraph + 5 adjacent citations.
2. **ch12:** Consolidate 10 sections → 3-4 deeper.
3. **ch99 length:** 3pp tight.
4. **humanize-pass merge:** After content lock.
5. **Sim audit order:** Fishery first (6 fixes), lqc_fvi second.

---

## Master execution order

### Phase A — Content lock

| # | Task | File(s) | Detail | Est |
|---|------|---------|--------|---|
| A1 | ch99 conclusion expand | `ch99_conclusion/tex/conclusion.tex` | 3pp tight. Open problems + future directions. | 1 |
| A2 | ch10b chapter intro | `ch10b_rl_for_ci/tex/rl_for_ci.tex` (insert ~line 14) | High-level, no math, no roadmap. 3-5 paragraphs. | 0.5 |
| A3 | ch00 intro + abstract refresh | `ch00_introduction/tex/{intro,abstract,language}.tex` | Verify chapter list reflects ch10b/ch11/ch12. Abstract is one line; expand. | 1 |
| A4 | ch09 RLHF: name Constitutional AI, RLAIF, scalable oversight | `ch09_rlhf/tex/rlhf.tex` §5.3 | Bai 2022 (CAI), Lee 2023 (RLAIF), Bowman 2022 (scalable oversight). Add to refs.bib. | 0.5 |
| A5 | ch09 RLHF: GARP research-gap paragraph | `ch09_rlhf/tex/rlhf.tex` §5.3.2 (~line 64) | Existing diagnostic stays. Add gap framing. Cite Distortion of AI Alignment (2505.23749), RLHF→Direct (2601.06108), Optimistic Mirror Descent (2502.16852), COMAL (2410.23223), GPO (2402.05749). | 0.5 |

### Phase B — Sim audit fixes (audit-driven, concrete)

**B1. fishery_paradigms (30% → ≥75%).** Source: `audits/ch12_world_models__fishery_paradigms_2026-05-19.md`. Files: `ch12_world_models/sims/fishery_paradigms.py`, `ch12_world_models/sims/fishery_env.py`, `ch12_world_models/tex/s09_dual_sim.tex`.

| # | Fix |
|---|---|
| B1.1 | Rename "Model-Based LQ" → "Model-Based Joint LS" / "MB-DP". No LQ on logistic. |
| B1.2 | Add genuine open-access / myopic baseline (h = p/c). Show collapse tragedy. |
| B1.3 | Arifovic GA election operator: include stock-dynamics term. |
| B1.4 | Document or remove Q-Learning's action-grid prior `h_max = 1.5*r*K/4` (also in GA, MBPO). |
| B1.5 | Add parameter-recovery table (cobweb has one). |
| B1.6 | Report stock + harvest trajectories (figure currently shows only regret). |

Then: rerun → refresh `_stdout.txt` → recompile chapter PDF → re-audit. Est 1.5-2 sessions.

**B2. lqc_fvi_fqi (30% → ≥75%).** Source: `audits/ch03_theory__lqc_fvi_fqi_2026-05-19.md`. Files: `ch03_theory/sims/lqc_fvi_fqi.py`, `ch03_theory/tex/planning_learning_v3.tex`.

| # | Fix |
|---|---|
| B2.1 | DQN to ≥10 seeds + SEs (currently single seed). |
| B2.2 | Pick (a) honest tex framing (sim shows bias term only, noise-free regime), OR (b) inject Gaussian noise + show O(1/√N) shrinkage. Default (a). |
| B2.3 | Tex frame: comparison is illustrative (correct basis + known model vs deep approximator), not horse-race. |
| B2.4 | Reconcile tex line 170 ("converges in single projected iteration") with sim's 9 iterations. Drop or qualify. |

Then: rerun → refresh `_stdout.txt` → recompile chapter PDF → re-audit. Est 1-1.5 sessions.

### Phase C — Cross-cutting polish

| # | Task | File | Notes | Est |
|---|------|------|---|---|
| C1 | Seed-bump 5 sims to ≥10 seeds | carbon_constrained_production, robust_consumption_savings, benchmark_bus_engine, brock_mirman_bellman, trust_region_lqc | Means + SEs. | 1 |
| C2 | ch12 consolidate 10 → 3-4 sections | `ch12_world_models/tex/s0*.tex` + `world_models.tex` | Fold s01+s02 into chapter intro. Merge s04+s05+s07 (deep MBRL). Merge s06+s08 (objectives/convergence). Keep s03, s09, s10 standalone. | 1-2 |
| C3 | Refresh `_stdout.txt` for sims touched last 14d | various | Per `feedback_update_stdout`. | 0.5 |
| C4 | refs.bib final sweep | `docs/refs.bib` | After A4+A5 cites land. Verify zero orphans / hallucinations. | 0.5 |

### Phase D — Submit

| # | Task | File | Notes |
|---|------|------|---|
| D1 | humanize-pass → main | branch op | Merge-commit (provenance). After A+B+C. |
| D2 | Full compile + page count | `docs/main.tex` | Clean build. |
| D3 | `scripts/package_arxiv.sh` → tarball | `scripts/` | |
| D4 | Submit to arxiv | external | |

### Phase E — Post-arxiv (deferred)

| # | Task | Notes |
|---|------|---|
| E1 | Journals: csur_50pp draft | `~/.claude/plans/create-a-new-folder-abstract-ocean.md` |
| E2 | Journals: fntml_100pp draft | Same plan |
| E3 | Remaining 7 journal carve-outs | Same plan |
| E4 | "Economic Models for RL" chapter placement | Defer |
| E5 | Populate `papers/` dirs with reference PDFs | Low priority |

---

## Verification

- Phase A: per-chapter compile via `compile_chapter.tex`. Show PDF path.
- Phase B: rerun sim → refresh `_stdout.txt` → recompile chapter PDF → 7-pt audit → target ≥75%.
- Phase C: full build `cd docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex`.
- Phase D: arxiv tarball inspected, page count, refs.bib zero orphans.
