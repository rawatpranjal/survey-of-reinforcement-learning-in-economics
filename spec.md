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
| A4/5 bonus (done 2026-05-22) | ch09 §5.3 umbrella flattened; new §5.7 axiom-aware aggregation: math drill on Ge 2024 LCPO + sim `axiom_aware_aggregation.py` reproducing Theorems 3.1 + 4.3 on 6-candidate construction | `ch09_rlhf/tex/rlhf.tex` §5.7, `ch09_rlhf/sims/axiom_aware_aggregation.{py,png,tex}`, `audits/ch09_rlhf__axiom_aware_aggregation_2026-05-22.md` | Bullshit 15%. Not in original A–E list. | done |

### Phase B — Sim audit fixes (audit-driven, concrete)

**B1. fishery_paradigms — partially closed (re-audit 2026-05-22: 30%).** Source: `audits/ch12_world_models__fishery_paradigms_reaudit_2026-05-22.md`. Polish-pass landed B1.1, B1.2, B1.5 in code; B1.3 and B1.4 disclosed via footnote (not removed); **B1.6 NOT addressed**. `compute_paradigm` never stores `s_t`/`h_t`, so the trajectory figure was never produced. Re-audit by adversarial Opus caught the gap that polish-Sonnet missed.

| # | Fix | Status |
|---|---|---|
| B1.1 | Rename "Model-Based LQ" → "Model-Based DP" | LANDED |
| B1.2 | Open-access / myopic `h = p/c` baseline; collapse tragedy | LANDED |
| B1.3 | GA election operator stock-dynamics term | FOOTNOTED only |
| B1.4 | Q-Learning/GA/MBPO `h_max` prior documented or removed | FOOTNOTED only |
| B1.5 | Parameter-recovery table | LANDED |
| B1.6 | Stock + harvest trajectory figure | **MISSING — defer** |

**B1.6 next session:** instrument `compute_paradigm` to store seed-0 `s_curves` + `h_curves`; add second panel to `generate_outputs`. ~30-60 min. Cache invalidation for all paradigms.

**B2. lqc_fvi_fqi — CLOSED (re-audit 2026-05-22: 10%).** Source: `audits/ch03_theory__lqc_fvi_fqi_reaudit_2026-05-22.md`. All four polish items verifiably done.

| # | Fix | Status |
|---|---|---|
| B2.1 | DQN ≥10 seeds + SEs | LANDED (seeds 42..51, mean 0.7164 ± SE 0.0680) |
| B2.2 | Honest framing (sim = bias term only, noise-free regime) | LANDED (footnote `planning_learning_v3.tex:168`) |
| B2.3 | Comparison framed as illustrative, not horse-race | LANDED (body prose line 178 + table caption) |
| B2.4 | Reconcile line 170 with sim's 9 iterations | LANDED (operator-fact vs sim-init split) |

### Phase C — Cross-cutting polish

| # | Task | File | Notes | Est |
|---|------|------|---|---|
| C1 | Seed-bump 5 sims to ≥10 seeds | carbon_constrained_production (DONE 2026-05-22), robust_consumption_savings, benchmark_bus_engine, brock_mirman_bellman, trust_region_lqc | 4/5 already at ≥10. Carbon bumped 5→10. Tex updated for new λ peak 1.405±0.002, Lag return 178±4, emissions 28±4. | DONE |
| C2 | ch12 consolidate 10 → 6 sections | `ch12_world_models/tex/s0*.tex` + `world_models.tex` | DONE 2026-05-22. s01+s02→s01_intro; s04+s05+s07→s04_deep_mbrl; s06+s08→s06_objectives_convergence; s03/s09/s10 standalone. Zero label collisions. 7 originals backed up in tex/backups/2026-05-22-174705_*. ch12 PDF 41pp, mtime today. Two cross-chapter `section:rl_algorithms` warnings expected. | DONE |
| C3 | Refresh `_stdout.txt` for sims touched last 14d | robust_consumption_savings (DONE 2026-05-22, byte-identical, no drift) | Per `feedback_update_stdout`. | DONE |
| C4 | refs.bib orphan/missing-key sweep | `docs/refs.bib` | DONE 2026-05-22. 491 entries, 435 cited, 0 missing, 54 expected orphans (world-models queue + macro/games queue). Report `docs/plans/2026-05-22-refs-bib-sweep.md`. One key-year mismatch flagged: `Thoeni2026nmfg` body has `year={2025}`. | DONE |

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
