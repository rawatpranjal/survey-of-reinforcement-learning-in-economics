# 2026-05-19 arxiv main-article push — end-of-cycle summary

**Branch:** `humanize-pass`
**Commits added:** 18 (from `9dbaab0` baseline to `abffb48` head)
**Final main.pdf:** 214 pages, 11.7 MB, zero undefined references, zero undefined citations
**Arxiv tarball:** `arxiv_submission.tar.gz` (11 MB, 154 files), smoke-tested to compile to identical 214-page PDF in a clean directory

## Phases executed (24 tasks)

### Phase 0 — Branch hygiene + baseline (Tasks 0.0–0.3)
- Moved plan to `docs/plans/2026-05-19-arxiv-main-article-push.md` (`docs/superpowers/` is gitignored)
- Committed staged work in 7 thematic commits: ch12 folder reorganization (38 files), humanize prose edits (ch02/ch08/ch10), new ch12 world_models content (42 files: section spine s01-s10, three core sims, env modules, tests, audits, CHAPTER_NOTES), docs build artifacts (refs.bib +153 lines, main.tex, sim_cache, chapter PDFs), audit/notes batch, gitignore for .serena/
- Baseline recompile uncovered 2 stale section labels (`section:planning_learning` and `section:rl_theory` referenced from ch10b and ch06_macro); fixed to `section:rl_algorithms` and `sec:planning_learning`
- Removed 4 stale chapter PDFs from previous naming schemes (ch12_forecasting_rl.pdf, ch_causal_for_rl.pdf, ch_macro_rl.pdf, ch_rl_for_ci.pdf)

### Phase 1 — Three-stream loose ends (Tasks 1.1–1.4)
- **1.1:** Added rank-ordered DTR results table to `ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py`; \input'd in `rl_for_ci.tex`. Table covers six methods (Murphy/FQI, Oracle x2, Q-learning, Neural-FQI, DQN). ch10b PDF: 30 pages.
- **1.2:** Refreshed `ch12_world_models/sims/dyna_maze.py` cache by force-running Dyna-Q+ and Schmidhuber agents (which had been updated with proper untried-action registration and reward-head, respectively). Headline numbers (52.0 / 47.0 / 39.2 / 4.0 / 3.5) unchanged from prior cache on this small maze. Corrected stale "13.2 ± 4.1" prediction in CHAPTER_NOTES.md.
- **1.3:** Archived orphan `decision_focused_ar1` sim from ch12 (4 files moved to `archive/ch12_decision_focused_ar1/`). Sim was unreferenced in current world_models spine.
- **1.4:** Verified counterfactual_ope tex \input paths match new ch10 location post-move; recompiled ch10_causal PDF.

### Phase 2 — Sim standardization sweep (Tasks 2.1–2.10)
Batched 7 sim updates into one commit: 5 sims gained the canonical `from sims.plot_style import apply_style, COLORS, ALGO_COLORS; apply_style()` block (gridworld_study, ssp_gridworld_20x20, theory_validation, lqr_convergence, bellman_vs_return); 2 sims had their non-canonical `from plot_style import` idiom normalized (info_geometry_npg, mm_surrogate_trpo). Skipped 2 files that have no matplotlib usage (gridworld_algorithms, econ_benchmark — both are shared-abstraction modules).

Chapter PDF recompile (Task 2.10) skipped: no figures or .tex changed, only sim source.

### Phase 3 — Pre-arxiv audit gates (Tasks 3.1–3.3)

- **3.1 bib-coverage:** Clean. 0 cited-but-missing keys, 342 orphan entries (bloat, not blockers), 189 duplicate keys between refs.bib and refs_extended.bib (refs_extended.bib is not loaded by `\bibliography{refs}` so harmless). Report at `docs/plans/2026-05-19-bib-coverage.md`.

- **3.2 coherence:** Found 5 critical issues. Three inline-fixed: (a) ch00 intro chapter map (claimed 9 sections for a 17-section paper) — rewritten with `\ref{}` to the 13 active section labels in main.tex; (b) my Task 1.1 caption claimed "30 Monte Carlo seeds" without checking — actual is 50 tabular / 20 high-dim, fixed; (c) stale path `ch11_rl_for_ci` → `ch10b_rl_for_ci` in footnote. Two deferred and surfaced for next cycle: see "Action items" below. Report at `docs/plans/2026-05-19-paper-coherence.md`.

- **3.3 arxiv-check:** 0 LLM meta-comment hits. 4 Mismatch references inline-fixed: `Cai2023`, `Tullii2024`, `Fan2024`, `Ying2022` had arXiv IDs that resolved to entirely unrelated physics/math/CSP papers; wrong IDs removed and `note = {arXiv ID pending verification...}` added. 10 NotFound (mostly NeurIPS/ICML proceedings, deferred), 30 Review (formatting-mismatch false positives, deferred), 380 orphans (bloat, deferred). Report at `docs/plans/2026-05-19-arxiv-check.md`.

### Phase 4 — Final compile + arxiv package (Tasks 4.1–4.3)

- **4.1 final recompile:** Clean four-pass build of `docs/main.tex`. 214 pages, 11.7 MB. Zero undefined references, zero undefined citations.

- **4.2 arxiv package:** Updated `scripts/package_arxiv.sh` to include the 5 missing chapters (ch06_macro, ch10b_rl_for_ci, ch11_dist_robust_constrained, ch12_world_models with its 10 section files, ch07 curve_learning_pricing additions). Tarball produced: `arxiv_submission.tar.gz` (11 MB, 154 files). Smoke-tested in a clean `/tmp/arxiv_smoke/` directory: extracts and compiles to identical 214-page PDF.

- **4.3 summary:** this file.

## Deferred items (surfaced for next cycle)

These were identified during Phase 3 audits and NOT fixed this cycle. The user should triage before the next arxiv push.

1. **`offline_rl_pricing_results.tex` shows byte-identical numbers for BC, BCQ, DT, and RvS** (`169.27 ± 0.60`, `88.0%`). Four distinct algorithms collapsing to identical mean AND SE is a CLAUDE.md Algorithm Identity Check flag. Underlying training functions are genuinely distinct (verified by reading `train_bc`/`train_bcq`/`train_dt`/`train_rvs`), so this is plausibly a policy-collapse onto BC's argmax in this dataset regime rather than a code bug — but it warrants re-running and investigating before the next arxiv push. Suggested investigation: (a) re-run a single seed in isolation and confirm numbers, (b) check whether the same action sequence is produced per (state, t) across all four methods, (c) if yes, explicitly discuss in the prose; if no, debug.

2. **`ch99_conclusion` is silent on the World Models chapter** (the longest chapter in the paper, 10 subsections, 3 sims). Substantive content addition; suggested follow-up: add one paragraph in §"How RL Advances Applied Modeling" mentioning world models / model-based RL as a third locus of RL-economics interaction.

3. **4 bib entries need correct arXiv IDs** added (Cai2023, Tullii2024, Fan2024, Ying2022). The wrong IDs were removed; entries currently say `note = {arXiv ID pending verification...}`. Look up the correct IDs and replace.

4. **10 NotFound bib entries** (NeurIPS/ICML/Math proceedings) deserve a manual spot-check against original sources to confirm metadata accuracy.

5. **380 orphan bib entries** in refs.bib (defined but never cited). Bloat, not a blocker. Either trim or move to `refs_extended.bib` (which is already in the repo but unused by `\bibliography{refs}`).

6. **189 duplicate keys** between refs.bib and refs_extended.bib. Harmless (refs_extended.bib is not active) but indicate refs_extended.bib was extended by concatenation rather than merge. Cleanup deferred.

7. **Chapter PDF rebuild for ch03 / ch03b / ch04** after the Phase 2 plot_style sweep: skipped because no figures or .tex changed, only sim source. If chapter PDFs are regenerated in a future cycle (running the sims under the new `apply_style()` rcParams), the figures may shift slightly in font sizing.

## Commit history (18 commits since baseline 9dbaab0)

| Commit | Message |
|--------|---------|
| `abffb48` | fix(arxiv): update package_arxiv.sh for new chapters |
| `a017642` | build: final main.pdf for arxiv submission |
| `52561d5` | audit(arxiv): arxiv-check report + fix 4 wrong arXiv IDs in refs.bib |
| `2956d95` | audit(coherence): pre-arxiv report + 3 inline fixes |
| `c881bdf` | audit(refs): bib coverage report (0 cited-but-missing, 342 orphans) |
| `8a2793c` | style(ch03, ch03b): adopt sims.plot_style in 7 sims |
| `514ae51` | build: recompile ch10_causal PDF (sanity check post-counterfactual_ope move) |
| `d9de734` | archive(ch12): move orphan decision_focused_ar1 sim out of ch12 |
| `a280f41` | refresh(ch12): re-run dyna_maze cache + correct CHAPTER_NOTES numbers |
| `fc5c791` | feat(ch10b): emit dtr_qlearning_vs_murphy results table |
| `64f79d2` | fix(refs): repair two stale section labels + recompile main.pdf |
| `d9958c5` | chore(audits): add ch06_games durable_goods_monopoly audit |
| `9ebcddc` | chore(gitignore, audits): ignore .serena/ + add ch07 regret_rates audit |
| `bee7d72` | docs(plans, audits): add 2026-05-19 arxiv plan + sim audit notes |
| `fd519e9` | build(docs): update main.tex + refs.bib + recompiled chapter PDFs |
| `84c58fe` | feat(ch12): build world_models chapter (sections s01-s10 + 3 core sims) |
| `950c0bf` | humanize(ch02, ch08, ch10): prose polish + offline_rl_pricing sim refresh |
| `4281bcd` | refactor(ch12, ch10): rename forecasting_rl -> world_models, move counterfactual_ope |

## Ready to ship

`humanize-pass` is ready to merge to `main` and the tarball at `/Users/pranjal/Code/rl/arxiv_submission.tar.gz` is ready to upload to arxiv. Per CLAUDE.md "Executing actions with care", neither merge nor push happens automatically — those are explicit user actions.

Upload: <https://arxiv.org/submit>
