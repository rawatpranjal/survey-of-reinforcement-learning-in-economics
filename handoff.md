# Handoff — 2026-05-22 — humanize-pass

## Where we left off
Phase B re-audits + Phase C polish landed. **B2 closes (10%).** **B1 regressed to 30%** — adversarial Opus caught B1.6 (stock+harvest trajectory figure missing); polish-Sonnet had skipped it. C1 (carbon 5→10 seeds), C2 (ch12 10→6 sections), C3 (savings stdout), C4 (bib sweep) all DONE. Carbon tex numbers drifted; updated in-session.

## Active streams (clustered)
```
[content lock]  A1 ch99 expand || A2 ch10b intro || A3 ch00 refresh   — PARALLEL
                A4 ch09 RLHF lit fills || A5 GARP gap para             — PARALLEL
[sim audits]    B1.6 fishery trajectory figure                         — single deferred item
[submit D]      blocked on A + B1.6                                    — SEQUENTIAL
```

## Decisions made this session
- Phase B: Opus re-audit only, no fixes (user pre-decided).
- Adversarial Opus on B1 caught what polish-Sonnet missed. Validates the "separate executor from verifier" rule from CLAUDE.md.
- Carbon tex updated in-session (no_hiding) when 10-seed values drifted: λ peak `1.407±0.003 → 1.405±0.002`, Lag QL return `172±3 → 178±4`, emissions `23±2 → 28±4`, Unc QL return `254±6 → 258±3`, "five seeds" → "ten seeds" ×3.
- ch12 fold landed at 10→6 (not spec's 3-4) because 6 is the natural floor after applying spec merge rules: s01_intro + s03_dyna_q + s04_deep_mbrl + s06_objectives_convergence + s09_dual_sim + s10_synthesis.
- B1.3 / B1.4 in fishery remain footnote-only (no code fix). Polish audit accepted; re-audit notes Reviewer-2 still bites but not score-moving.

## Open questions
- Should B1.6 be folded into Phase B (closes that phase entirely) or treated as a known deferred gap that ships? Strict reading of spec ("close when all six landed/footnoted") fails on B1.6.
- Merge humanize-pass → main now or wait for Phase A?

## Landmines / gotchas
- **ch12 working-tree state**: 7 old section files deleted, 3 new files (s01_intro.tex, s04_deep_mbrl.tex, s06_objectives_convergence.tex) UNTRACKED. Stage deletions + new files in the same commit to avoid orphan-deletion churn.
- Old ch12 originals saved in `ch12_world_models/tex/backups/2026-05-22-174705_*.tex`. Keep per backup pattern.
- `carbon_constrained_production` cache regenerated with 10 seeds. Do NOT bump back to 5 — tex numbers now match 10-seed run.
- ch11 PDF + ch12 PDF compiled this session — each shows one `section:rl_algorithms` cross-chapter warning. Expected in chapter-only compile; resolves in master `main.tex` build.
- spec.md Phase B1 now lists B1.6 as "MISSING — defer". ~30-60 min next session.
- `refs.bib` has minor flag: `Thoeni2026nmfg` key year mismatches `year={2025}` body. See `docs/plans/2026-05-22-refs-bib-sweep.md`.
- ch03_theory cache 3.7G (from prior /start). Untouched. Leave alone.

## Suggested next move
**B1.6 (fishery trajectory figure) — 30-60 min**, closes Phase B fully. Instrument `compute_paradigm` to store seed-0 `s_curves` + `h_curves`; add second panel to `generate_outputs`; invalidate caches. Then move to A1 ch99. This unblocks the entire Phase D pipeline.
