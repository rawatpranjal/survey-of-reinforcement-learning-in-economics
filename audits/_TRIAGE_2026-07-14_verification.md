# Verification — _TRIAGE_2026-07-14.md

Independent verifier pass, read-only. Four checks run against the judge audit files in
`/Users/pranjal/Code/rl/audits/` and git.

## Verdict summary

- **CHECK 1 (evidence links support their rows): PASS**
- **CHECK 2 (no dropped findings / score-distribution matches): PASS**
- **CHECK 3 (ranking-formula consistency): PASS** (second re-verification 2026-07-14, after the tie-group reorder — non-increasing order holds and every tie group, including the priority-9 and priority-6 groups the prior pass flagged, now obeys "higher judge score, then lower effort"; all rank cross-references correct)
- **CHECK 4 (skip-gate spot check): PASS**

---

## CHECK 1 — Evidence links support their rows — PASS

Every ranked row citing a judge report (Ranks 1-9, 12, 16 under the current numbering) was
checked against the cited audit file. Stated finding and score both appear in the report in
each case. Rank numbers below reflect the second reorder (knowledge_ladder 11→9, dtr 13→12,
appA batch 14→16); findings and scores are unchanged by the renumbering.

| Rank | Cited file | Score in row | Score in report | Finding match |
|---|---|---|---|---|
| 1 cobweb | ch12..cobweb | 50% | 50% | "seven" prose at s09:4,15,29; captions miscount panels (Finding 1); false policy-distance ordering (Finding 2). Match. |
| 2 fishery | ch12..fishery | 30% | 30% | env clips harvest to h_max=1.5, stock declines ~12-15 steps (Finding 1); h_traj pre-clip overlays stock line (Finding 2). Match. |
| 3 bm_fvi_fqi | ch03a_bm..bm_fvi_fqi | 25% | 25% | caption promises "Right: alpha trajectory," PNG single-panel (Finding 1, caption at planning_learning_v3.tex:324). Match. |
| 4 carbon | ch11..carbon | 25% | 25% | "five seeds" at tex:377 beside 10-seed table (Finding 1); "seventeen percent" should be sixteen at tex:400 (Finding 2). Match. |
| 5 axiom | ch09..axiom | 25% | 25% | proof "Choose δ∈(0,1)" at rlhf.tex:106 above sim's δ=2 (Finding F1); soften empirical framing of oracle-enforced LCPO PO=0 (F2). Match. |
| 6 offline | ch08..offline_rl_pricing | 25% | 25% | FQI in pessimism family at lines 147,165 / excluded at 156,160,65 (Finding 1); "broader coverage worsens FQI" rebutted by figure 16.7→27.4 (Finding 2, flagged since 05-19). Match. |
| 7 job_search | ch09..job_search_preference_learning | 25% | 25% | stdout truncated, main results table missing (Finding 1); online 73.92±0.05 / offline 73.99±0.03 / p=0.09 has no artifact (Finding 2); sibling job_search_rlhf.py collides on two chapter PNGs (Finding 3). Match. |
| 8 arch | ch02..algorithm_architectures | 25% (cap) | 25% (diagram cap) | r_t vs r_{t+1} clash across panel(c) formula, chapter TD eq (.tex:97), env-loop label (Finding 1); caption variance claim no figure referent (Finding 2). Match. |
| 9 knowledge_ladder | ch07..knowledge_ladder | 20% | 20% | both result tables orphaned while prose cites "four checkpoints" + "rate-diagnostic columns" (Finding 1); stale stdout headers (Finding 2). Match. |
| 12 dtr | ch10b..dtr_qlearning_vs_murphy | 15% | 15% | seed_scheme added only to DQN_HD_CONFIG, not the tabular configs whose seeds changed in the same commit 32107f7 — latent stale-cache (Finding F3). Match. |
| 16 appA batch | 13 appA reports | 10-20% | banach/envelope/gradient 20; hilbert/lagrangian/robbins/spectral 10; span 10-20 confirmed | banach predicted-vs-measured iteration cols diverge, γ=0.9/0.99 rows nan/analytic-extrapolation (banach F1); envelope panel-B title "holds" overclaim (score line); gradient "measured" factor = closed-form geometric base (score line); missing SEs (lln_clt, markov); ~8 stdouts point at removed rl-theory-proofs worktree (lln_clt/neumann/spectral + AppB item 6). Match. |

No discrepancy. All 11 rows trace to their evidence.

---

## CHECK 2 — No dropped findings / score-distribution match — PASS

Grepped the "Bullshit score" line from all 24 `*_2026-07-14.md` audit files. Every score
maps to the triage's score-distribution table, sim by sim, with no omissions and no
mismatches:

| Score | Audit files (grepped) | Triage table |
|---|---|---|
| 50% | cobweb_paradigms | matches |
| 30% | fishery_paradigms | matches |
| 25% | bm_fvi_fqi, carbon, axiom, offline_rl_pricing, job_search_preference_learning, algorithm_architectures | matches (all 6) |
| 20% | banach, envelope, gradient_descent, knowledge_ladder | matches (all 4) |
| 15% | lipschitz, lln_clt, markov_stationary, martingale_convergence, neumann_series, dtr | matches (all 6) |
| 12% | jensen_gap, brock_mirman_newton | matches |
| 10% | hilbert_projection, lagrangian_duality, robbins_monro, spectral_radius | matches (all 4) |

24 sims accounted for (1+1+6+4+6+2+4), exactly the 24 audit files on disk.

Sims scoring ≥25%: cobweb (50), fishery (30), bm_fvi_fqi (25), carbon (25), axiom (25),
offline (25), job_search (25), algorithm_architectures (25) = 8 sims. All 8 appear in the
ranked focus list (Ranks 1-8). None missing.

---

## CHECK 3 — Ranking-formula consistency — PASS (second re-run, after tie-group reorder)

Second re-verification, 2026-07-14, against the triage file as reordered a second time (the
priority-9 and priority-6 tie groups were rearranged to resolve the prior FAIL). Tie rule as
stated by the triage (line 29): "Priority = Severity × Liveness … Ties broken by higher judge
score, then lower effort." Recomputed from each row's stated Sev, Live, judge score, and
effort (Live range "1-3" taken at its main-text max of 3):

| Rank | Item | Sev | Live | Priority | Judge score | Effort |
|---|---|---|---|---|---|---|
| 1 | cobweb | 5 | 3 | 15 | 50% | M |
| 2 | fishery | 4 | 3 | 12 | 30% | S-M |
| 3 | bm_fvi_fqi | 4 | 3 | 12 | 25% | S |
| 4 | carbon | 4 | 3 | 12 | 25% | S |
| 5 | axiom | 4 | 3 | 12 | 25% | S |
| 6 | offline | 4 | 3 | 12 | 25% | S |
| 7 | algorithm_architectures | 4 | 3 | 12 | 25% | S |
| 8 | job_search | 4 | 3 | 12 | 25% | M |
| 9 | knowledge_ladder | 3 | 3 | 9 | 20% | S |
| 10 | risk_sensitive_inventory | 3 | 3 | 9 | — (structural) | M |
| 11 | kuhn_poker | 3 | 3 | 9 | — (structural) | M |
| 12 | dtr | 2 | 3 | 6 | 15% | S |
| 13 | runner registry | 2 | 3 | 6 | — (structural) | S |
| 14 | stale stdouts | 2 | 3 | 6 | — (structural) | M |
| 15 | orphan outputs | 2 | 1-3 | 6 (max) | — (structural) | M |
| 16 | appA batch | 2 | 2 | 4 | 10-20% | S-M |
| 17 | bairds | 2 | 1 | 2 | — (inherited) | S |
| 18 | master-plan drift | 2 | 1 | 2 | — (structural) | S |
| 19 | ch09 legacy | 2 | 1 | 2 | — (structural) | M |
| 20 | wind-farm | 1 | 1 | 1 | — (inherited) | M |

**Non-increasing sort — holds.** Priority sequence 15, 12, 12, 12, 12, 12, 12, 12, 9, 9, 9,
6, 6, 6, 6, 4, 2, 2, 2, 1. Every consecutive step is non-increasing; appA (Rank 16, priority
4) sits below the priority-6 block.

**Every tie group now obeys "higher judge score, then lower effort."**
- Priority 12 (Ranks 2-8): fishery (30%) leads on judge score despite its S-M effort; the
  five 25%/S rows follow; job_search (25%/M) is last by lower-effort tiebreak.
- Priority 9 (Ranks 9-11) — the first reorder target: knowledge_ladder (judge 20%, effort S)
  is now at the top of the group, ahead of the two no-score structural rows
  risk_sensitive_inventory and kuhn_poker (both effort M). It wins on both sub-criteria
  (20% > none; S < M). The prior FAIL is resolved.
- Priority 6 (Ranks 12-15) — the second reorder target: dtr (judge 15%, effort S) now leads
  the block ahead of the three no-score structural rows; runner registry (structural, effort
  S) precedes stale-stdouts and orphans (both structural, effort M) by lower effort. The prior
  ambiguity is resolved in favor of the judge-scored row.
- Priority 2 (Ranks 17-19): bairds and master-plan drift (both effort S) precede ch09 legacy
  (effort M) by lower effort.

**Cross-references — all correct after the reorder.** quick-wins item 6 "Rank 9 — \input
knowledge_ladder_results.tex" (line 61) matches knowledge_ladder now at Rank 9; item 7 "Rank
12 — dtr config-hash key" (line 62) matches dtr now at Rank 12; item 8 "Rank 13 — registry
fixes" (line 63) matches runner registry now at Rank 13; items 1-5 still resolve (Ranks 4, 3,
5, 6, 7). The ch09-legacy row's "Rank-8 filename collision" / "fixes Rank 8's collision"
(line 51) matches job_search, unchanged at Rank 8. Appendix C "Ranks 13, 14, 15" (line 92)
matches the three roadmap-W2 structural items runner-registry (Rank 13, = roadmap S4),
stale-stdouts (Rank 14, = roadmap S6), orphans (Rank 15, = roadmap S1), and correctly excludes
dtr (now Rank 12, a judge finding, not a roadmap-W2 item). No stale rank reference found.

---

## CHECK 4 — Skip-gate spot check — PASS

For each of the 5 sampled skipped sims: `git log -1 --format=%cs` on the `.py` and its
`_stdout.txt`, plus `git status --porcelain`. Appendix A gate = (py last commit ≤ 2026-05-20)
∧ (stdout commit ≥ py commit) ∧ (clean in git status).

| Sim (.py) | py commit | stdout commit | py ≤ 05-20 | stdout ≥ py | clean | Gate |
|---|---|---|---|---|---|---|
| ch03b_deeprl_practice/sims/brock_mirman_bellman.py | 2026-05-19 | 2026-05-19 | yes | yes (eq) | yes | PASS |
| ch04_control_problems/sims/benchmark_bus_engine.py | 2026-05-19 | 2026-05-19 | yes | yes (eq) | yes | PASS |
| ch10_causal/sims/confounded_ope.py | 2026-03-23 | 2026-03-23 | yes | yes (eq) | yes | PASS |
| ch06_games/sims/cournot_bertrand_marl.py | 2026-05-19 | 2026-05-19 | yes | yes (eq) | yes | PASS |
| ch12_world_models/sims/dyna_maze.py | 2026-05-19 | 2026-05-19 | yes | yes (eq) | yes | PASS |

All 5 pass all three gate conditions. `git status --porcelain` returned empty for every
py/stdout pair (clean). All 5 are also present in the triage's "Skipped, May score carried"
list (line 69). Note: gate-2 holds by equality in every sampled case (stdout committed in
the same commit as the py); the "≥" is satisfied, not strict.

---

## Discrepancy list (after second 2026-07-14 reorder)

**Resolved (all previously-raised CHECK 3 items now fixed):**
- Non-increasing priority holds: 15, 12×7, 9×3, 6×4, 4, 2×3, 1. The old appA break (priority
  4 above priority-6 rows) is closed by moving appA to Rank 16.
- Priority-9 tie group fixed by the second reorder: knowledge_ladder (judge 20%, effort S) is
  now Rank 9, above the two no-score structural rows risk_sensitive_inventory (Rank 10) and
  kuhn_poker (Rank 11), both effort M. It wins on both sub-criteria (20% > none; S < M), so the
  stated tie rule is satisfied. (Was the prior FAIL driver.)
- Priority-6 tie group fixed by the second reorder: dtr (judge 15%, effort S) is now Rank 12,
  above the three no-score structural rows; runner registry (Rank 13, effort S) precedes
  stale-stdouts (Rank 14) and orphans (Rank 15), both effort M. The judge-scored row now leads
  its tie, resolving the prior ambiguity.
- Priority-12 tie ordering consistent with "higher judge score, then lower effort" (fishery's
  30% leads; job_search 25%/M correctly trails).
- All rank cross-references correct under the new numbering: quick-wins item 6 → Rank 9, item
  7 → Rank 12, item 8 → Rank 13; ch09-legacy row → Rank 8 (job_search); Appendix C → Ranks
  13/14/15 (runner-registry/stale-stdouts/orphans), with dtr at Rank 12 correctly excluded.
- CHECK 1 rank numbers updated to the new numbering (judge-cited rows now Ranks 1-9, 12, 16);
  findings and scores unchanged by the renumbering.

No open discrepancies. All four checks PASS.
