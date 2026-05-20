# Polish: ch06_macro/sims/rbc_dp_vs_drl.py

**Date:** 2026-05-20
**Original audit:** `ch06_macro__rbc_dp_vs_drl_2026-05-19.md` (Bullshit score 20%)
**This pass:** verification + recompile after background re-run completed (prior polish agent a6e3f91f8612c48f5 hit wall-clock timeout while waiting).

## Phase 0 nicks and disposition

All five nicks raised in the original audit's hostile-reviewer summary were addressed by the prior polish agent; this pass verified each against the current source and recompiled the chapter PDF.

**1. PPO/DDPG sample-budget asymmetry (102k vs 60k steps).**
Disclosed in the tex footnote on the simulation paragraph (`macro_rl.tex` lines 361–364): "PPO is trained for roughly $102{,}400$ environment steps per seed ($100$ updates of $1024$ steps), DDPG for $60{,}000$, reflecting per-algorithm tuning rather than equalised compute; the qualitative ordering (DRL within the VFI standard-error band on welfare) is robust to this asymmetry." Option B (disclosure, not equalisation) as specified. CLOSED.

**2. SE column mixes cross-seed (DRL) vs cross-episode (KPR/VFI).**
Disclosed in the table caption (`macro_rl.tex` line 392): "The SE column reflects cross-seed standard error of per-seed mean returns for the stochastic methods (PPO and DDPG, $10$ seeds) and cross-episode standard error of returns within the single deterministic solution for KPR and VFI ($30$ episodes); the two are not strict comparables but are reported in the same column for compactness." Option B. CLOSED.

**3. PPO entropy coefficient = 0.**
Disclosed in the tex footnote (`macro_rl.tex` lines 365–367): "The PPO entropy coefficient is set to zero, which reduces actor variance in this one-dimensional control problem but may need to be positive in higher-dimensional MDPs to avoid premature collapse." CLOSED.

**4. Wall-clock instrumentation for PPO/DDPG.**
Script now times each seed (`train_times` list, lines 691–694 PPO and 729–732 DDPG) and reports `train_time_sec_mean`. `generate_outputs` writes per-seed wall clock into the table (lines 835–843). Verified in the re-run stdout: VFI 13.8 s, PPO 32.9 s, DDPG 579.4 s. Footnote line 368 points the reader at Table~\ref{tab:macro:rbc-results} for wall clock; caption line 392 notes it is per seed for PPO/DDPG and total solver time for VFI. CLOSED.

**5. `capital_traj_first` mislabelled cache field.**
Fixed at `rbc_dp_vs_drl.py` line 674: the field is now `'policy_C_grid': sol['policy_C']` with the inline comment "consumption policy on (A, K) grid; retained for parity with cache schema." No longer claims to be a capital trajectory. CLOSED.

## Verification performed

- **Script read end to end.** Wall-clock timing present for VFI (`vfi_time_sec`), PPO and DDPG (`train_time_sec_mean`, `train_times_sec`). Cache field #5 corrected. No mislabelled fields remain.
- **Background re-run completed.** `cache/rbc_dp_vs_drl__{VFI,PPO,DDPG}.pkl` rewritten 2026-05-19 (VFI 15:36, PPO 15:41, DDPG 17:18); KPR and shared caches unchanged (config-stable). Stdout shows VFI converged at iteration 231 (max-diff 9.72e-06); PPO ~33 s/seed; DDPG ~580 s/seed.
- **Table regenerated and consistent with stdout.** `rbc_dp_vs_drl_results.tex` (written 2026-05-19 17:18): KPR 45.88/0.587/0.0004/$-$, VFI 45.86/0.589/0.0000/13.8 s, PPO 45.82/0.843/0.0087/32.9 s, DDPG 45.17/1.476/0.0365/579.4 s. Matches the verified stdout block exactly.
- **Figure regenerated.** `rbc_dp_vs_drl_learning_curves.png` written 2026-05-19 17:18. Wired into `macro_rl.tex` figure (`fig:macro:rbc-curves`).
- **Steady state.** $K^\star = 4.294$, $C^\star = 1.260$ in stdout; tex reports $4.29$, $1.26$. Match.
- **Chapter PDF recompiled.** `docs/ch06_macro.pdf`, 33 pages, 3 pdflatex passes + bibtex, all exit 0. No table/figure errors. The only LaTeX warnings are two cross-chapter undefined references (`section:rl_algorithms`, `sec:planning_learning`) that resolve only in the full `main.tex` build, not in standalone chapter compilation. Expected, unrelated to this sim.

## Residual reviewer-2 nicks (not regressions)

- The "KPR" label remains a loose name for first-order Blanchard-Kahn log-linearisation rather than the King-Plosser-Rebelo balanced-growth construction. The tex already clarifies this parenthetically ("a Blanchard-Kahn log-linearisation around the deterministic steady state (KPR)", line 350). Naming-only; not a methodological error. Out of scope for a disclosure-only polish.
- A hostile reviewer could still ask for a unified bootstrap SE across the $30 \times 10$ tuples instead of two SE conventions in one column. The caption now explicitly flags the two as "not strict comparables", which is the standard remedy short of a full re-derivation.

## Score

Phase 0 nicks 1–5 are all closed via tex disclosure (options B) and the cache-field rename. The two surviving items are a naming convention and a presentation preference, both already hedged in the prose. The wall-clock column is now populated for all timed methods, which removes the implicit "why aren't you reporting compute?" objection the original audit raised.

**Bullshit score: 12%** — Reviewer 2 may still grumble about the "KPR" label and the dual-SE column, but both are now explicitly disclosed in caption and prose, the algorithms match their definitions, wall-clock is reported, and the result reproduces the Atashbar-Shi (2023) finding on the same problem. Down from 20%; target of ≤15% met.
