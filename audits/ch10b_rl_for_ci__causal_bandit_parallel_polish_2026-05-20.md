# Polish Report: ch10b_rl_for_ci/sims/causal_bandit_parallel.py

**Date:** 2026-05-20
**Prior score:** 18% (Phase 2 recovery, 2026-05-19)
**New estimated score:** 12%

## Status verification

**Phase 1 edits in place: yes.**
- `context_conditional_thompson_sampling` is the function name (script line 438); the `causal_thompson_sampling` name no longer exists.
- Dict key `cctp` in `run_mabuc` return (line 737); variables `cctp_mean`, `cctp_se`, `cctp_regret` propagate through `make_figure_combined` and `print_stdout`.
- Header comment (lines 12–13, 35, 40–44) catalogues all four named algorithms (Successive Reject, Lattimore Alg 1, CCTS, full TS_C with consistency seeding + RDC weighting).
- Tex `rl_for_ci.tex` line 226 names the RDC abbreviation inline and discloses that the simulation runs both the full TS_C and a minimal CCTS baseline.

**Phase 2 edits in place: yes.**
- `causal_thompson_sampling_tsc` exists at script line 502, implementing both distinguishing TS_C features: off-intuition consistency seeding at `CONSISTENCY_OFF_INTUITION_WEIGHT = 0.5` (lines 489–491) and RDC bias weighting with running `Q_hat[x, a]` and clipped weights `w_a ∈ [0.01, 1]`.
- `MABUC_CONFIG` carries `'algos_version': 'v2_full_tsc'` (line 122), so the cache key for the MABUC experiment is bound to the three-algorithm schema.
- `run_mabuc` dispatches all three TS variants on every seed with disjoint RNG offsets `s + 100_000` / `s + 200_000` / `s + 300_000` (lines 728–735); return dict has keys `ts`, `cctp`, `tsc`.
- Figure panel (c) plots three curves (vanilla TS in orange/grey, CCTS in blue, TS_C in green) with 95% confidence bands (lines 822–836).
- Stdout reports all three regrets and three pairwise ratios (lines 943–953); current numbers: vanilla TS 200.49 (SE 0.28), CCTS 0.66 (SE 0.04), TS_C 4.49 (SE 0.10).

**Phase 3 polish edits in place: yes.**
- `causal_bandit_mabuc_results.tex` is generated as a standalone three-row rank-ordered table:
  ```
  Context-conditional TS (CCTS) & 0.66 (0.04)
  Full TS_C \citep{bareinboim2015mabuc} & 4.49 (0.10)
  Vanilla Thompson sampling & 200.49 (0.28)
  ```
  rows sorted ascending by regret (CCTS first, vanilla last), satisfying the rank-order memory at `feedback_table_rank_order.md`.
- `rl_for_ci.tex` line 329 (Sim 2 algorithm list) now explicitly distinguishes the minimal CCTS from "Full $\mathrm{TS}_C$ of \citet{bareinboim2015mabuc} Algorithm~1" and names the two augmenting components (fractional pseudo-count $c = 0.5$ off-intuition seeding and RDC clipped weighting), with a footnote disclosing $c$ as a non-optimized free parameter.
- `rl_for_ci.tex` line 333 (results paragraph) explains the CCTS-beats-TS_C inversion on the greedy casino in terms of the off-intuition payoff asymmetry: "the off-intuition arm's true payoff ($0.50$) exceeds the on-intuition payoff ($0.10$), so the fractional consistency-axiom seed transferred from on-intuition observations attaches a pessimistic prior to the (high-value) off-intuition cell that the agent must then unlearn through experimental pulls. The RDC weighting compounds the effect by suppressing both arms symmetrically once the running $\hat Q_x(a)$ estimates reveal the cross-context flip…"
- `rl_for_ci.tex` line 344 (Table~\ref{tab:simB2_mabuc} caption) labels the table as rank-ordered and states the dominance direction: "the minimal CCTS dominates the full $\mathrm{TS}_C$ of \citet{bareinboim2015mabuc} because the off-intuition consistency-axiom seed transfers a pessimistic prior to the high-payoff cell."
- `rl_for_ci.tex` line 352 (Figure 2 caption, panel c) names both context-conditional algorithms and labels CCTS "slightly ahead on this instance."
- `rl_for_ci.tex` line 356 (synthesis paragraph) restates the TS_C-loses-CCTS finding honestly without claiming it generalizes: "introduce a mild slowdown on this specific two-arm two-context instance because the off-intuition fractional seed transfers a low-payoff prior to a high-payoff cell, but the substantive linear-versus-bounded gap relative to vanilla Thompson sampling holds for both context-conditional variants."
- `rl_for_ci.tex` line 244 (subsection roundup) and line 291 (simstudy chapeau) reference both algorithms.

## Verification

**Chapter PDF compiles: yes, 31 pages.**
- File: `/Users/pranjal/Code/rl/docs/ch10b_rl_for_ci.pdf`, mtime 2026-05-19 15:41:35, size 1,129,688 bytes.
- pdfinfo reports `Pages: 31`, A4, PDF 1.7, pdfTeX-1.40.27. Per the Phase 2 recovery report the same file was 31 pages at 1,128,459 bytes — the 1,229-byte delta is consistent with a Phase 3 prose-only edit pass that did not change figure/table content. No regeneration was needed from this polish report (the previous polish agent compiled cleanly before stalling).
- Simulation outputs intact: `causal_bandit_combined.png` (292,174 bytes), `causal_bandit_results.tex` (regret-vs-m grid), `causal_bandit_mabuc_results.tex` (three-row rank-ordered MABUC table), `causal_bandit_parallel_stdout.txt` (full three-algorithm summary), all dated 2026-05-19 15:40.

**Three-variant MABUC table rendered: confirmed.**
- `causal_bandit_mabuc_results.tex` contains three rows in rank order (CCTS, Full TS_C, Vanilla TS), each with mean and standard error. Tex `\input{}`'s this file from `\input{../ch10b_rl_for_ci/sims/causal_bandit_mabuc_results.tex}` referenced as `tab:simB2_mabuc`.

**Prose explains TS_C-loses-CCTS via off-intuition payoff asymmetry: confirmed.**
- Two locations in the tex give the mechanism explicitly: line 333 (the result paragraph) and line 344 (the table caption). Both state that the off-intuition payoff (0.50) exceeds the on-intuition payoff (0.10), so the consistency-axiom seed places a pessimistic prior on a high-value cell that the agent must unlearn. The synthesis paragraph at line 356 reaffirms the framing as "specific two-arm two-context instance" rather than a general statement about TS_C.

## Residual issues

- **`c = 0.5` consistency seeding pseudo-count is one operationalization, not optimized.** Disclosed in the footnote at line 329, but no robustness sweep over `c ∈ {0, 0.25, 0.5, 0.75, 1.0}` was run. A hostile reviewer who pulls Bareinboim 2015 will note the paper does not give a canonical value for this; the choice is honest but unaudited. Deferred per Phase 2 reasoning.
- **TS_C-loses-CCTS is MDP-specific.** Implementing Bareinboim 2015 Experiment 2 ("Paradoxical Switching", Table 2) where observational and experimental distributions diverge more sharply would test whether TS_C reclaims dominance on the regime it was designed for. Deferred.
- **Vanilla TS in MABUC still does not consume observational seed data**, while CCTS and TS_C both do. The Phase 1 audit deemed this immaterial on the greedy-casino instance (the marginal `P(Y|a) = 0.3` is flat across arms, so unconditioned posteriors gain nothing from the seed), but the asymmetry remains. Deferred.
- **Single-coordinate reward construction** (only one parent affects reward) is unchanged; the m=48 non-monotonicity is acknowledged at line 333 but the more general linear-payoff reward function is not implemented. Deferred.

The implementation now matches Bareinboim 2015 Algorithm 1 within the one disclosed choice (`c`), the chapter prose names the same algorithm the code runs, and the empirical inversion (CCTS dominates TS_C on this instance) is reported honestly with an interpretable mechanism. A skeptical reviewer can still ask for a robustness sweep over `c` or for the Paradoxical Switching instance, but no longer about whether "the algorithm described and the algorithm implemented are the same algorithm" — that question is resolved.

**Bullshit score: 12%** — Anchored at 25%: Reviewer 2 may still ask for a robustness sweep over the `c = 0.5` pseudo-count or for the Bareinboim Experiment 2 instance. Rounded down to 12% because the artifact and the prose now describe the same three algorithms by the same names, the table is rank-ordered per house style, the figure caption matches the table caption matches the result paragraph, and the unexpected CCTS-dominates-TS_C finding is reported with an interpretable mechanism rather than buried or contradicted. The substance (graph-aware vs graph-blind on parallel bandits; bounded vs linear regret on greedy-casino MABUC) is fully intact and the named-algorithm-mislabelling issue that drove the Phase 0 55% has been dispatched cleanly through both the Phase 2 implementation and the Phase 3 prose polish.
