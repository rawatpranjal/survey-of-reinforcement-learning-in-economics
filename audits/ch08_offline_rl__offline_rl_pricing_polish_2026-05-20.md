# Polish Verify: ch08_offline_rl/sims/offline_rl_pricing.py

**Date:** 2026-05-20
**Pass type:** Light verify-only polish (no script re-run, no algorithm changes)
**Prior scores:** Phase 0 audit 50% → Phase 1 fix 25% → Phase 2 recovery 15%
**Open flag entering polish:** `_INDEX.md` row reads "Phase 1/2 mismatch open — script on Phase 2 config but tex describes Phase 1 collapse (169.27)."

## What this pass did

Verified that the Phase 2 recovery commit (719243f, per the recovery report)
actually landed in the working tree and that every published artifact
(script config, table, stdout, prose, figure caption, chapter PDF) now
tells the same Phase 2 story. No code was edited, no script rerun, no
tex prose rewritten.

## Verification checklist

### 1. Script config — Phase 2 active

`ch08_offline_rl/sims/offline_rl_pricing.py`:

- `BEHAVIORAL_MARKUPS = np.array([5, 7, 8, 9], dtype=float)` (line 82). ✓
- `CONFIG_VERSION = 14` (line 123). ✓ (Phase 1 was 13.)
- `BEHAVIORAL_MARKUPS` is in `ENV_PARAMS` dict (line 135) so the cache key reflects the regime-dependent preferred prices.
- Header comment (line 21) explicitly states "BEHAVIORAL_MARKUPS = [5, 7, 8, 9]".
- Phase 1 setting documented for posterity at line 74 as a comment.

### 2. Stale 169.27 references in tex prose — none

`grep -n "169.27" ch08_offline_rl/tex/offline_rl.tex` returns ONE hit on
line 147. Context: it is inside the rebalance footnote acknowledging the
Phase 1 collapse: "The earlier version of this experiment used a
state-independent behavioral with 85% mass on $p = 10$. Under that
distribution, BC, BCQ-D, DT, and RvS all collapsed to the constant
policy $\hat\pi(s) = 10$ and reported bit-identical returns of
$169.27 \pm 0.60$; ... The state-dependent softer behavioral used here
breaks the collapse and lets the four methods differentiate."

This is not a live result number; it is a deliberate didactic mention
of the historical collapse number, framed as the failure mode that the
rebalance fixes. Keeping it in the footnote is the right call — it
documents why the experiment uses a state-dependent behavioral rather
than a concentrated one. No edit needed.

### 3. Live results — Phase 2 numbers in prose

Line 156 ("Two patterns emerge from Table~\ref{tab:offline_main}...")
and line 158 ("The four supervised-conditioning methods now produce
distinct policies..."):

| Method | Tex prose says | Phase 2 report says | Match |
|---|---|---|---|
| RvS | 97.0% | 97.0% | ✓ |
| BC | 96.8% | 96.8% | ✓ |
| DT | 96.3% | 96.3% | ✓ |
| CQL | 92.6% | 92.6% | ✓ |
| BCQ-D | 92.0% | 92.0% | ✓ |
| IQL-argmax | 91.8% | 91.8% | ✓ |
| FQI | 24.7% | 24.7% | ✓ |

Target return $R^\star = V^\ast(s_0) \approx 184$ explicitly named in
line 158. The Brandfonbrener2022 stochastic-return-conditioning footnote
also lands on line 158. The pessimism-vs-supervised reframing (under
near-on-policy behavioral, BC is itself near-optimal and pessimism pays
a small cost for robustness) is in line 156.

### 4. Coverage paragraph (line 169) — Phase 2 numbers

| Method, $\epsilon_b = 0.9$ | Tex prose says | Phase 2 report says | Match |
|---|---|---|---|
| BC | 95% | 95.4% | ✓ |
| CQL | 92% | 92.0% | ✓ |
| IQL-argmax | (near 92%) | 91.5% | ✓ |
| BCQ-D | 25.6% | 25.6% | ✓ |
| DT/RvS | 86% | 85.5 / 86.7 | ✓ |
| FQI | 17--27% range | 16.7 / 27.4 / 25.4 across $\epsilon_b$ | ✓ |

Concentrability framework callback to Definition~\ref{def:concentrability}
and the PEVI bound \eqref{eq:pevi_bound} preserved on line 169.

### 5. Table file — Phase 2

`ch08_offline_rl/sims/offline_rl_pricing_results.tex`:
8 rows, all distinct numbers, rank-ordered by performance:
192.41 / 186.58 / 186.28 / 185.27 / 178.08 / 177.05 / 176.67 / 47.48.
Labels: DP Oracle / RvS / BC / DT / CQL / BCQ-D / IQL-argmax / FQI.
No four-row collapse, no `169.27`. ✓

### 6. Stdout file — Phase 2

`ch08_offline_rl/sims/offline_rl_pricing_stdout.txt`:
Lines 56–62 report the same eight rows in the same order. Coverage
table (lines 67–69) reports the eps=0.05 / 0.3 / 0.9 columns matching
the Phase 2 recovery report exactly. ✓

### 7. Chapter PDF freshness

`docs/ch08_offline_rl.pdf` (May 19 05:36) is younger than
`ch08_offline_rl/tex/offline_rl.tex` (May 19 05:35:51) and younger
than `offline_rl_pricing_results.tex` (May 19 05:33:14). The compiled
PDF reflects the Phase 2 prose and the Phase 2 table. ✓ No
recompile needed.

## Findings

The `_INDEX.md` open flag ("script on Phase 2 config but tex describes
Phase 1 collapse (169.27)") was a snapshot from before the Phase 2
recovery commit landed. As of 2026-05-20:

- Script is on Phase 2 (`BEHAVIORAL_MARKUPS = [5, 7, 8, 9]`, config v14).
- All result numbers in tex prose (lines 156, 158, 169) are Phase 2.
- The 169.27 appears exactly once, inside the rebalance footnote that
  documents the prior collapse and why it was rebalanced. This is
  pedagogically correct and not a live result claim.
- Table, stdout, and chapter PDF are all on Phase 2.

The "Phase 1/2 mismatch" is closed.

## Residual issues unchanged from Phase 2

Carried forward without comment in this verify-only pass:
- IQL-argmax policy-extraction step is argmax-over-Q, not AWR (footnote
  on line 85 discloses).
- BCQ-D is the discrete-action variant, not continuous BCQ (footnote
  on line 105 discloses, with \citet{Fujimoto2019b}).
- DT is fused-token, not three-token-per-step (footnote in DT subsection
  discloses).
- DT/RvS sensitivity to $R^\star$ is not swept (the choice
  $R^\star = V^\ast(s_0) \approx 184$ is named in line 158 prose; no
  sensitivity analysis reported).
- FQI's 24.7% collapse is reported but not diagnosed (overestimation
  cascade is the stated mechanism, no ablation against a target-network
  FQI variant).

## Action items for `_INDEX.md`

Replace the row:

```
| 25% | ch08_offline_rl | offline_rl_pricing | [link](ch08_offline_rl__offline_rl_pricing_2026-05-19.md) | no — **Phase 1/2 mismatch open** |
```

with:

```
| 15% | ch08_offline_rl | offline_rl_pricing | [link](ch08_offline_rl__offline_rl_pricing_2026-05-19.md) | no (Phase 2 recovery; verified 2026-05-20) |
```

And remove the "OPEN" note in the Resolved High-Risk Findings table:
the entry should now read "Phase 2 recovery landed; verified
2026-05-20" rather than "Parallel session is reconciling."

## Bullshit score after polish

Phase 2's 15% score holds. No new sources of error introduced; the only
identified `_INDEX.md` flag was a stale entry from a snapshot taken
before the Phase 2 commit landed, not an actual artifact mismatch. The
substance of the chapter (rank order, coverage sweep, pessimism vs
supervised-conditioning reframing) is intact and internally
self-consistent across script, table, stdout, prose, and PDF.

**Bullshit score: 15%** — Reviewer 2 can still ask why BC outperforms
the pessimism family on the headline table (the rebalanced behavioral
is near-optimal, so imitation is hard to beat) and why FQI collapses to
24.7% rather than the audit-suggested 81.2% (broader action coverage
gives the unconstrained $\max_{a'}$ operator more out-of-distribution
Q-values to overestimate). Both findings are owned in the tex prose;
the chapter survives a hostile read.
