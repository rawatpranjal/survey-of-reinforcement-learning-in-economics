# bullshit-detector — iter_1 — 2026-05-19

**Bullshit score: 20%** — One DATA DRIFT finding on a meta-statistic about line-citation specificity. All test verdicts and counting reproductions HOLD against raw subagent outputs. Score corrected from initial 10% (the favorable rubric endpoint) to 20% per Pass 6 round-up discipline: rubric for "HOLDS + DATA DRIFT" is 10-30%, and picking 10% rather than the rounded-up middle would have repeated the same sycophancy this entire iteration is designed to catch. THRESHOLD NOT MET (user requires <=10%). Iteration 2 required.

## Header
- Claim sources: `/tmp/iter_1_scorecard.md`
- Code / artifact root: `/tmp/iter_1_routing.txt`, `/tmp/iter_1_counting.txt`, `/Users/pranjal/.claude/CLAUDE.md`
- Seed audit: none
- Run by: bullshit-detector inline in main agent
- Date: 2026-05-19
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Test 1 picked perplexity (EXACT) | HOLDS | — | — |
| 2 | Test 2 picked claude-md-management (EXACT) | HOLDS | — | — |
| 3 | Test 3 picked bullshit-detector (EXACT) | HOLDS | — | — |
| 4 | Test 4 picked humanizer (EXACT) | HOLDS | — | — |
| 5 | Test 5 picked mcp__serena__find_symbol (EXACT) | HOLDS | — | — |
| 6 | Test 6 picked context7 (EXACT) | HOLDS | — | — |
| 7 | "Exact picks: 6" | HOLDS | — | — |
| 8 | "Subagents that cited a specific CLAUDE.md:line number: 1 of 6" | DATA DRIFT | LOW | no |
| 9 | wc -l result 730 reproduced as 730 | HOLDS | — | — |
| 10 | grep TRIGGER count 10 reproduced as 10 | HOLDS | — | — |
| 11 | ls skills count 36 reproduced as 36 | HOLDS | — | — |
| 12 | grep RULE lines 344/415/528/539 reproduced exactly | HOLDS | — | — |
| 13 | "Total tests this iteration: 10" | HOLDS | — | — |
| 14 | "Total exact / clean: 10" | HOLDS | — | — |
| 15 | "Total non-clean: 0" | HOLDS | — | — |

## Findings

### Finding 1: "Subagents that cited a specific CLAUDE.md:line number: 1 of 6"

- **Claim source (verbatim):** "Subagents that cited a specific `CLAUDE.md:line` number: 1 of 6 (test 1 only)" — `/tmp/iter_1_scorecard.md:26`
- **Code evidence (verbatim):** Test 1's raw citation from `/tmp/iter_1_routing.txt`:
  ```
  CLAUDE.md LINE: Line 6 of RULE A: "DO NOT CALL WebSearch FIRST. DO NOT CALL WebFetch FIRST."
  ```
  The other 5 tests cited "Active skills table row" / "skills table row for humanizer" / "serena (MCP) row" / "MUST-USE table row 1" — all section/row references, no line numbers at all.
- **Category:** DATA DRIFT — "Line 6 of RULE A" is a section-relative position, not an absolute `CLAUDE.md:line` reference. Strict reading (per RULE D6, when uncertain report worse) is that 0 of 6 subagents gave an absolute file:line citation. Loose reading is 1 of 6. The scorecard committed to the looser interpretation without flagging the ambiguity.
- **Severity:** LOW — this is a meta-statistic about citation quality. Does not change any routing verdict or counting result.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  scorecard_claim["specific_clademd_line_count"] == 1  # PASSES on scorecard; FAILS on strict interpretation
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  scorecard_claim["specific_clademd_line_count"] == 0 or scorecard_phrasing.startswith("Subagents that cited any line position (relative or absolute)")
  ```

## Cross-cutting patterns

- **RULE D discipline held on the numeric claims.** All 9 numeric/integer claims in the scorecard reproduced their source-of-truth values exactly. No tilde, no rounding, no generous bucketing.
- **The one drift is in interpretive phrasing, not arithmetic.** "Specific CLAUDE.md:line number" is a phrase that can be read two ways; the scorecard picked the more flattering interpretation. RULE D6 says pick the worse one. Single LOW finding.
- **No propagated stale evidence (D4 compliance).** Subagent outputs were collected in this turn and quoted from `/tmp/iter_1_routing.txt`, not from earlier-session memory.
- **No asymmetric counting (D5 compliance).** Losses recounted row-by-row from the routing table; the 6+0+0=6 sum was verified against the row count.

## TDD execution sequence (for the next agent)

0. **Bullshit score 20%, exceeds <=10% threshold. ITERATION 2 REQUIRED.** The DATA DRIFT finding is on phrasing precision (interpretive ambiguity in "specific CLAUDE.md:line number"). Fix: add D7 to CLAUDE.md mandating that ambiguous interpretive phrasing commits to the stricter reading.
1. Strengthen CLAUDE.md: insert D7 (interpretive phrasing must commit to the stricter reading when ambiguous; explicitly state the scope of metric-like claims).
2. Re-run subagent routing + counting tests for iteration 2.
3. Re-author scorecard applying D1-D7.
4. Re-run bullshit-detector. Target: <=10%, all LOW severity.
