# Polish Report: ch07_bandits/sims/knowledge_ladder.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch07_bandits__knowledge_ladder_2026-05-19.md` (25%)
**Polish target:** <= 15%

---

## Scope

This is a tex-only polish pass. No re-run, no code changes, no figure regeneration. The three remaining "nicks" flagged by the prior audit were all about exposition, not implementation, so the fix surface is `ch07_bandits/tex/dynamic_pricing.tex` lines 163-165.

## What changed

### 1. Homegrown delta-hat estimator now disclosed (option B)

Added a footnote to the UCB-PI bullet in line 163 that:
- States the exact formula used: `delta_hat = delta_hat_max + (delta_hat_max - mean delta_hat)`, with the per-segment half-widths defined from the WARP bounds.
- Discloses provenance: "I am not aware of a published source for this exact form. It is included as a transparent reference point against the standard UCB-PI."
- Notes that Misra (2019) also estimates delta from data but uses a different expression.

This is the option-B fix specified in the polish brief. It does not falsely attribute the formula and it does not hide it in code.

### 2. Rate-diagnostic instability now flagged for all methods

Line 165 previously acknowledged only UCB-PI's failure to stabilize at R/log T. The polish broadens the disclosure to walk the reader through every column of the diagnostic table: "only Thompson Sampling's R/sqrt(T) is close to stable; epsilon-greedy's R/T and LTE's R/T^{2/3} drift downward, UCB1's R/sqrt(T) grows, and R/log T for plain UCB-PI quadruples." The concluding sentence now states explicitly that "T = 200,000 is too short for the predicted rates to visibly settle for most of these algorithms."

### 3. Finite-sample / asymptotic inversion called out

Line 165 now includes the sentence: "at this horizon the finite-sample ordering inverts the asymptotic rate order, with UCB-PI's logarithmic-rate algorithm dominated by TS empirically and by LTE despite LTE's nominal T^{2/3} rate." The reader is told the figure's legend order does not match the regret order.

## What did not change

- No code edits to `knowledge_ladder.py`. The delta-hat formula stays as it is; the polish is to *attribute* it correctly, not to replace it.
- No re-run with longer T. The brief flagged this as out of scope.
- Figure file, stdout, and tables are untouched. The cached numbers still match the regret values cited in the prose.

## Verification

Compiled `ch07_bandits` chapter PDF from `docs/`:

```
cd docs && pdflatex -jobname=ch07_bandits ... \
  && bibtex ch07_bandits \
  && pdflatex ... && pdflatex ...
```

Output: `/Users/pranjal/Code/rl/docs/ch07_bandits.pdf` (15 pages, 1.46 MB). No bibtex errors; one Misra2019 citation key resolved.

## Residual exposure

A hostile reviewer can still object that:
- the delta-hat formula is unusual even with disclosure, and would prefer the variance-tuned variant become the canonical UCB-PI on the figure.
- 10 seeds is a floor and LTE's relative SE remains ~14%.
- the figure legend still orders by claimed rate while the actual finish order is permuted; only the prose flags this.

These are real but they are now flagged in the text rather than buried, and the substantive numbers (UCB-PI-tuned wins, TS second, plain UCB-PI worse than LTE) are reported honestly.

**Bullshit score: 15%** -- Reviewer 2 still notes the homegrown delta-hat and the unstable rate columns, but the tex now lists every column that fails to stabilize, attributes the delta-hat formula transparently, and tells the reader the finite-sample ordering inverts the asymptotic order. The figure remains the same; the prose around it now matches what the figure actually shows.
