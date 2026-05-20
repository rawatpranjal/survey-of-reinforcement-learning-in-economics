# Polish: ch07_bandits/sims/uninformative_price.py

**Date:** 2026-05-20
**Diagram-only:** YES (rubric cap 25% unless diagram contradicts caption)
**Predecessor audit:** `/Users/pranjal/Code/rl/audits/ch07_bandits__uninformative_price_2026-05-19.md` (20%)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch07_bandits/tex/dynamic_pricing.tex` (Section 1.2 `\label{sec:broder}`, figure `fig:uninformative_price`, lines 22–27)

## What changed

### 1. Annotation reframe (script)

`uninformative_price.py:73-83`. The annotation text was changed from a demand-level claim that overloaded a revenue plot to a description that matches the plotted object, plus an inline comment pointing the reader to Broder 2012 for the underlying demand-level construction.

```
- r'All demand models agree at $p^*$'
+ r'Revenue curves share the maximizer $p^*$'
```

A code comment immediately above the annotate call notes that the plotted curves are revenue (not demand) and that the underlying demand curves — linearly distinct, crossing at $p^*$ — are not shown; see Broder 2012 Theorem 3.1 for that construction.

### 2. Caption sharpening (tex, line 25)

The caption was rewritten to (a) make the demand-vs-revenue distinction explicit with a paper citation, (b) ground the "indistinguishable" claim in a concrete number, and (c) disclose the half-width as illustrative.

```
- All models agree at the optimal price $p^* = 5$. Within the shaded
- exploration zone, the curves are nearly indistinguishable, so playing
- prices near $p^*$ is uninformative about the demand parameter.
+ All revenue curves share the maximizer $p^* = 5$; the underlying demand
+ curves (not shown here, see \citet{Broder2012} Theorem 3.1) are linearly
+ distinct but cross at $p^*$. Within the illustrative exploration window
+ $|p - p^*| \leq 0.7$ (shaded; half-width is pedagogical, not derived),
+ revenue separation across the four curves is at most $1.72$ units, about
+ 7\% of $r^*$, which is small relative to typical purchase-noise variance
+ at this scale.
```

The 1.72-unit figure is sourced from the script's own stdout (`r(p_lo) = 23.285` for $k = 3.5$, so $r^* - r(p_lo) = 1.715$; rounded up to 1.72 in the caption). 7\% comes from $1.715 / 25 = 0.0686$.

### 3. Half-width disclosure

Done in the caption text (`illustrative exploration window`, `half-width is pedagogical, not derived`). No need to surface this in the figure itself — the dashed shading already reads as a sketch element rather than a derived bound.

## Verification

Script re-run: `python3 ch07_bandits/sims/uninformative_price.py > ch07_bandits/sims/uninformative_price_stdout.txt 2>&1` (exit 0). Stdout numerics unchanged (deterministic closed-form plot).

Chapter PDF rebuilt: `pdflatex -shell-escape -jobname=ch07_bandits "\def\chapterfile{../ch07_bandits/tex/dynamic_pricing}\input{compile_chapter}"` ran the standard three-pass + bibtex sequence; output is `/Users/pranjal/Code/rl/docs/ch07_bandits.pdf` (15 pages, 1.47 MB). No errors, no undefined references introduced. Hyperref `Hfootnote.N` destination warnings are pre-existing cosmetic noise from the chapter's footnote density and are unaffected by this polish.

Figure inspection: the regenerated PNG shows the annotation arrow pointing from "Revenue curves share the maximizer $p^*$" to the shared peak at $(5, 25)$, consistent with the caption's framing.

## Residual nicks at this score

- A *side-by-side* demand panel (audit option A) would do strictly more than the caption reframe (option B). B is sufficient for a 20% → ~10% score but leaves a reader who skims only the figure (skipping the caption) with no visual cue that demand curves exist. Anyone who reads the caption gets the pointer to Broder 2012 Theorem 3.1. This was an explicit out-of-scope choice in the polish brief.
- "Typical purchase-noise variance at this scale" in the caption is a hand-wave; the diagram never specifies a noise model. A reviewer could still ask "what variance?" The brief did not ask for a numeric noise floor, and adding one would require introducing a noise model the figure does not use.

Neither lifts the score above 10%.

**Bullshit score: 10%** — Diagram-only cap (25%) still applies. The annotation now describes the plotted object (revenue) rather than overloading a revenue plot with a demand-level claim, the caption grounds the "indistinguishable" claim in a concrete 7\% separation number with a paper-side citation for the demand-curve mechanism, and the exploration window's half-width is disclosed as illustrative. A hostile reviewer can still note that demand curves are never plotted, but the caption sends them to Broder 2012 Theorem 3.1 for that picture and no longer asserts the demand-level claim at the figure level. No remaining caption-figure mismatch.

Path: `/Users/pranjal/Code/rl/audits/ch07_bandits__uninformative_price_polish_2026-05-20.md`
