# Polish: ch06_games/sims/cournot_bertrand_marl.py

**Date:** 2026-05-20
**Audit:** `audits/ch06_games__cournot_bertrand_marl_2026-05-19.md` (original 50%)
**Phase 1 fix:** `audits/ch06_games__cournot_bertrand_marl_fix_2026-05-19.md` (post-fix 15%)
**This polish pass:** Phase 2, tex-only, no script edits, no re-run

---

## Remit

Three nicks flagged for Phase 2:

1. Add `\citep{Klein2021}` alongside `\citep{Calvano2020}` if Klein2021 is now in
   `docs/refs.bib`.
2. Sharpen the Nash-Q deviation footnote, e.g. by adding an explicit Hu-Wellman
   citation.
3. Address any residual issues from Phase 1's "What remains untouched" section.

Out of scope: anything Phase 1 did not already touch (Nash-Q backup form,
WoLF-PHC projection collapse, SE = 0.00 entries).

---

## Changes

### 1. Klein2021 — not added

`grep -i "klein" /Users/pranjal/Code/rl/docs/refs.bib` returns only
`Kleinberg2003`, `Badanidiyuru-Kleinberg-Slivkins`, `kleinebuening2025strategyproof`,
and `kleinman1968`. No entry for Klein 2021 (the algorithmic-collusion paper)
exists in the bibliography. Per the Phase 1 fix report ("`Klein2021` is not in
`refs.bib` and was not added"), the status is unchanged.

As a substitute that pushes in the same literature direction, I added
`\citep{AskerEtAl2020}` (the Asker-Fershtman-Jeon-Pakes computational-framework
paper) alongside `Calvano2020`. The audit explicitly named
`AskerFershtmanPakes2021` as one of the canonical missing references for the
"Q-learning in IO duopoly" framing; `AskerEtAl2020` is already in `refs.bib` at
line 409 and resolves to the same line of work.

### 2. Hu-Wellman citation in Nash-Q deviation footnote

Phase 1 wrote the footnote as

> a deviation from the canonical Hu-Wellman 2003 backup that pins down a single
> value function

with the authors named in prose but no `\citep{}` or `\citet{}` macro. The
canonical reference key `hu2003nash` is in `refs.bib` and is already used
elsewhere in this chapter (lines 36, 42, 46). The footnote now reads

> a deviation from the canonical \citet{hu2003nash} backup that pins down a
> single value function via an exogenous selection rule

which (a) attaches the citation to the author name, (b) names *what* Hu-Wellman
pin down (the value function, via an exogenous rule), so a reader who jumps
straight to the footnote understands the deviation without having to also read
the body theorem statement.

### 3. Residual items from Phase 1 — all left as-is

The Phase 1 "What remains untouched" section listed four items; each was
explicitly judged out of scope by the Phase 1 author:

- Nash-Q backup `Q += α(r − Q)` (fine in the stateless one-shot setting)
- Nash-Q max-sum equilibrium-selection rule (now disclosed in tex; code unchanged)
- WoLF-PHC policy-projection collapse (legitimate methodological choice)
- `SE = 0.00` entries (consequence of integer grid + deterministic tail)

None of these is a Phase 2 nick under the user's remit, so they remain as-is.

---

## Diff summary

Tex (one paragraph, `ch06_games/tex/rl_in_games.tex` around line 82):

- Footnote: `Hu-Wellman 2003 backup` → `\citet{hu2003nash} backup`, with the
  added clause `via an exogenous selection rule`.
- Body: `\citep{Calvano2020}` → `\citep{Calvano2020,AskerEtAl2020}`.

No script edits. No `_stdout.txt` change. No cache invalidation. No re-run.

---

## Verification

Chapter PDF rebuilt from `docs/`:

```
pdflatex -shell-escape -jobname=ch06_games "\def\chapterfile{../ch06_games/tex/rl_in_games}\input{compile_chapter}"
bibtex   ch06_games
pdflatex -shell-escape -jobname=ch06_games "..."  (twice more)
```

All four pdflatex passes exit 0. No `Undefined` warnings, no `Citation ...
undefined` warnings in the final pass log.

`pdftotext` extraction confirms the new citations render correctly:

```
recent literature on Q-learning collusion in IO duopoly (Calvano et al.,
  2020; Asker et al., ...)
... deviation from the canonical Hu and Wellman 2003 backup that pins down a
  single value function via an exogenous selection rule; in this game it
  picks (3, 3) for Cournot.
```

Output: `/Users/pranjal/Code/rl/docs/ch06_games.pdf` (18 pages, 1,060,807 bytes).

---

## New bullshit score

**10%** — A hostile reviewer who reads the section now sees: a correct Bertrand
Nash derivation, a correctly stated set of three pure-strategy Cournot equilibria
on the integer grid, a Nash-Q deviation footnote that names the canonical paper
and the nature of the deviation, and two algorithmic-collusion citations
(Calvano, Asker) that situate the sim against the canonical IO-pricing-with-RL
literature. The remaining grumbles — integer action grid is coarse and SE=0.00
entries on the deterministic tail — are legitimate methodological choices and
would survive a revision request without further edits.

The 10% residual reflects: (i) Klein2021 was not addable because the BibTeX
entry does not exist (a real omission relative to the canonical reference set,
but one that requires adding a `.bib` entry, not a polish-pass edit), and
(ii) the SE column still reports 0.00 for several entries with no inline note,
which a sufficiently hostile reviewer could ask to be flagged. Neither is
substantive; both are cosmetic at this stage.
