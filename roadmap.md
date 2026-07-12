# Roadmap — RL Survey Monograph

Single point of contact. STATUS is the current state, CURRENT GOAL is what we drive
toward, BACKLOG is the ranked work. Long-term learnings live in `memory.md`; deep-dive
research lives in the `docs/` wiki (indexed at `docs/index.md`).

---

## STATUS

_Updated 2026-07-12._

- **Where the paper is.** arXiv survey, Phases A-C closed (content lock, sim-audit fixes,
  polish). de-AI prose pass done across 17 chapters (`main.pdf` ~232pp). The old tasklist
  `spec.md` (2026-05-21) has been retired into this roadmap.
- **Two workstreams opened 2026-07-10.** (1) RL theory chapter theorem+proof rigor;
  (2) ruthless sim→table number automation. Research written up in the `docs/` wiki.
- **Working branch.** All new work is on the `rl-rigor` worktree (branch `rl-rigor`, off
  `humanize-pass` content). The primary checkout stays read-only. `main` is a clean
  fast-forward of `humanize-pass`; the actual `main` pointer move + push is deferred to the
  `rl-rigor` → `main` merge (which subsumes spec's Phase D1).
- **Just landed (Workstream 0, repo cleanup).** Deleted dead trees `ch03a/` and
  `arxiv_submission 2/`, dead `planning_learning_v2.tex`; untracked the 17 `tex/backups/`
  dirs and 35 compiled chapter PDFs (kept `docs/main.pdf`); retired `spec.md`/`handoff.md`,
  relocated `bloat.md`/`journal_target.md`/`CI_RL*.md` into `docs/`, moved the Ibarz paper
  into a gitignored `papers/` home; salvaged the `Kang2025` bib key into `docs/refs.bib`
  before deleting the forked bib. Build verified: all 53 `\input` files present, zero
  cited-but-missing keys.
- **In-flight chunk.** None of Workstream 1/2 started. Reference docs written. Take it slow,
  one chunk at a time, fresh-agent verify each before the next.
- **Next executable chunk:** T0 (acquire + read theory sources) or S0 (number manifest).
  Both `[CRITICAL]` entry points, independent, can run in parallel.
- **Open handback for the user:** whether to merge `rl-rigor` → `main` (and push) now, or
  keep accumulating on `rl-rigor`.

---

## CURRENT GOAL

Give the RL theory chapter (`ch03_theory`) explicit, highlighted Theorem + Proof rigor in
the style of Enoch Kang's write-up, citing the latest results, so the chapter reads as the
rigorous foundation of the monograph. In parallel, make every number in the paper traceable
to generated simulation output, with a build gate that fails on any hand-typed result
number. Slow and chunked: one unit, fresh-agent verify, then the next.

---

## BACKLOG

Ranked within each workstream. `[CRITICAL]` = load-bearing / entry point. `(gate)` = human
or fresh-agent checkpoint before proceeding. Full detail in `docs/theory-rigor-rl.md` and
`docs/sim-automation-audit.md`.

### Workstream 1 — RL theory chapter: theorem + proof rigor

1. **T0 [CRITICAL] Acquire + read sources.** Fetch `EK_RL_note.pdf` (gtown email
   `19efb63bbcb7528f`), arXiv 2502.14131, van der Laan-Kallus 2512.23805, Park et al.,
   Zhang et al. 2023, Antos-Szepesvári-Munos 2008 → markdown in `docs/`. Add the needed keys
   to `docs/refs.bib` (`Kang2025` already salvaged). _Accept:_ full-text markdown exists and
   is read (`/read`); proof skeletons extracted into `docs/theory-rigor-rl.md`.

2. **T1 Proof architecture + LaTeX scaffolding.** Highlighted-proof presentation (shaded
   `tcolorbox`/`mdframed` "Proof" env) + theorem/lemma/definition env set; the target result
   list; a compiling skeleton. _Accept:_ skeleton compiles; no prose yet.

3. **T2..Tn Draft each theorem + proof, one at a time.** Original then modern RL notation;
   full proof; latest citation; verified quote-checkable; proof highlighted. Candidates:
   Banach/contraction; policy improvement (with proof); Tsitsiklis-Van Roy TD + projected
   Bellman bound; deadly triad + Baird; Zhang 2023 online-SGD local convergence; van der
   Laan-Kallus FQE + the open sup-norm gap (labelled Open Problem). _Accept per chunk:_
   compiles; fresh-agent proof check passes; citation matches source.

4. **T_final Verify + expert read (gate).** Full compile, page count, fresh-agent check that
   each proof proves its theorem and citations match. Optional hand to Rust / Enoch.

### Workstream 2 — Ruthless sim → table automation

5. **S0 [CRITICAL] Number manifest.** Machine-readable inventory of every results number →
   source class (generated `\input` / orphan / hand-typed-table / prose-or-caption). Re-derives
   the provisional audit counts. _Accept:_ every entry names a real `path:line`; lands in
   `docs/sim-automation-audit.md`.

6. **S1 Reconnect cheap wins.** ch11 risk-sensitive: `\input` the existing generated file,
   delete the inline retyped table. Triage the orphans. _Accept:_ compile clean; each
   reconnected number matches its source `.tex`.

7. **S2 Convert own-sim hand-typed tables to generated.** ch03b results; any ch04 own-sim.
   ch04 third-party literature numbers stay hand-typed but get a cited source.

8. **S3 Prose/caption numbers → macro pipeline.** Sims emit `\def\chXXstat{...}` into a
   `*_macros.tex`; prose/captions reference the macro. Worst offenders first.

9. **S4 Fix runner + broken paths.** Add ch06_macro / ch10b / ch12 to `run_all_sims.py`
   REGISTRY (or glob). Fix the dead write dirs (`ch07_rlhf/`, `ch02_planning_learning/`) and
   hardcoded absolute paths. Regenerate `_stdout.txt`.

10. **S5 [CRITICAL] Honesty gate.** `scripts/check_numbers.py` fails the build if a chapter
    tex holds a results number not traceable to a generated `.tex`/macro. _Accept:_ catches a
    deliberately hand-typed test number.

11. **S6 Full rebuild + hash check.** Rerun all sims; confirm every consumed number matches
    fresh output; full build; page count.

### Inherited — arXiv submission (from retired `spec.md`)

12. **D1-D4** merge `rl-rigor` → `main` (subsumes the humanize-pass merge); full compile +
    page count; `scripts/package_arxiv.sh` tarball; submit arXiv v2 (gate: submit).

13. **Deferred** journal carve-outs (spec Phase E); reader-feedback citations promised by
    email: Moll (arXiv 2512.18892, 2602.20141), Weaver (Mksc 2022.0247).

---

## Conventions

- One chunk at a time. Frame → build → fresh-agent verify → ship. 3-strike halt.
- Every results number traces to generated output (the point of Workstream 2).
- Writes happen in a worktree, never the primary checkout. Commit only when asked.
- Reference research lives in the `docs/` wiki: `docs/index.md`, `docs/theory-rigor-rl.md`,
  `docs/sim-automation-audit.md`.
