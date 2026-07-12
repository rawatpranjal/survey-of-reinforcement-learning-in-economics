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
- **New workstreams added 2026-07-12 (items 14-26), not started.** W3 primary-source RAG
  retrieval (infra), W4 wiki synthesis articles, W5 proof-source library, W6 hard oracle tests
  for the sims. Dependencies: W3 is infra; W5 feeds Workstream 1; W6 reinforces Workstream 2;
  W4 uses W3 + W5. Suggested order: W3 first, then W5 and W4, with W6 in parallel.
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

### Workstream 3 — Primary-source RAG retrieval tool (fast, simple, primary only)

Infra the other workstreams lean on: fast pull-up of the actual paper text. Suggested to
build first.

14. **R0 [CRITICAL] Corpus inventory + reuse check.** List primary-source docs on disk
    (chapter `papers/*.pdf` + their markdown conversions, arXiv sources in `docs/`). Check
    existing tooling first (`websource` shared library, `/source` skill, any prior
    embeddings) so acquisition is not reinvented. Scope: primary sources only, exclude our own
    notes/audits. _Accept:_ a corpus manifest (path, title, format).
15. **R1 Ingest + embed.** Convert PDF-only sources to markdown (docling/tomd), chunk
    section-aware (~500-1000 tokens), embed with a local model, store vectors. Keep it simple:
    one script plus a flat store (faiss, or cosine over a pickled matrix if faiss is overkill),
    no server. _Accept:_ store built over the manifest; rebuild is one command.
16. **R2 Query CLI.** `rl-retrieve "query" [--k 8] [--source FILTER]` returns top-k passages as
    {paper, section/page, text} in under a second. _Accept:_ a known probe ("Bellman
    contraction proof") returns the correct paper span.
17. **R3 Wire into workflows (gate).** T0/T2 (proofs) and W4 (synthesis) call it; optional thin
    MCP/skill wrapper. _Accept:_ one real retrieval used inside a W5 proof lookup.

### Workstream 4 — Wiki build-out (secondary synthesis articles)

Grow `docs/` from a few notes into a knowledge base of our own synthesis, distinct from the
primary papers. Uses W3.

18. **W4.0 Taxonomy + template.** Fix the wiki topic list (chapters + cross-cutting themes) and
    a synthesis-article template (TLDR, key primary sources, synthesis, open questions,
    cross-links to chapters/sims/proofs). _Accept:_ taxonomy + template in `docs/index.md`.
19. **W4.1..W4.n One synthesis article per topic** (ranked): deadly-triad, offline-RL,
    bandits-pricing, MFG-macro, RLHF-alignment, causal-OPE, world-models. Each built from the
    W3 corpus, every claim quote-checkable against a primary source. _Accept per article:_
    verifiable source spans; links resolve; indexed.
20. **W4.crit Completeness critic (gate).** A pass that asks which topics/sources are missing;
    findings become the next articles.

### Workstream 5 — Proof-source library (Enoch-style explainers). Feeds Workstream 1.

Find more proof-based explainers like Enoch's and mine them for the cleanest proofs to reuse
(with attribution) in the theory chapter and wiki. Uses W3.

21. **P0 [CRITICAL] Candidate list.** Assemble proof-carrying RL explainers / monographs /
    notes: Agarwal-Jiang-Kakade-Sun (RL theory monograph), Szepesvari 2010 (Algorithms for
    RL), Bertsekas (RL and Optimal Control), Meyn (Control Systems and RL), Sutton-Barto,
    reputable RL-theory course notes, Enoch's `EK_RL_note.pdf`. _Accept:_ a ranked source list
    with acquisition status.
22. **P1 Acquire + convert + index.** Pull full text (`/source`), convert to markdown, index in
    the W3 RAG. _Accept:_ each source is readable markdown on disk and retrievable.
23. **P2 Proof library.** `docs/proof-library.md`: key results indexed by theorem, each with the
    cleanest source proof + citation + page, for direct reuse in T2..Tn. _Accept:_ every entry
    names a source with a quote-checkable span; covers the T2..Tn candidate results.

### Workstream 6 — Hard oracle tests for every simulation. Reinforces Workstream 2.

Guarantee each sim is correct on toy/analytical cases so the numbers carry zero hallucination,
bugs, or mistakes. Complements the S5 numbers-gate and the 7-point sim audit in CLAUDE.md.
Independent of the others.

24. **H0 [CRITICAL] Oracle inventory.** For each live sim, identify a toy case with a known
    closed-form / exact answer (2-state MDP with hand-computed $V^*/Q^*$; tiny bandit with known
    optimal arm and regret; DP vs analytic; contraction rate; policy-improvement monotonicity).
    _Accept:_ a table mapping each sim to its oracle.
25. **H1..Hn Hard test per sim.** `tests/test_<sim>.py`: assert the algorithm hits the oracle
    within tight tolerance on the toy case, plus invariants (monotone improvement, convergence,
    never beats the oracle). One at a time. _Accept per test:_ fails if the algorithm is broken,
    passes on the real code.
26. **H.gate Runner + gate.** `pytest` over all sim tests, wired alongside the S5 numbers-gate.
    _Accept:_ every live sim has at least one hard oracle test; runner green.

---

## Conventions and working preferences

How the user wants this work run. These govern every chunk above.

- **Go slow, one chunk at a time.** Frame → build → fresh-agent verify → ship. 3-strike halt.
  Break work into small units; finish and verify one before starting the next.
- **One decision at a time.** Surface open decisions singly, in plain chat, as numbered
  options (1/2/3) with a recommendation. Do not batch questions; do not use a multiple-choice
  form.
- **Decide the small stuff; escalate only the load-bearing.** Assume sensible defaults and
  repo convention for reversible calls. Bring the user only the directional or irreversible
  choices. Never hand back a plan-to-do-it when the next action is obvious.
- **Rigor, radical transparency.** Theorem then explicit, highlighted proof, citing the latest
  source. Read the actual primary source (full text, not abstracts); quote-checkable spans, no
  hallucinated claims.
- **Zero manual numbers.** Every results number traces to generated simulation output; a build
  gate fails on any hand-typed result number (Workstream 2). Hard oracle tests prove each sim
  correct on toy cases (Workstream 6).
- **Primary sources only** for the retrieval corpus and the reading (Workstream 3, 5).
- **Never self-certify.** A fresh agent or a hard pass/fail gate confirms substantive work.
  Report failures and real numbers plainly; open the live artifact before saying done.
- **Worktree discipline.** All writes in a worktree, never the primary checkout. Commit under
  the user's name, no AI mention, no co-author trailer, no em or en dashes in messages. Merge
  to `main` by fast-forward; the protected-branch push to `main` is the user's to run.
- **`docs/` is the wiki.** Research `.md` accumulates in `docs/` under `docs/index.md` and
  grows over time: `docs/theory-rigor-rl.md`, `docs/sim-automation-audit.md`, plus the planned
  `docs/proof-library.md` and synthesis articles.
