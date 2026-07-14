# Roadmap — RL Survey Monograph

Single point of contact. STATUS is the current state, CURRENT GOAL is what we drive
toward, BACKLOG is the ranked work. Long-term learnings live in `memory.md`; deep-dive
research lives in the `docs/` wiki (indexed at `docs/index.md`).

---

## STATUS

_Updated 2026-07-14._

- **Where the paper is.** arXiv survey, Phases A-C closed. `main.pdf` builds at 265pp with
  zero undefined references. The 2026-07-14 judge-audit triage cycle
  (`audits/_TRIAGE_2026-07-14.md`) is fully closed: all 20 ranked items fixed or decided.
- **Landed 2026-07-14 (triage close-out, commits 3b84e25..610d949 on `main`).** Ranks 1-9,
  12, 13 were fixed earlier in the cycle. This session closed the rest: rank 19 ch09 legacy
  archive (41 files) + registry fixes; rank 8 job_search full 54-min rerun, ablation footnote
  now artifact-backed; rank 11 kuhn wired into ch06 after fixing a real exploitability-metric
  bug and a Nash-family bug (corrected results reverse the method ranking, FP stalls at 0.33);
  rank 17 bairds rewritten to the true six-state Baird 1995 star, fresh audit 12%; rank 15
  orphan sweep (64 more files archived, landmines deleted, bus-engine table wired); rank 14
  regen wave; rank 16 appA polish across all 13 sims; rank 18 master-plan table; rank 10
  risk_sensitive rerun at 10 seeds (March single-seed numbers did not reproduce, story
  inverted) and wired into ch11 with SE column + policy figure.
- **Workstream side-effects.** Backlog S1 (ch11 reconnect + orphan triage) and S4 (runner
  registry + dead write paths) are effectively done via the triage fixes. S6 partially done
  (stale-stdout wave). Registry now 59 scripts, all paths verified.
- **In-flight chunk.** None. Next: wind-farm chunk (below) or T0/S0 entry points.
- **Open handback for the user:** none blocking; wind-farm chunk is user-approved and specced.

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

### Wind-farm curse-of-dimensionality chunk (user-approved 2026-07-14, triage rank 20)

Compile `ch03_theory/tex/curse_of_dimensionality.tex` into the survey and bring its sim to
standard. Sequenced as one chunk with internal gates:

- **WF1 Bibliography.** 14 of the section's 15 citation keys are missing from `docs/refs.bib`
  (chow1989complexity, papadimitriou1987, traub1988, du2021, liu2022deep, lu2025, jin2020,
  jin2021, ayoub2020, zanette2020, kearns1999, hotz1993, bray2022comment, rust1997). Verify
  near-miss existing keys (`HotzMiller1993`, `AyoubVTR2020`, `kearns2002`, `rust1996numerical`)
  before adding duplicates. _Accept:_ zero cited-but-missing keys for the section.
- **WF2 Sim refactor + 10-seed run.** Move `wind_farm_curse_study.py` from the nested
  `papers/curse_of_dimensionality/sims/` to `ch03_theory/sims/`; house conventions (sim_cache,
  plot_style, argparse flags, compute/output split); `n_seeds` 1 → 10 for the RL methods
  (~2h run); runner registry entry. _Accept:_ mean ± SE table, no "± 0" cells.
- **WF3 Tex rewrite + compile.** Sim subsection to 2 paragraphs / 1 table / 1 figure
  (keep computation_times.png); strip first person; de-overclaim the three theory-pathway
  framings (implementations are illustrative analogies, not the papers' algorithms); regenerate
  all prose numbers from the new run; `\input` slot at `docs/main.tex:163`. _Accept:_ full
  build clean; every number traces to generated output.
- **WF4 Audit (gate).** Full 7-point sim audit + judge report. _Accept:_ <50%, findings fixed.

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
