# Project Memory — RL Survey Monograph

Long-term decisions and learnings, with the why. A line earns a place only if a future
session would get it wrong without it.

## Scientific / structural decisions

- **Theorem+proof rigor goes in the RL theory chapter (`ch03_theory`), and it is about RL,
  not IRL.** Enoch Kang's 2026 write-up is the *style* model (state result → show proof →
  cite latest → highlight the proof), plus a proof source for the deep-RL convergence
  results. It is not an instruction to add inverse-RL. The theory chapter should read as the
  monograph's rigorous foundation. Detail: `docs/theory-rigor-rl.md`.
  _Why:_ the user corrected an early misread ("not IRL, this is the RL chapter"). The chapter
  today has one `theorem` env and zero `proof` envs, so this is net-new rigor.

- **Every number in the paper must trace to generated simulation output; zero manual entry,
  enforced by a build gate.** Target is all numbers (tables → captions → prose via macros),
  reached in stages. Detail + manifest: `docs/sim-automation-audit.md`.
  _Why:_ the user's rule is "eliminate ANY manualization to keep numbers at very high rigor."
  Hand-typed numbers silently drift from the sims that produced them.

## Working conventions learned

- **`docs/` IS the project wiki.** Research `.md` files accumulate in `docs/` and get
  organized under `docs/index.md`. There is no `docs/wiki/` subdir. Roadmap and memory point
  into `docs/`.

- **All `\input` paths in the LaTeX are written relative to `docs/` (the compile dir), not to
  the including file.** A tool that resolves nested `\input`s relative to the parent file
  will silently miss sub-files (e.g. ch12's `s01_intro`..`s09_dual_sim`). Resolve every
  `\input` against `docs/`. This bit the bib-coverage check once (false "84 missing" that was
  really 0 once resolved correctly).

- **`docs/refs.bib` is trimmed to exactly the cited set** (489 keys == 489 cited, 2026-07).
  A `scripts/trim_orphan_bib.py` prunes orphans, so an uncited entry added now (e.g.
  `Kang2025`) may be stripped until something cites it. Harmless for the build.

- **Sync a non-checked-out branch with `git fetch origin <b>:<b>`, never `git branch -f`.**
  `git fetch origin main:main` refuses a non-fast-forward instead of silently moving the shared
  ref. `git branch -f main origin/main` force-moves the ref for every worktree and can discard
  an unpushed commit; the global rules ban that form in a shared tree. (It was used twice on
  2026-07-12 and was safe only because both moves happened to be fast-forwards.)

## Repo cleanup (2026-07-12, Workstream 0)

- New work lives on the **`rl-rigor` worktree** (off `humanize-pass`); the primary checkout
  is read-only. `main` is a clean ff of `humanize-pass`; the `main` move + push is deferred to
  the `rl-rigor` → `main` merge.
- Deleted as dead/diverged: `ch03a/` (its `\input` was commented at `docs/main.tex:170`,
  replaced by `ch03a_bm/`), `arxiv_submission 2/` (stale snapshot, forked bib), the abandoned
  `.claude/worktrees/` checkout, `planning_learning_v2.tex`. All recoverable from git history.
  _Salvaged `Kang2025` from the forked bib into `docs/refs.bib` before deleting it._
