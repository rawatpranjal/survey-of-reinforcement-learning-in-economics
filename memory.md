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

## Reference PDFs: 16 were the wrong paper (2026-07-15)

- **The `scripts/download_chXX_*.sh` scripts fabricated arXiv IDs, and curl saved whatever
  was at that ID under the intended filename.** They were auto-generated (by a
  `generate_download_scripts.py` that is not in the repo) from metadata that was itself
  invented: the Bennett-Kallus entry read `Authors: Bennett, Daniel T.` (the real author is
  Andrew Bennett) and pointed at arXiv 2306.12351, which is a paper on the union-closed
  conjecture. The real paper is 2110.15332. `curl` cannot tell the difference, so the repo
  accumulated astronomy, fluid-dynamics and pure-maths papers wearing RL filenames.
- A sweep of all 658 PDFs under `chXX/papers/` found **16 that were a different paper
  entirely**; 6 were cited in the survey. `tamar2015_coherent_risk.pdf` was the TRPO paper,
  so a coherent-risk citation pointed at trust-region optimization. `ch03b_deeprl_practice`
  held 6 of the 16, in the chapter that carries the deep-RL limitations argument. 13 were
  re-downloaded and verified against page 1; 3 were deleted (2 were "Super-RL: A general
  framework for offline reinforcement learning", a title whose only match anywhere on the web
  is this repo's own corrupted copy, so it appears to name no real paper).
- Seven of the corrupted `.md` extractions were **tracked and public on GitHub**, not just
  local. The `.pdf` files are gitignored but the `.md` are not, so a bad download ships.
- The download scripts were untracked local files and have been deleted. **Do not regenerate
  them.** Use `websource "<title>"` (it resolves via Crossref/OpenAlex rather than a
  guessed ID) and verify page 1 before trusting the file.
- `scripts/check_paper_pdfs.py` re-runs the sweep. It compares filename tokens against the
  first two pages and errs toward flagging: the ~24 standing flags are the Bertsekas book
  split into section-labelled chunks, which the heuristic cannot judge. **It cannot catch a
  right-title/wrong-version PDF**, which is how a Charpentier 2020 preprint sat behind a 2021
  journal citation.
