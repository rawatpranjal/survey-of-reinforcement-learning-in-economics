---
owner: manuscript release maintainer
outcome: a locally compiled and independently smoke-tested arXiv submission archive
risk: medium
tracks:
  - scripts/package_arxiv.sh
  - scripts/check_bib.py
  - docs/main.tex
last_verified: 2026-07-31
---

# arXiv package runbook

This procedure creates the source archive for an arXiv submission. It does not upload the
archive, change an arXiv record, push a branch, or spend money.

## Preconditions

- Run from a clean committed manuscript worktree.
- Build `docs/main.pdf` through BibTeX and then run three more LaTeX passes so the
  bibliography, table of contents, and page references have converged.
- Confirm that every generated figure and table referenced by active LaTeX exists.
- Uploading requires separate current approval for the exact archive and arXiv record.

## Safe check

```bash
python3 scripts/check_bib.py --main docs/main.tex
git status --short
```

Expected result: the bibliography check exits zero and the worktree has no tracked changes.

## Procedure

```bash
bash scripts/package_arxiv.sh
```

The command builds the staging directory, archive, and verification record under temporary
sibling paths. It promotes the verified trio with same-filesystem renames only after the staged
source and a fresh archive extraction both compile and match `docs/main.pdf`. The verification
record moves last and acts as the commit marker.

## Expected evidence

- `arxiv_submission.tar.gz` exists and has a recorded SHA-256 digest, size, and archive-member
  count.
- The archive contains no macOS `._` metadata, links, absolute paths, or parent-directory entries.
- The packaging log reports a successful staged-source compilation and page count.
- A second extraction into a fresh temporary directory compiles with no undefined references
  or citations and matches the source PDF page count and extracted-text hash.

## Stop conditions

- Stop if the bibliography check fails, a manifest path is missing, or staged compilation fails.
- Stop if the archive contains absolute paths, parent-directory entries, caches, source papers,
  or build artifacts other than the required bibliography output.
- Stop before upload unless the user approves the exact archive and destination.

## Recovery and rollback

Correct the source or manifest, rebuild the manuscript, and rerun the procedure. A failure before
promotion leaves the canonical staging directory, archive, and verification record untouched. A
rename failure during promotion restores the previous trio. The command does not alter an external
arXiv record, so no external rollback exists before upload.

## Verification

```bash
tar tzf arxiv_submission.tar.gz
shasum -a 256 arxiv_submission.tar.gz
cat arxiv_submission.verify.txt
```

The packaging command itself rejects a dirty tracked worktree, checks archive paths and links,
extracts into a new temporary directory, compiles through BibTeX, and compares the page count
and `pdftotext` hash with `docs/main.pdf`. The evidence file records the exact commit, archive
digest, archive-member count, page count, and both text hashes. Canonical outputs remain untouched
until all checks pass.

## Automation and cadence

Not applicable: manual operation. Run the procedure only from a publication-gated commit.
