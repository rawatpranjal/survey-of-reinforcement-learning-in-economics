#!/usr/bin/env bash
# Build a journal-version PDF from its main.tex (or specified entry file).
#
# Usage:
#   ./build.sh <version_dir> [entry_file]
#
# Examples:
#   ./build.sh fntml_monograph_100pp
#   ./build.sh fntml_monograph_100pp abstract.tex
#   ./build.sh jel_outline outline.tex          # if added later
#
# The script extends TEXINPUTS and BIBINPUTS with ./shared so that each
# version's main.tex can \input{glossary}, \bibliography{refs}, and
# \includegraphics from figs/ without local copies.

set -euo pipefail

VERSION_DIR="${1:-}"
ENTRY="${2:-main.tex}"

if [[ -z "$VERSION_DIR" ]]; then
  echo "Usage: $0 <version_dir> [entry_file]" >&2
  exit 1
fi

JOURNALS_ROOT="$(cd "$(dirname "$0")" && pwd)"
SHARED="$JOURNALS_ROOT/shared"
TARGET="$JOURNALS_ROOT/$VERSION_DIR"

if [[ ! -d "$TARGET" ]]; then
  echo "No such version directory: $TARGET" >&2
  exit 1
fi
if [[ ! -f "$TARGET/$ENTRY" ]]; then
  echo "No entry file: $TARGET/$ENTRY" >&2
  exit 1
fi

export TEXINPUTS=".:$SHARED:$TARGET:${TEXINPUTS:-}"
export BIBINPUTS=".:$SHARED:${BIBINPUTS:-}"

JOB="${ENTRY%.tex}"
cd "$TARGET"
pdflatex -shell-escape -interaction=nonstopmode "$ENTRY" || true
bibtex "$JOB" || true
pdflatex -shell-escape -interaction=nonstopmode "$ENTRY" || true
pdflatex -shell-escape -interaction=nonstopmode "$ENTRY"

echo
echo "PDF: $TARGET/$JOB.pdf"
