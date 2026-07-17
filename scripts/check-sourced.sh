#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/check-sourced.sh DIR" >&2
  exit 2
fi

dir="$1"
if [[ ! -d "$dir" ]]; then
  echo "not a directory: $dir" >&2
  exit 2
fi

printf "%-10s %-7s %-8s %-8s %s\n" "STATUS" "PAGES" "WORDS" "SIZE" "MARKDOWN"

shopt -s nullglob
for original in "$dir"/*.{pdf,PDF,html,HTML,docx,DOCX,txt,TXT}; do
  stem="${original%.*}"
  md="${stem}.md"
  pages="-"
  if [[ "${original,,}" == *.pdf ]] && command -v pdfinfo >/dev/null 2>&1; then
    pages="$(pdfinfo "$original" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)"
    [[ -n "$pages" ]] || pages="-"
  fi

  if [[ -f "$md" ]]; then
    words="$(wc -w < "$md" | tr -d ' ')"
    size="$(wc -c < "$md" | tr -d ' ')"
    if [[ "$words" -ge 200 && "$size" -ge 1000 ]]; then
      status="DONE"
    else
      status="PARTIAL"
    fi
    printf "%-10s %-7s %-8s %-8s %s\n" "$status" "$pages" "$words" "$size" "$md"
  else
    printf "%-10s %-7s %-8s %-8s %s\n" "FAILED" "$pages" "-" "-" "$md"
  fi
done

for md in "$dir"/*.md; do
  stem="${md%.md}"
  has_original=0
  for ext in pdf PDF html HTML docx DOCX txt TXT; do
    if [[ -f "${stem}.${ext}" ]]; then
      has_original=1
      break
    fi
  done
  if [[ "$has_original" -eq 0 ]]; then
    words="$(wc -w < "$md" | tr -d ' ')"
    size="$(wc -c < "$md" | tr -d ' ')"
    printf "%-10s %-7s %-8s %-8s %s\n" "ORPHAN" "-" "$words" "$size" "$md"
  fi
done
