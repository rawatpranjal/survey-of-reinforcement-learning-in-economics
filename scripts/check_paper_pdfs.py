#!/usr/bin/env python3
"""Flag reference PDFs whose contents do not match their filename.

Written after a 2026-07-15 sweep found 16 PDFs under chXX/papers/ that were a
completely different paper. The cause was scripts/download_chXX_*.sh, which
curl'd hardcoded arXiv IDs that did not belong to the paper named in the
filename, so arXiv served a real but unrelated PDF and curl saved it under the
intended name. Nothing downstream noticed. Six of the sixteen were cited in the
survey; tamar2015_coherent_risk.pdf was in fact the TRPO paper.

Usage:
    python3 scripts/check_paper_pdfs.py            # scan, print findings
    python3 scripts/check_paper_pdfs.py --quiet    # exit 1 if anything flagged

Heuristic: take the alphabetic tokens of the filename and look for at least one
in the PDF's first two pages. Deliberately crude, and it errs toward flagging.

Known limits, both real:
  - A right-title/wrong-version PDF passes (the extraction is not compared
    against a canonical record), so this does not catch preprint-vs-journal.
  - Scanned PDFs with no text layer cannot be judged and are reported
    separately rather than flagged.
Read every flagged file before acting; false positives are common when the
filename concatenates author surnames (AdusumilliEckardt2022) or uses a short
code (EffMFCG, RICEN22).
"""

import argparse
import glob
import os
import re
import subprocess
import sys

# Words too generic to identify a paper by.
STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "survey",
    "paper",
    "learning",
    "reinforcement",
    "deep",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
}


def page_text(path, pages=2):
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-f", "1", "-l", str(pages), path, "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.lower()
    except Exception:
        return None


def tokens(basename):
    return [
        t.lower()
        for t in re.split(r"[^A-Za-z]+", basename)
        if len(t) > 4 and t.lower() not in STOP
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quiet", action="store_true", help="print only flagged files; exit 1 if any"
    )
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    pdfs = sorted(
        glob.glob(os.path.join(root, "ch*/papers/**/*.pdf"), recursive=True)
        + glob.glob(os.path.join(root, "appA*/papers/**/*.pdf"), recursive=True)
    )

    flagged, unjudgeable, ok = [], [], 0
    for p in pdfs:
        rel = os.path.relpath(p, root)
        txt = page_text(p)
        if txt is None:
            flagged.append((rel, "EXTRACT-FAIL", ""))
            continue
        if len(txt.strip()) < 40:
            unjudgeable.append((rel, "no text layer (scanned image?)"))
            continue
        toks = tokens(os.path.basename(p)[:-4])
        if not toks:
            unjudgeable.append((rel, "filename has no usable tokens"))
            continue
        if not any(t in txt for t in toks):
            first = " ".join(txt.split())[:70]
            flagged.append((rel, f"tokens {toks[:4]} absent", first))
        else:
            ok += 1

    if not args.quiet:
        print(
            f"scanned {len(pdfs)} PDFs: {ok} ok, "
            f"{len(unjudgeable)} unjudgeable, {len(flagged)} flagged\n"
        )
        for rel, why in unjudgeable:
            print(f"  [skip] {rel}\n         {why}")
        if unjudgeable:
            print()

    for rel, why, first in flagged:
        print(f"  [FLAG] {rel}\n         {why}\n         page 1: {first}")

    if flagged:
        print(
            f"\n{len(flagged)} flagged. Each is a candidate, not a verdict: "
            f"open it before acting."
        )
        return 1
    print("\nNo filename/content mismatches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
