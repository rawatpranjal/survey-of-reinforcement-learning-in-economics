#!/usr/bin/env python3
"""Transcribe all chapter PDFs to markdown using docling.

Discovers all ch*/papers/*.pdf files across the repo and converts each to
markdown via docling's DocumentConverter. Existing .md files are skipped.
"""

import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter


def convert_with_pymupdf(pdf_path: Path) -> str:
    """Fallback converter using PyMuPDF for PDFs that docling can't handle."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def discover_pdfs(repo_root: Path) -> list[Path]:
    """Find all PDFs in ch*/papers/ directories, sorted by chapter then name."""
    pdfs = sorted(repo_root.glob("ch*/papers/*.pdf"))
    return pdfs


def main():
    repo_root = Path(__file__).resolve().parent.parent
    pdfs = discover_pdfs(repo_root)

    if not pdfs:
        print("No PDFs found in ch*/papers/ directories.")
        sys.exit(1)

    # Group by chapter for display
    chapters = {}
    for pdf in pdfs:
        chapter = pdf.parent.parent.name
        chapters.setdefault(chapter, []).append(pdf)

    print(f"Found {len(pdfs)} PDFs across {len(chapters)} chapters:\n")
    for ch, files in sorted(chapters.items()):
        print(f"  {ch}: {len(files)} PDFs")
    print()

    converter = DocumentConverter()

    total = len(pdfs)
    skipped = 0
    succeeded = 0
    fallback_count = 0
    failed = 0
    failures = []

    for i, pdf_path in enumerate(pdfs, 1):
        chapter = pdf_path.parent.parent.name
        md_path = pdf_path.with_suffix(".md")

        if md_path.exists():
            print(f"[{i}/{total}] SKIP  {chapter}/{pdf_path.name} (markdown exists)")
            skipped += 1
            continue

        print(f"[{i}/{total}] Converting {chapter}/{pdf_path.name} ... ", end="", flush=True)
        t0 = time.time()

        try:
            result = converter.convert(str(pdf_path))
            md_text = result.document.export_to_markdown()
            md_path.write_text(md_text, encoding="utf-8")
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s)")
            succeeded += 1
        except Exception as e:
            # Docling failed -- try PyMuPDF as fallback
            try:
                md_text = convert_with_pymupdf(pdf_path)
                md_path.write_text(md_text, encoding="utf-8")
                elapsed = time.time() - t0
                print(f"OK (fallback, {elapsed:.1f}s)")
                succeeded += 1
                fallback_count += 1
            except Exception as e2:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.1f}s): docling: {e} | pymupdf: {e2}")
                failed += 1
                failures.append((pdf_path, f"docling: {e} | pymupdf: {e2}"))

    print(f"\n{'='*60}")
    print(f"Summary: {total} total, {succeeded} converted ({fallback_count} via fallback), {skipped} skipped, {failed} failed")

    if failures:
        print(f"\nFailed files:")
        for path, err in failures:
            print(f"  {path.relative_to(repo_root)}: {err}")


if __name__ == "__main__":
    main()
