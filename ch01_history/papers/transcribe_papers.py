"""Transcribe all PDFs in ch01_history/papers/ to markdown using docling."""

import sys
from pathlib import Path

from docling.document_converter import DocumentConverter


def main():
    papers_dir = Path(__file__).parent
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files.")
    converter = DocumentConverter()

    success = 0
    skipped = 0
    failed = 0

    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists():
            print(f"SKIP (already exists): {pdf_path.name}")
            skipped += 1
            continue

        try:
            result = converter.convert(str(pdf_path))
            md_text = result.document.export_to_markdown()
            md_path.write_text(md_text, encoding="utf-8")
            print(f"OK: {pdf_path.name}")
            success += 1
        except Exception as e:
            print(f"FAIL: {pdf_path.name} — {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {success} converted, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
