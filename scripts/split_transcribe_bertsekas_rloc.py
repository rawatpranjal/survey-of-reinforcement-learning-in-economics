#!/usr/bin/env python3
"""
Split and transcribe Bertsekas RL & Optimal Control (Draft) PDF.
Splits into ~50-page chunks, then transcribes each via docling.
"""

import fitz  # PyMuPDF
from pathlib import Path
from docling.document_converter import DocumentConverter

# Configuration
INPUT_PDF = Path("/Users/pranjal/Code/rl/ch02_planning_learning/papers/Reinforcement learning and Optimal Control - Draft version -- Dimitri P Bertsekas -- ( WeLib.org ).pdf")
OUTPUT_DIR = Path("/Users/pranjal/Code/rl/ch02_planning_learning/papers/bertsekas_rloc_draft")
CHUNK_SIZE = 50  # pages per chunk

def split_pdf():
    """Split PDF into chunks, return list of chunk paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(INPUT_PDF)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    chunks = []
    for start in range(0, total_pages, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_pages)
        chunk_name = f"chunk_{start+1:04d}_{end:04d}.pdf"
        chunk_path = OUTPUT_DIR / chunk_name

        if chunk_path.exists():
            print(f"Skipping {chunk_name} (exists)")
        else:
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start, to_page=end-1)
            new_doc.save(chunk_path)
            new_doc.close()
            print(f"Created {chunk_name}")

        chunks.append(chunk_path)

    doc.close()
    return chunks

def transcribe_chunks(chunks):
    """Transcribe each chunk PDF to markdown using docling."""
    md_dir = OUTPUT_DIR / "md"
    md_dir.mkdir(exist_ok=True)

    converter = DocumentConverter()

    for chunk_path in chunks:
        md_path = md_dir / f"{chunk_path.stem}.md"
        if md_path.exists():
            print(f"Skipping {chunk_path.name} (already transcribed)")
            continue

        print(f"Transcribing {chunk_path.name}...")
        try:
            result = converter.convert(str(chunk_path))
            md_text = result.document.export_to_markdown()
            md_path.write_text(md_text)
            print(f"  -> {md_path.name}")
        except Exception as e:
            print(f"  ERROR: {e}")

def main():
    print("=== Splitting PDF ===")
    chunks = split_pdf()
    print(f"\n=== Transcribing {len(chunks)} chunks ===")
    transcribe_chunks(chunks)
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
